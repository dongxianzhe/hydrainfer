import torch
from torch import Tensor
import pytest
from hydrainfer.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext, ImageProcessor, VisionModelParameters, VisionModelOutput
from hydrainfer.utils.torch_utils import str2device, str2dtype
from PIL import Image
import numpy as np


@pytest.mark.parametrize('path', [
    "/models/llava-1.5-7b-hf", 
    "/models/llava-v1.6-vicuna-7b-hf", 
    "/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024", 
    "/models/deepseek-vl2-tiny", 
    "/models/OpenGVLab/InternVL2-26B", 
])
def test_vision_model(path: str):
    str_device = 'cuda:0'
    str_dtype = 'fp16'
    device = str2device(str_device)
    dtype = str2dtype(str_dtype)
    print(f'============================================================')
    print(f'model {path}')
    factory = getModelFactory(ModelFactoryConfig(path=path, device=str_device, dtype=str_dtype), ModelFactoryContext())
    vision_config = factory.getVisionModelConfig()
    processor: ImageProcessor = factory.getProcessor()
    vision_model = factory.getVisionModel()

    height, width, n_channel = 1024, 1024, 3
    print(f'image info height {height} width {width} n_channel {n_channel}')
    random_array = np.random.randint(0, 256, (height, width, n_channel), dtype=np.uint8)
    image = Image.fromarray(random_array)

    images_tensor: Tensor = processor.process(image).to(device).to(dtype)
    print(f'images_tensor.shape {images_tensor.shape}')
    pixel_values = [images_tensor]
    vision_output = vision_model.forward(
        pixel_values=pixel_values, 
        model_params=VisionModelParameters(
            return_last_layer_attention=False, 
            original_image_sizes=[(height, width)]
        )
    )
    num_image_token_expected = vision_config.image_token_caculator.get_num_image_tokens((height, width))
    print(f'vision_output.image_features.shape {vision_output.image_features.shape}')
    print(f'number of image tokens expected {num_image_token_expected}')
    # breakpoint()
    assert vision_output.image_features.shape[0] == num_image_token_expected


if __name__ == '__main__':
    pytest.main([__file__, '-s'])