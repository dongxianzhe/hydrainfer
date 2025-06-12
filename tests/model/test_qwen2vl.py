import torch
from hydrainfer.model import ModelFactoryConfig, ModelFactoryContext, VisionModelParameters
from hydrainfer.model.qwen2_vl import Qwen2VLModelFactory

if __name__ == '__main__':
    device = torch.device('cuda:0')
    config = ModelFactoryConfig(
        path = "/mnt/cfs/9n-das-admin/llm_models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024", 
        dtype = "fp16", 
        device = "cuda:0", 
    )
    context = ModelFactoryContext()
    factory = Qwen2VLModelFactory(config, context)
    
    vision_config = factory.getVisionModelConfig()
    language_config = factory.getLanguageModelConfig()
    tokenizer = factory.getTokenizer()
    print(f'vision_config {vision_config}')
    print(f'language_config {language_config}')

    from PIL import Image
    import numpy as np
    height, width, n_channel = 1024, 1024, 3
    random_array = np.random.randint(0, 256, (height, width, n_channel), dtype=np.uint8)
    image = Image.fromarray(random_array)
    processor = factory.getProcessor()
    images_tensor = processor(
        text="", 
        images = image, 
        return_tensors="pt"
    )['pixel_values']
    print(f'images_tensor.shape {images_tensor.shape}')
    images_tensor = images_tensor.to(torch.half)
    images_tensor = images_tensor.to(device)
    width, height = image.size
    image_size = (height, width)
    print(f'image_size {image_size}')
    n_image_tokens = vision_config.image_token_caculator.get_num_image_tokens(image_size)
    print(f'n_image_tokens {n_image_tokens}')
    vision_model = factory.getVisionModel()

    pixel_values = [images_tensor]
    vision_model_params = VisionModelParameters(
        return_last_layer_attention = False, 
        original_image_sizes = [image_size], 
    )
    vision_output = vision_model.forward(
        pixel_values=pixel_values, 
        model_params=vision_model_params
    )
    image_features = vision_output.image_features
    print(f'image_features.shape {image_features.shape}')