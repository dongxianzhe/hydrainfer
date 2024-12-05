import matplotlib.pyplot as plt
import transformers


from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaConfig
from dxz.utils.model_utils import count_parameters, print_parameters
from uitl import pie_chart

# model_name_or_path = "llava-hf/llava-1.5-7b-hf"
model_name_or_path = "llava-hf/llava-1.5-13b-hf"
device = torch.device('cuda:0')
dtype = torch.half

# model = LlavaForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_name_or_path, device_map=device, torch_dtype=dtype)

config = LlavaConfig.from_pretrained(model_name_or_path)
model = LlavaForConditionalGeneration(config)

print(f'model.vision_tower         : {count_parameters(model.vision_tower)}')
print(f'model.multi_modal_projector: {count_parameters(model.multi_modal_projector)}')
print(f'model.language_model       : {count_parameters(model.language_model)}')

param_count = {
    'language_model       ' : count_parameters(model.language_model)[0], 
    'vision_tower         ' : count_parameters(model.vision_tower)[0], 
    'multi_modal_projector' : count_parameters(model.multi_modal_projector)[0], 
}

pie_chart(
    data=param_count.values(), 
    labels=param_count.keys(), 
    title='Model Parameter Proportion', 
    filename=f'{model_name_or_path[9:]}.png', 
    )