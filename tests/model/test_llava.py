from transformers import LlavaConfig
from dxz.model.llava import LlavaForConditionalGeneration
from dxz.utils.model_utils import count_parameters

# 1. test number of params
config = LlavaConfig.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration(config)

assert count_parameters(model) == (7063427072, 7063427072), f"wrong total model params {count_parameters(model)}"

# 2. test forward correctness
print('pass')