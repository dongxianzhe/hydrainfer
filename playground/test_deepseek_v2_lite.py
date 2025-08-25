# from vllm import LLM, SamplingParams

# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# llm = LLM(model="/mnt/cfs/9n-das-admin/llm_models/DeepSeek-V2-Lite")

# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "/mnt/cfs/9n-das-admin/llm_models/DeepSeek-V2-Lite"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")

old_forward = model.forward
def monkey_patch(*args, **kward):
    breakpoint()
    return old_forward(*args, **kward)
model.forward = monkey_patch

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
