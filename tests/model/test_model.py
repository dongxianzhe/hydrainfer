import torch
from dxz.request import Request, SamplingParameters
from dxz.entrypoint import OfflineSingleInstanceEntryPoint, OfflineSingleInstanceEntryPointConfig
from transformers import LlamaForCausalLM as LlamaForCausalLMRef
from transformers import AutoTokenizer


model_name = 'meta-llama/Llama-2-7b-hf'
prompt = "who are you?"
def transformer_generate():
    dtype = torch.half
    device = torch.device('cuda:0')
    model = LlamaForCausalLMRef.from_pretrained(model_name)
    model.to(dtype)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    inputs = tokenizer(prompt, return_tensors="pt")

    output_tokens = model.generate(inputs.input_ids.to(device), max_length=20)[0].tolist()
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print(output_text)

def engine_generate():
    request = Request(
        request_id = 0, 
        prompt = prompt, 
        image = None, 
        image_base64 = None, 
        sampling_params = SamplingParameters(max_tokens=20)
    )
    config = OfflineSingleInstanceEntryPointConfig()
    config.model_factory_config.model_name='meta-llama/Llama-2-7b-hf'
    config.epdnode_config.kv_cache_config.n_blocks=512
    config.epdnode_config.image_cache_config.n_blocks=16
    config.update_config_value()

    entrypoint = OfflineSingleInstanceEntryPoint(config)
    outputs = entrypoint.generate([request])
    for output in outputs:
        print(output.text)


# you can test vllm generate by the following commnad in command line
# vllm serve llava-hf/llava-1.5-7b-hf --port=8888
# curl -i -X POST http://localhost:8888/v1/completions -H "Content-Type: application/json" -d '{"model": "llava-hf/llava-1.5-7b-hf", "prompt":"who are you?", "max_tokens": 20, "temperature": 0}'

if __name__ == '__main__':
    print('----------------------------- transformer -------------------------------')
    # transformer_generate()
    print('-----------------------------     our     -------------------------------')
    engine_generate()