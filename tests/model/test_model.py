import torch
from dxz.engine.engine import EngineConfig
from dxz.request.request import Request, SamplingParameters
from dxz.entrypoint.mllm import MLLM, MLLMConfig
from dxz.cluster.epdnode import EPDNodeConfig
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

    config = MLLMConfig(epdnode_config=EPDNodeConfig(engine_config=EngineConfig(model_name=model_name)))
    mllm = MLLM(config)

    outputs = mllm.generate([request])
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