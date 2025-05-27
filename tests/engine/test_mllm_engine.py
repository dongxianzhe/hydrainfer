from PIL import Image
from hydrainfer.engine.mllm_engine import MLLMEngine, MLLMEngineConfig
from hydrainfer.utils.profiler import profile

def test_llava_engine():
    # 1. prepare input
    image_path = f'./dataset/cherry_blossom.jpg'
    image = Image.open(image_path)
    question = "What is the content of this image?"
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    num_prompts = 3
    inputs = [{
        "prompt" : prompt, 
        "multi_modal_data":{
            "image": image
        },
        "max_tokens": 50,
    } for _ in range(num_prompts)]
    # 2. create engine
    config = MLLMEngineConfig(batch_policy='nobatch')
    engine = MLLMEngine(config)
    # 3. offline inference
    with profile('engine.generate'):
        output_texts = engine.generate(inputs)
    for output_text in output_texts:
        print(output_text)
        print('==================================================')

if __name__ == '__main__':
    test_llava_engine()