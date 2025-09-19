import pytest
from hydrainfer.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext, Tokenizer


@pytest.mark.parametrize('path', [
    "/models/llava-1.5-7b-hf", 
    "/models/llava-v1.6-vicuna-7b-hf", 
    "/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024", 
    "/models/deepseek-vl2-tiny", 
    "/models/OpenGVLab/InternVL2-26B", 
])
def test_tokenizer(path: str):
    print(f'============================================================')
    print(f'model: {path}')
    factory = getModelFactory(ModelFactoryConfig(path=path), ModelFactoryContext())

    vision_config = factory.getVisionModelConfig()
    tokenizer: Tokenizer = factory.getTokenizer()

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": f"{vision_config.image_token}\nWhat's the content of this image?"},
    ]

    prompt = tokenizer.apply_chat_template(messages)
    print(f'prompt: {prompt}')
    token_ids = tokenizer.encode(prompt)
    print(f'token_ids: {token_ids}')

    text = ""
    for token_id in token_ids:
        text += tokenizer.decode(token_id)
    print(f'text: {text}')


if __name__ == '__main__':
    pytest.main([__file__, '-s'])