import pytest
from hydrainfer.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext


@pytest.mark.parametrize('path', [
    "/models/llava-1.5-7b-hf", 
    "/models/llava-v1.6-vicuna-7b-hf", 
    "/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024", 
    "/models/deepseek-vl2-tiny", 
    "/models/OpenGVLab/InternVL2-26B", 
])
def test_profiler(path: str):
    factory = getModelFactory(ModelFactoryConfig(path=path), ModelFactoryContext())
    profiler = factory.getModelProfiler()
    params_config = profiler.profile_model_params()
    print(f'path {path} {params_config}')


if __name__ == '__main__':
    pytest.main([__file__, '-s'])