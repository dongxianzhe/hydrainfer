import pytest
from hydrainfer.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext
from hydrainfer.model.model_profiler import VisionLanguageModelProfiler


@pytest.mark.parametrize('path', [
    "/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf", 
    "/mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf", 
    "/mnt/cfs/9n-das-admin/llm_models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024", 
    "/mnt/cfs/9n-das-admin/llm_models/deepseek-vl2-tiny", 
])
def test_profiler(
    path: str
):
    factory = getModelFactory(ModelFactoryConfig(path=path), ModelFactoryContext())
    profiler = factory.getModelProfiler()
    params_config = profiler.profile_model_params()
    print(f'path {path} {params_config}')


if __name__ == '__main__':
    pytest.main([__file__, '-s'])