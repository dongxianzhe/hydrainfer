import os
import safetensors.torch
from dataclasses import dataclass
from hydrainfer.utils.logger import getLogger
from model_factory import ModelProfiler, ModelParamsConfig
logger = getLogger(__name__)


def profile_model_params_from_safetensor(model_path: str, prefixes: list[str]) -> dict[str, int]:
    # counter's key is tensor prefix, value is number of params with this prefix
    counter: dict[str, int] = {prefix:0 for prefix in prefixes}
    for entry in os.scandir(model_path):
        if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
            for name, weight in safetensors.torch.load_file(entry.path).items():
                for prefix in prefixes: # todo optimize with trie tree
                    if name.startswith(prefix):
                        counter[prefix] += weight.numel()
    return counter


class VisionLanguageModelProfiler(ModelProfiler):
    def __init__(self, path: str, vision_model_prefixes: list[str], language_model_prefixes: list[str]):
        self.path = path
        self.vision_model_prefixes = vision_model_prefixes
        self.language_model_prefixes = language_model_prefixes

    def profile_model_params(self) -> ModelParamsConfig:
        prefixes: list[str] = self.vision_model_prefixes + self.language_model_prefixes
        counter = profile_model_params_from_safetensor(self.path, prefixes)
        config = ModelParamsConfig(
            vision_model_params = sum([counter[prefix] for prefix in self.vision_model_prefixes]), 
            language_model_parmas = sum([counter[prefix] for prefix in self.language_model_prefixes])
        )
        return config