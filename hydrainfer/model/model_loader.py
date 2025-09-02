import os
import json
import dacite
import safetensors.torch
from torch import nn
from typing import TypeVar, Type, Optional, Union
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)


def load_safetensor(model_with_prefix_list: list[tuple[nn.Module, str]], param_with_name_list: list[tuple[nn.Parameter, str]], model_weights_path: str):
    # model_with_prefix_list: a list of elements, each element is model and its model params tensor's name's prefix in safetensor file
    # param_with_name_list: a list of elements, each element is a param and it's name in safetensor file
    state_dicts = [model.state_dict() for model, prefix in model_with_prefix_list]
    loaded_sets = [set() for _ in range(len(state_dicts))]
    loaded_params_set = set()
    for entry in os.scandir(model_weights_path):
        if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
            logger.info(f'load safetensor from {entry.path}')
            for name, weight in safetensors.torch.load_file(entry.path).items():
                loaded: bool = False
                for i, (model, prefix) in enumerate(model_with_prefix_list):
                    if name.startswith(prefix):
                        name_removed_prefix = name.removeprefix(prefix)
                        if name_removed_prefix in state_dicts[i]:
                            state_dicts[i][name_removed_prefix].data.copy_(weight)
                            loaded_sets[i].add(name_removed_prefix)
                            loaded = True
                            break
                if loaded:
                    continue
                for param, param_name in param_with_name_list:
                    if param_name == name:
                        param.copy_(weight)
                        loaded_params_set.add(name)
                        loaded = True
                        break
                if loaded:
                    continue
    for state_dict, loaded_set in zip(state_dicts, loaded_sets):
        assert len(state_dict) == len(loaded_set), f'expected load {len(state_dict)} tensors, but only {len(loaded_set)} tensors are loaded'
    assert len(loaded_params_set) == len(param_with_name_list), f'expected load {len(param_with_name_list)} tensors, but only {len(loaded_params_set)} tensors are loaded'


T = TypeVar("T")
def load_config_from_json(config_class: Type[T], model_weights_path: str) -> T:
    with open(os.path.join(model_weights_path, 'config.json'), "r") as file:
        json_config = json.load(file)
    config = dacite.from_dict(data_class=config_class, data=json_config)
    return config
