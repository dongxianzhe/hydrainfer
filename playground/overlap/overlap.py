# from hydrainfer.model import ModelFactory, ModelFactoryConfig, ModelFactoryContext
# config = ModelFactoryConfig(
#     path="/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf", 
# )
# context = ModelFactoryContext()
# factory = ModelFactory(config, context)
# model = factory.getLanguageModel()

import torch
from torch import nn
import time

if __name__ == '__main__':
    model = nn.Linear(1024, 1024 * 1024, bias=False)

    start = time.time()
    model.to(torch.device('cuda:0'))
    end = time.time()
    print(f'dur {end - start}')