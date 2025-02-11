import time
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass, field, fields
from dxz.request import Request, OfflineInferenceOutput
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.engine import RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, OfflineOutputTokenProcessor, RequestProcessParameters
from dxz.cluster.epdnode import EPDNode, NodeConfig
from dxz.utils.counter import Counter
from dxz.utils.config_util import CLIConfig


@dataclass
class OfflineSingleInstanceEntryPointConfig(CLIConfig):
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    epdnode_config: NodeConfig = field(default_factory=NodeConfig)

    def update_shared_config(self):
        self.epdnode_config.model_factory_config = self.model_factory_config
        self.epdnode_config.update_shared_config()

    def update_config_value(self):
        self.epdnode_config.update_config_value()

    def __post_init__(self):
        self.update_shared_config()
        self.update_config_value()

class OfflineSingleInstanceEntryPoint:
    def __init__(self, config: OfflineSingleInstanceEntryPointConfig):
        factory = getModelFactory(config.model_factory_config, ModelFactoryContext(process_group=None))
        self.tokenizer = factory.getTokenizer() 
        self.node = EPDNode(config.epdnode_config)

    def generate(self, requests: list[Request]) -> list[OfflineInferenceOutput]:
        finished = Counter()
        bar = tqdm(range(len(requests)))
        arrival_time = time.perf_counter()
        outputs: list[OfflineInferenceOutput] = [OfflineInferenceOutput(arrival_time=arrival_time) for _ in range(len(requests))]
        for output, request in zip(outputs, requests):
            self.node.add_request(request, RequestProcessParameters(
                output_token_processors=[OfflineOutputTokenProcessor(output, self.tokenizer, finished, bar)],
            ))
        
        while finished.value() < len(requests):
            self.node.step()

        return outputs