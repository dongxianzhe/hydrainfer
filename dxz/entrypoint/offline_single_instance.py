import time
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass, field, fields
from dxz.request import Request, OfflineInferenceOutput
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.engine import RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, OfflineOutputTokenProcessor, RequestProcessParameters
from dxz.cluster.epdnode import EPDNode
from dxz.utils.counter import Counter


@dataclass
class OfflineSingleInstanceEntryPointConfig:
    request_processor_config: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    batch_scheduler_config: BatchSchedulerConfig = field(default_factory=BatchSchedulerConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    n_kv_blocks: int = 512
    n_image_blocks: int = 16

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'OfflineSingleInstanceEntryPointConfig':
        attrs = [attr.name for attr in fields(cls) if attr.name not in ['request_processor_config', 'model_factory_config', 'batch_scheduler_config', 'executor_config', 'worker_config',]]
        model_factory_config = ModelFactoryConfig.from_cli_args(args)
        request_processor_config = RequestProcessorConfig.from_cli_args(args)
        batch_scheduler_config = BatchSchedulerConfig.from_cli_args(args)
        executor_config = ExecutorConfig.from_cli_args(args)
        worker_config = WorkerConfig.from_cli_args(args)
        config = cls(
            request_processor_config = request_processor_config, 
            model_factory_config     = model_factory_config, 
            batch_scheduler_config   = batch_scheduler_config, 
            executor_config          = executor_config, 
            worker_config            = worker_config, 
            **{attr: getattr(args, attr) for attr in attrs}
        )
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = ModelFactoryConfig.add_cli_args(parser)
        parser = RequestProcessorConfig.add_cli_args(parser)
        parser = BatchSchedulerConfig.add_cli_args(parser)
        parser = ExecutorConfig.add_cli_args(parser)
        parser = WorkerConfig.add_cli_args(parser)
        parser.add_argument('--n-kv-blocks', type=int, default=512, help='number kv cache blocks')
        parser.add_argument('--n-image-blocks', type=int, default=16, help='number image cache blocks')
        return parser


class OfflineSingleInstanceEntryPoint:
    def __init__(self, config: OfflineSingleInstanceEntryPointConfig):
        factory = getModelFactory(config.model_factory_config, ModelFactoryContext(process_group=None))
        self.tokenizer = factory.getTokenizer() 
        self.node = EPDNode(
            request_processor_config = config.request_processor_config, 
            model_factory_config     = config.model_factory_config, 
            batch_scheduler_config   = config.batch_scheduler_config, 
            executor_config          = config.executor_config, 
            worker_config            = config.worker_config, 
            n_kv_blocks              = config.n_kv_blocks, 
            n_image_blocks           = config.n_image_blocks, 
        )

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