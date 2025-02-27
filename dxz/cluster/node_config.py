import argparse
from typing import Optional
from dataclasses import dataclass, field, fields
from dxz.utils.config_util import CLIConfig
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.memory import TokenCacheBlockManagerConfig
from dxz.engine import RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, BatchSchedulerProfilerConfig

@dataclass
class NodeConfig(CLIConfig):
    request_processor_config: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    batch_scheduler_config: BatchSchedulerConfig = field(default_factory=BatchSchedulerConfig)
    kv_cache_config: TokenCacheBlockManagerConfig = field(default_factory=TokenCacheBlockManagerConfig)
    image_cache_config: TokenCacheBlockManagerConfig = field(default_factory=TokenCacheBlockManagerConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    batch_scheduler_profiler_config: BatchSchedulerProfilerConfig = field(default_factory=BatchSchedulerProfilerConfig)
    enable_encode: bool = True, 
    enable_prefill: bool = True, 
    enable_decode: bool = True, 
    zmq_send_url: Optional[str] = None
    debug_migrate: bool = True

    def update_shared_config(self):
        self.request_processor_config.model_factory_config = self.model_factory_config
        self.worker_config.model_factory_config = self.model_factory_config
        self.executor_config.model_factory_config = self.model_factory_config

    def update_config_value(self):
        model_factory = getModelFactory(self.model_factory_config, ModelFactoryContext())
        language_model_config = model_factory.getLanguageModelConfig()
        vision_model_config = model_factory.getVisionModelConfig()
        self.kv_cache_config.n_layers = language_model_config.n_layers
        self.kv_cache_config.n_tokens = 2
        self.kv_cache_config.n_heads = language_model_config.n_kv_heads
        self.kv_cache_config.head_size = language_model_config.head_dim
        self.kv_cache_config.dtype = self.model_factory_config.dtype
        self.kv_cache_config.device = self.model_factory_config.device
        self.image_cache_config.n_layers = 1
        self.image_cache_config.n_tokens = 1
        self.image_cache_config.block_size = vision_model_config.num_image_tokens
        self.image_cache_config.n_heads = language_model_config.n_qo_heads
        self.image_cache_config.head_size = language_model_config.head_dim
        self.image_cache_config.dtype = self.model_factory_config.dtype
        self.image_cache_config.device = self.model_factory_config.device
        self.worker_config.has_language_model = self.has_language_model
        self.worker_config.has_vision_model = self.has_vision_model

    def __post_init__(self):
        self.update_shared_config()
        self.update_config_value()

    @property
    def has_vision_model(self) -> bool:
        return self.enable_encode

    @property
    def has_language_model(self) -> bool:
        return self.enable_prefill or self.enable_decode

    @property
    def has_kv_cache(self) -> bool:
        return self.enable_prefill or self.enable_decode

    @property
    def has_image_cache(self) -> bool:
        return self.enable_encode or self.enable_prefill


    @staticmethod
    def sub_configs_from_cli_args(cls, args: argparse.Namespace, prefix: str="--") -> dict[str, 'CLIConfig']:
        return {
            'request_processor_config': RequestProcessorConfig.from_cli_args(args, prefix=prefix), 
            'model_factory_config': ModelFactoryConfig.from_cli_args(args, prefix), 
            'batch_scheduler_config': BatchSchedulerConfig.from_cli_args(args, prefix), 
            'kv_cache_config': TokenCacheBlockManagerConfig.from_cli_args(args, prefix=prefix+'kv_'), 
            'image_cache_config': TokenCacheBlockManagerConfig.from_cli_args(args, prefix=prefix+'image_'), 
            'executor_config': ExecutorConfig.from_cli_args(args, prefix=prefix), 
            'worker_config': WorkerConfig.from_cli_args(args, prefix=prefix), 
            'batch_scheduler_profiler_config': BatchSchedulerProfilerConfig.from_cli_args(args, prefix=prefix), 
        }

    @staticmethod
    def add_sub_configs_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        for sub_cls in [RequestProcessorConfig, ModelFactoryConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, BatchSchedulerProfilerConfig]:
            parser = sub_cls.add_cli_args(parser, prefix=prefix)
        parser = TokenCacheBlockManagerConfig.add_cli_args(parser, prefix=prefix+'kv-')
        parser = TokenCacheBlockManagerConfig.add_cli_args(parser, prefix=prefix+'image-')
        return parser

    @staticmethod
    def add_curr_config_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        parser.add_argument(f'{prefix}debug-migrate', action='store_true', help='Enable debug mode for request migrate.')
        return parser