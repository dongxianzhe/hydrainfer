import time
import torch
import argparse
from dataclasses import dataclass, field, fields
from dxz.request import Request
from dxz.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.engine import RequestProcessor, RequestProcessorConfig, RequestProcessorContext, ImageEmbed, Fill, EmptyInstruction, BatchScheduler, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, getWorker, WorkerContext, ExecutorContext, InstructionExecutor, Engine, BatchRequest, RequestProcessParameters
from dxz.memory import MemoryConfig, getMemoryManagementUnit, MemoryContext, MemoryManagementUnit


@dataclass
class EPDNodeConfig:
    request_processor_config: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    batch_scheduler_config: BatchSchedulerConfig = field(default_factory=BatchSchedulerConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EPDNodeConfig':
        attrs = [attr.name for attr in fields(cls) if attr.name not in ['request_processor_config', 'model_factory_config', 'memory_config', 'batch_scheduler_config', 'executor_config', 'worker_config',]]
        model_factory_config = ModelFactoryConfig.from_cli_args(args)
        request_processor_config = RequestProcessorConfig.from_cli_args(args)
        memory_config = MemoryConfig.from_cli_args(args)
        batch_scheduler_config = BatchSchedulerConfig.from_cli_args(args)
        executor_config = ExecutorConfig.from_cli_args(args)
        worker_config = WorkerConfig.from_cli_args(args)
        config = cls(
            request_processor_config = request_processor_config, 
            model_factory_config     = model_factory_config, 
            memory_config            = memory_config, 
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
        parser = MemoryConfig.add_cli_args(parser)
        parser = BatchSchedulerConfig.from_cli_args(parser)
        parser = ExecutorConfig.from_cli_args(parser)
        parser = WorkerConfig.from_cli_args(parser)
        return parser


@dataclass
class EPDNodeContext:
    pass


class EPDNode(Engine):
    def __init__(self, config: EPDNodeConfig, context: EPDNodeContext):
        self.config = config
        model_factory = getModelFactory(config.model_factory_config, ModelFactoryContext())
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.vision_model_config = model_factory.getVisionModelConfig()
        self.processor = model_factory.getProcessor() 
        self.tokenizer = model_factory.getTokenizer() 
        self.mmu = getMemoryManagementUnit(self.config.memory_config, MemoryContext(
                n_layers = self.language_model_config.n_layers,
                head_size = self.language_model_config.head_dim, 
                num_kv_heads = self.language_model_config.n_kv_heads, 
                dtype = config.model_factory_config.dtype, 
                device = config.model_factory_config.device, 
            )
        )
        # self.image_token_mmu = getMemoryManagementUnit(
        #     config = MemoryConfig(
        #         memory_management_policy = 'vanilla', 
        #         num_blocks = 30, 
        #         block_size = vision_model_config.num_image_tokens, 
        #     ), 
        #     context = MemoryContext(
        #         n_layers = 1, 
        #         num_kv_heads = language_model_config.n_qo_heads, 
        #         head_size = language_model_config.head_dim, 
        #         dtype = context.model_factory_config.dtype, 
        #         device = context.model_factory_config.device, 
        #         n_tokens=1, 
        #     )
        # )
        # scheduler
        self.batch_scheduler = BatchScheduler(self.config.batch_scheduler_config)
        # executor
        self.worker = getWorker(config.worker_config, WorkerContext(model_factory_config=config.model_factory_config))
        context.worker = self.worker
        self.executor = InstructionExecutor(config.executor_config, ExecutorContext(
                model_factory_config = config.model_factory_config, 
                block_size = config.memory_config.block_size, 
                mmu = self.mmu, 
                image_token_mmu=None, 
                worker = self.worker, 
                dtype = config.model_factory_config.dtype, 
                device = config.model_factory_config.device, 
            ), 
        )
        self.request_processor = RequestProcessor(
            config = self.config.request_processor_config, 
            context = RequestProcessorContext(
                tokenizer = self.tokenizer, 
                processor = self.processor, 
                image_token_id = self.vision_model_config.image_token_id, 
                num_image_tokens = self.vision_model_config.num_image_tokens, 
                n_layers = self.language_model_config.n_layers,
                batch_scheduler=self.batch_scheduler, 
            ), 
        )

    def add_request(self, request: Request, params: RequestProcessParameters):
        self.request_processor.process(request, params)

    @torch.inference_mode()
    def step(self):
        # 1. schedule requests
        batch = self.batch_scheduler.step()
        if len(batch) == 0:
            return

        # 2. execute instructions
        batch_fill = BatchRequest()
        batch_image_embed = BatchRequest()
        batch_empty = BatchRequest()
        for rcb, inst in batch:
            if isinstance(inst, Fill):
                batch_fill.append(rcb)
                continue
            if isinstance(inst, EmptyInstruction):
                batch_empty.append(rcb)
                continue
            if isinstance(inst, ImageEmbed):
                batch_image_embed.append(rcb)
                continue
            raise Exception(f'unsupported instrction {type(inst)}')

        future = self.executor.execute_image_embed(batch_image_embed)
        self.executor.execute_fill(batch_fill)
        if future is not None:
            future.result()
        self.executor.execute_empty(batch_empty)

        # 3. scheduler requests
        t = time.perf_counter()
        for rcb, _ in batch:
            if rcb.is_finished():
                rcb.metric.finished_time = t
                for vkvc in rcb.virtual_kv_caches:
                    self.mmu.realloc(vkvc, 0)
            else:
                self.batch_scheduler.schedule_running(rcb)

if __name__ == '__main__':
    from dxz.request import Request, SamplingParameters
    from dxz.engine import PrintOutputTokenProcessor

    config = EPDNodeConfig()
    config.batch_scheduler_config.debug_mode = True
    config.model_factory_config.model_name = "llava-hf/llava-1.5-7b-hf"
    node = EPDNode(config, EPDNodeContext())
    request = Request(
        request_id = 0, 
        prompt = f"what's the weather like today?", 
        image = None, 
        image_base64 = None, 
        sampling_params = SamplingParameters(max_tokens=10)
    )
    node.add_request(request, RequestProcessParameters(
        output_token_processors = [], 
        print_output_text=True, 
    ))
    for i in range(10):
        node.step()