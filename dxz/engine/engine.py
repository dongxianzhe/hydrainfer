from typing import Optional
from dxz.request import Request
from dxz.engine import RequestControlBlock, RequestProcessParameters, RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, BatchScheduler, getWorker, WorkerContext, InstructionExecutor, ExecutorContext, RequestProcessor, RequestProcessorContext, Worker
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.memory import TokenCacheBlockManager, TokenCacheBlockManagerContext


class Engine:
    def add_request(self, request: Request, params: RequestProcessParameters):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class AsyncEngine:
    async def add_request(self, request: Request, params: RequestProcessParameters):
        raise NotImplementedError

    async def step(self):
        raise NotImplementedError

    async def step_loop(self):
        raise NotImplementedError

    async def register_node(self, node: "AsyncEngine"): 
        raise NotImplementedError

    async def migrate(self, rcb: RequestControlBlock):
        raise NotImplementedError


class EngineComponentFactory:
    def __init__(self,
        request_processor_config: RequestProcessorConfig, 
        model_factory_config: ModelFactoryConfig, 
        batch_scheduler_config: BatchSchedulerConfig, 
        executor_config: ExecutorConfig, 
        worker_config: WorkerConfig, 
        n_kv_blocks: int = 512, 
        n_image_blocks: int = 16, 
    ):
        self.request_processor_config = request_processor_config
        self.model_factory_config     = model_factory_config
        self.batch_scheduler_config   = batch_scheduler_config
        self.executor_config          = executor_config
        self.worker_config            = worker_config
        self.n_kv_blocks              = n_kv_blocks
        self.n_image_blocks           = n_image_blocks

        self.kv_block_size          = 16
        model_factory = getModelFactory(self.model_factory_config, ModelFactoryContext())
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.vision_model_config = model_factory.getVisionModelConfig()
        self.processor = model_factory.getProcessor() 
        self.tokenizer = model_factory.getTokenizer() 

    def get_kv_cache_block_manager(self) -> TokenCacheBlockManager:
        return TokenCacheBlockManager(TokenCacheBlockManagerContext(
            n_layers = self.language_model_config.n_layers,
            n_tokens = 2, 
            n_blocks = self.n_kv_blocks, 
            block_size = self.kv_block_size, 
            n_heads = self.language_model_config.n_kv_heads, 
            head_size = self.language_model_config.head_dim, 
            dtype = self.model_factory_config.dtype, 
            device = self.model_factory_config.device, 
        ))

    def get_image_cache_block_manager(self) -> TokenCacheBlockManager:
        return TokenCacheBlockManager(TokenCacheBlockManagerContext(
            n_layers = 1,
            n_tokens = 1, 
            n_blocks = self.n_image_blocks, 
            block_size = self.vision_model_config.num_image_tokens, 
            n_heads = self.language_model_config.n_qo_heads, 
            head_size = self.language_model_config.head_dim, 
            dtype = self.model_factory_config.dtype, 
            device = self.model_factory_config.device, 
        ))

    def get_batch_scheduler(self) -> BatchScheduler: 
        return BatchScheduler(self.batch_scheduler_config)

    def get_executor(self, kv_cache_block_manager: Optional[TokenCacheBlockManager], image_cache_block_manager: Optional[TokenCacheBlockManager]) -> InstructionExecutor:
        worker = getWorker(self.worker_config, WorkerContext(model_factory_config=self.model_factory_config, has_language_model=kv_cache_block_manager is not None, has_vision_model=image_cache_block_manager is not None))
        return InstructionExecutor(self.executor_config, ExecutorContext(
                model_factory_config = self.model_factory_config, 
                block_size = self.kv_block_size, 
                kv_cache_block_manager = kv_cache_block_manager, 
                image_cache_block_manager = image_cache_block_manager, 
                worker = worker, 
                dtype = self.model_factory_config.dtype, 
                device = self.model_factory_config.device, 
            ), 
        )

    def get_request_processor(self, batch_scheduler: BatchScheduler) -> RequestProcessor:
        return RequestProcessor(
            config = self.request_processor_config, 
            context = RequestProcessorContext(
                tokenizer = self.tokenizer, 
                processor = self.processor, 
                image_token_id = self.vision_model_config.image_token_id, 
                num_image_tokens = self.vision_model_config.num_image_tokens, 
                n_layers = self.language_model_config.n_layers,
                batch_scheduler=batch_scheduler, 
                ep_migrate = True, 
                pd_migrate = False, 
            ), 
        )