import io
import time
import base64
import threading
from PIL import Image
from torch import Tensor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import Optional
import argparse

from dxz.request.request import Request
from dxz.request.rcb import RequestControlBlock, OutputTokenProcessor
from dxz.request.request_processor import RequestProcessor, RequestProcessorConfig, RequestProcessorContext
from dxz.engine.engine import EngineConfig, Engine, EngineContext
from dxz.model.model_factory import ModelFactory, getModelFactory, ModelFactoryConfig, ModelFactoryContext


from dxz.request.request_processor import RequestProcessor, RequestProcessorConfig
from dxz.engine.isa import ImageEmbed, ImageEmbedFill, ImageFill, TextFill, InstructionListBuilder
class SingleNodeRequestProcessor(RequestProcessor):
    def __init__(self, config: RequestProcessorConfig, context: RequestProcessorContext):
        super().__init__()
        self.config = config
        self.context = context
        self.tokenizer = context.tokenizer
        self.processor = context.processor

    def insert_image_tokens(self, token_ids: list[int], num_image_tokens):
        # replace each image_token_id with num_image_tokens image_token_id
        inserted_token_ids: list[int] = []
        for token_id in token_ids:
            if token_id == self.context.image_token_id:
                inserted_token_ids.extend([self.context.image_token_id] * (num_image_tokens - 1))
            inserted_token_ids.append(token_id)
        return inserted_token_ids 

    def process(self, request: Request) -> RequestControlBlock:
        # 1. images
        image: Optional[Image.Image] = None
        images_tensor: Optional[Tensor] = None # (n_images, n_channels, width, height)
        if request.image_base64 is not None:
            image = Image.open(io.BytesIO(base64.b64decode(request.image_base64)))
        if image is None and request.image:
            image = request.image
        if image is not None:
            images_tensor = self.processor(
                text="", 
                images = image, 
                return_tensors="pt"
            )['pixel_values']
        n_pixel_values_images = images_tensor.shape[0] if images_tensor is not None else 0
        # 2. token_ids
        token_ids = self.tokenizer.encode(request.prompt)
        n_token_ids_images = token_ids.count(self.context.image_token_id)
        assert n_token_ids_images == n_pixel_values_images, f"image number is not equal between text and image list {n_token_ids_images} {n_pixel_values_images}"
        token_ids = self.insert_image_tokens(token_ids, self.context.num_image_tokens)
        n_prompt_tokens = len(token_ids)
        token_ids = token_ids + [-1] * (request.sampling_params.max_tokens - 1) # -1 will be set when executing
        # 3. image_overwrite_mask
        image_overwrite_mask = [token_id == self.context.image_token_id for token_id in token_ids]
        # 4. position_ids
        position_ids = list(range(len(token_ids)))
        # 5. cache_ids
        n_virtual_kv_caches: int = self.context.n_layers
        layer_virtual_kv_cache_ids = list(range(self.context.n_layers))
        cache_ids = list(range(len(token_ids)))
        # 6. instruction list
        builder = InstructionListBuilder()
        if images_tensor is not None:
            if self.config.disaggregate_embed_prefill:
                embed = ImageEmbed(
                    pixel_values = images_tensor,
                    image_features_dst = None,
                    token_pruning_params = None, 
                )
                prefill = ImageEmbedFill(
                    image_features = None, 
                    token_ids = token_ids[:n_prompt_tokens], 
                    position_ids = position_ids[:n_prompt_tokens], 
                    cache_ids = [cache_ids[:n_prompt_tokens] for _ in range(self.context.n_layers)], 
                    kv_cache_ids = layer_virtual_kv_cache_ids, 
                    sample = True, 
                    sample_dst = None, 
                )
                embed.image_features_dst = prefill
                builder.append(embed)
                builder.append(prefill)
            else:
                prefill = ImageFill(
                    pixel_values = images_tensor, 
                    token_ids = token_ids[:n_prompt_tokens], 
                    position_ids = position_ids[:n_prompt_tokens], 
                    cache_ids = [cache_ids[:n_prompt_tokens] for _ in range(self.context.n_layers)], 
                    kv_cache_ids = layer_virtual_kv_cache_ids, 
                    sample = True, 
                    sample_dst = None, 
                )
                builder.append(prefill)
        else:
            prefill = TextFill(
                token_ids = token_ids[:n_prompt_tokens], 
                position_ids = position_ids[:n_prompt_tokens], 
                cache_ids = [cache_ids[:n_prompt_tokens] for _ in range(self.context.n_layers)], 
                kv_cache_ids = layer_virtual_kv_cache_ids, 
                sample = True, 
                sample_dst = None, 
            )
            builder.append(prefill)

        last_inst = prefill
        left = n_prompt_tokens     
        while left + 1 <= len(token_ids):
            right = left + 1
            decode = TextFill(
                token_ids = token_ids[left:right], 
                position_ids = position_ids[left:right], 
                cache_ids = [cache_ids[left:right] for _ in range(self.context.n_layers)], 
                kv_cache_ids = layer_virtual_kv_cache_ids, 
                sample = True, 
                sample_dst = None, 
            )
            builder.append(decode)
            last_inst.sample_dst = decode
            last_inst = decode
            left = right

        instructions = builder.build_instruction_list()
        return RequestControlBlock(
            instructions = instructions, 
            n_virtual_kv_caches = n_virtual_kv_caches, 
            sampling_params = request.sampling_params, 
            output_token_processor = None
        )


@dataclass
class EPDNodeConfig:
    multi_thread_request_process: bool = False
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    request_processor_config: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    engine_config: EngineConfig = field(default_factory=EngineConfig)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EPDNodeConfig':
        attrs = [attr.name for attr in fields(cls) if attr.name not in ['model_factory_config', 'request_processor_config', 'engine_config']]
        model_factory_config = ModelFactoryConfig.from_cli_args(args)
        request_processor_config = RequestProcessorConfig.from_cli_args(args)
        engine_config = EngineConfig.from_cli_args(args)
        config = cls(model_factory_config=model_factory_config, request_processor_config=request_processor_config, engine_config=engine_config, **{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--multi-thread-request-process', action='store_true', help='Enable multi-threading for request processing.')
        parser = ModelFactoryConfig.add_cli_args(parser)
        parser = RequestProcessorConfig.add_cli_args(parser)
        parser = EngineConfig.add_cli_args(parser)
        return parser


class EPDNode:
    def __init__(self, config: EPDNodeConfig):
        self.config = config
        self.engine = Engine(self.config.engine_config, EngineContext(model_factory_config=config.model_factory_config))
        model_factory = getModelFactory(config.model_factory_config, ModelFactoryContext(process_group=None))
        self.vision_model_config = model_factory.getVisionModelConfig()
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.processor = model_factory.getProcessor() 
        self.tokenizer = model_factory.getTokenizer() 

        self.request_processor_context = RequestProcessorContext(
            tokenizer = self.tokenizer, 
            processor = self.processor, 
            image_token_id = self.vision_model_config.image_token_id, 
            num_image_tokens = self.vision_model_config.num_image_tokens, 
            n_layers = self.language_model_config.n_layers,
        )
        self.processor = SingleNodeRequestProcessor(
            config = self.config.request_processor_config, 
            context = self.request_processor_context, 
        )

        self.add_request_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=32)

    def add_request(self, request: Request, output_processor: OutputTokenProcessor):
        if self.config.multi_thread_request_process:
            self.executor.map(self._add_request_async, [(request, output_processor)])
        else:
            self._add_request(request, output_processor)

    def _add_request_async(self, request: Request, output_processor: OutputTokenProcessor):
        with self.add_request_lock:
            self._add_request(request, output_processor)
    
    def _add_request(self, request: Request, output_processor: OutputTokenProcessor):
        arrival_time = time.perf_counter()
        rcb = self.processor.process(request=request)
        rcb.metric.arrival_time = arrival_time
        rcb.output_token_processor = output_processor
        self.engine.schedule([rcb])

    def step(self):
        self.engine.step()