import threading
import io
import base64
import random
import argparse
from torch import Tensor
from dataclasses import dataclass, fields
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from typing import Literal, Optional
from dxz.request.request import Request
from dxz.engine import Instruction, TextFill, ImageFill, EmptyInstruction, ImageEmbed, ImageEmbedFill, InstructionList, InstructionListBuilder, MigrateRequest, RequestControlBlock, OutputTokenProcessor, BatchScheduler, PrintTextOutputTokenProcessor


@dataclass
class RequestProcessorConfig:
    multi_thread_request_process: bool = False
    disaggregate_embed_prefill: bool = True

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'RequestProcessorConfig':
        attrs = [attr.name for attr in fields(cls)]
        config = cls(**{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--multi-thread-request-process', action='store_true', help='Enable multi-threading for request processing.')
        parser.add_argument('--disaggregate-embed-prefill', action='store_true', help='Enable disaggregation of embedding prefill.')
        return parser


@dataclass
class RequestProcessorContext:
    tokenizer: AutoTokenizer
    processor: AutoProcessor
    image_token_id: int
    num_image_tokens: int
    n_layers: int
    ep_migrate: bool = False
    pd_migrate: bool = False
    batch_scheduler: BatchScheduler = None


@dataclass
class RequestProcessParameters:
    output_token_processors: list[OutputTokenProcessor]
    print_output_text: bool = False


class RequestProcessor:
    def __init__(self, config: RequestProcessorConfig, context: RequestProcessorContext):
        super().__init__()
        self.config = config
        self.context = context
        self.tokenizer = context.tokenizer
        self.processor = context.processor

        if config.multi_thread_request_process:
            self.lock = threading.Lock()
            self.executor = ThreadPoolExecutor(max_workers=32)
        self.batch_scheduler = context.batch_scheduler


    def process(self, request: Request, params: RequestProcessParameters):
        if self.config.multi_thread_request_process:
            self.executor.map(self._request_process_lock_wrapper, [(request, params)])
        else:
            self._request_process(request, params)

    def _request_process_lock_wrapper(self, request: Request, params: RequestProcessParameters):
        with self.lock:
            self._request_process(request, params)
    
    def _insert_image_tokens(self, token_ids: list[int], num_image_tokens):
        # replace each image_token_id with num_image_tokens image_token_id
        inserted_token_ids: list[int] = []
        for token_id in token_ids:
            if token_id == self.context.image_token_id:
                inserted_token_ids.extend([self.context.image_token_id] * (num_image_tokens - 1))
            inserted_token_ids.append(token_id)
        return inserted_token_ids 

    def _request_process(self, request: Request, params: RequestProcessParameters):
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
        token_ids = self._insert_image_tokens(token_ids, self.context.num_image_tokens)
        n_prompt_tokens = len(token_ids)
        n_image_tokens = n_pixel_values_images * self.context.num_image_tokens
        token_ids = token_ids + [-1] * (request.sampling_params.max_tokens - 1) # -1 will be set when executing
        # 3. image_overwrite_mask
        image_overwrite_mask = [token_id == self.context.image_token_id for token_id in token_ids]
        # 4. position_ids
        position_ids = list(range(len(token_ids)))
        # 5. cache_ids
        cache_ids = list(range(len(token_ids)))
        image_token_cache_ids = list(range(n_image_tokens))
        # 6. stage division
        builder = InstructionListBuilder()
        if images_tensor is not None:
            if self.config.disaggregate_embed_prefill:
                image_token_cache_ids = list(range(n_pixel_values_images * self.context.num_image_tokens))
                embed = ImageEmbed(
                    pixel_values = images_tensor,
                    cache_ids=image_token_cache_ids, 
                    token_pruning_params = None, 
                )
                prefill = ImageEmbedFill(
                    image_token_cache_ids=image_token_cache_ids, 
                    image_token_mask=image_overwrite_mask[:n_prompt_tokens], 
                    token_ids = token_ids[:n_prompt_tokens], 
                    position_ids = position_ids[:n_prompt_tokens], 
                    cache_ids = cache_ids[:n_prompt_tokens], 
                    sample = True, 
                    sample_dst = None, 
                )
                builder.append(embed)
                if self.context.ep_migrate:
                    builder.append(MigrateRequest())
                builder.append(prefill)
            else:
                prefill = ImageFill(
                    pixel_values = images_tensor, 
                    token_ids = token_ids[:n_prompt_tokens], 
                    position_ids = position_ids[:n_prompt_tokens], 
                    cache_ids = cache_ids[:n_prompt_tokens], 
                    sample = True, 
                    sample_dst = None, 
                )
                builder.append(prefill)
        else:
            prefill = TextFill(
                token_ids = token_ids[:n_prompt_tokens], 
                position_ids = position_ids[:n_prompt_tokens], 
                cache_ids = cache_ids[:n_prompt_tokens], 
                sample = True, 
                sample_dst = None, 
            )
            builder.append(prefill)
        if self.context.pd_migrate:
            builder.append(MigrateRequest())

        last_inst = prefill
        left = n_prompt_tokens     
        while left + 1 <= len(token_ids):
            right = left + 1
            decode = TextFill(
                token_ids = token_ids[left:right], 
                position_ids = position_ids[left:right], 
                cache_ids = cache_ids[left:right], 
                sample = True, 
                sample_dst = None, 
            )
            builder.append(decode)
            last_inst.sample_dst = decode
            last_inst = decode
            left = right

        instructions = builder.build_instruction_list()
        # 7. output tokenizer
        rcb = RequestControlBlock(
            instructions = instructions, 
            sampling_params = request.sampling_params, 
        )
        for output_token_processor in params.output_token_processors:
            rcb.register_output_token_processor(output_token_processor)

        if params.print_output_text:
            rcb.register_output_token_processor(PrintTextOutputTokenProcessor(self.tokenizer))

        if self.batch_scheduler:
            self.batch_scheduler.schedule_new(rcb)
