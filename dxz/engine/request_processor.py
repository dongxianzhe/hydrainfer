import threading
import io
import base64
from torch import Tensor
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import Optional
from dxz.request.request import Request, RequestMetaData
from dxz.engine import Instruction, TextFill, ImageFill, EmptyInstruction, ImageEmbed, ImageEmbedFill, InstructionList, InstructionListBuilder, MigrateRequest, RequestControlBlock, OutputTokenProcessor, BatchScheduler, PrintTextOutputTokenProcessor, LogOutputTokenProcessor, OutputTokenParams, ScenarioClassifier, PullCache, EPMigrate, PDMigrate
from dxz.model.model_factory import ModelFactoryConfig, ModelFactoryContext, getModelFactory


@dataclass
class RequestProcessorConfig:
    multi_thread_request_process: bool = False
    num_request_process_workers: int = 32
    disaggregate_embed_prefill: bool = True
    ep_migrate: bool = False
    pd_migrate: bool = False
    model: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    ignore_eos: bool = False
    debug: bool = False


@dataclass
class RequestProcessParameters:
    output_token_processors: list[OutputTokenProcessor] = field(default_factory=list)
    outout_token_parmas: OutputTokenParams = field(default_factory=OutputTokenParams)


class RequestProcessorComponent:
    def process(self, request: Request, rcb: RequestControlBlock, params: RequestProcessParameters) -> RequestControlBlock:
        raise NotImplementedError


class RequestProcessorObserver:
    """
    RequestProcessorObserver update method should be thread-safe
    """
    def update(self, rcb: RequestControlBlock):
        raise NotImplementedError


class InstructionCreator(RequestProcessorComponent):
    def __init__(self, config: RequestProcessorConfig):
        super().__init__()
        self.config = config

        model_factory = getModelFactory(self.config.model, ModelFactoryContext())
        self.tokenizer = model_factory.getTokenizer()
        self.processor = model_factory.getProcessor()
        language_model_config = model_factory.getLanguageModelConfig()
        vision_model_config = model_factory.getVisionModelConfig()
        self.image_token_id = vision_model_config.image_token_id
        self.num_image_tokens = vision_model_config.num_image_tokens
        self.n_layers = language_model_config.n_layers

    def _insert_image_tokens(self, token_ids: list[int], num_image_tokens):
        # replace each image_token_id with num_image_tokens image_token_id
        inserted_token_ids: list[int] = []
        for token_id in token_ids:
            if token_id == self.image_token_id:
                inserted_token_ids.extend([self.image_token_id] * (num_image_tokens - 1))
            inserted_token_ids.append(token_id)
        return inserted_token_ids 

    def process(self, request: Request, rcb: RequestControlBlock, params: RequestProcessParameters) -> RequestControlBlock:
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
        n_token_ids_images = token_ids.count(self.image_token_id)
        assert n_token_ids_images == n_pixel_values_images, f"image number is not equal between text and image list {n_token_ids_images} {n_pixel_values_images}"
        token_ids = self._insert_image_tokens(token_ids, self.num_image_tokens)
        n_images = n_token_ids_images
        n_prompt_tokens = len(token_ids)
        n_image_tokens = n_pixel_values_images * self.num_image_tokens
        n_text_tokens = n_prompt_tokens - n_image_tokens
        token_ids = token_ids + [-1] * (request.sampling_params.max_tokens - 1) # -1 will be set when executing
        # 3. image_overwrite_mask
        image_overwrite_mask = [token_id == self.image_token_id for token_id in token_ids]
        # 4. position_ids
        position_ids = list(range(len(token_ids)))
        # 5. cache_ids
        cache_ids = list(range(len(token_ids)))
        image_token_cache_ids = list(range(n_image_tokens))
        # 6. stage division
        builder = InstructionListBuilder()
        if images_tensor is not None:
            if self.config.disaggregate_embed_prefill:
                image_token_cache_ids = list(range(n_pixel_values_images * self.num_image_tokens))
                embed = ImageEmbed(
                    pixel_values = images_tensor,
                    cache_ids=image_token_cache_ids, 
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
                if self.config.ep_migrate:
                    builder.append(EPMigrate())
                    builder.append(PullCache())
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
        if self.config.pd_migrate:
            builder.append(PDMigrate())
            builder.append(PullCache())

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
        if self.config.debug:
            print(f'{request.prompt[:10]} {instructions}')

        rcb.instructions = instructions
        rcb.request_metadata = RequestMetaData(
            n_images = n_images, 
            n_prompt_tokens = n_prompt_tokens, 
            n_image_tokens = n_image_tokens, 
            n_text_tokens = n_text_tokens, 
        )
        return rcb


class ScenarioPredictor(RequestProcessorComponent):
    def __init__(self, config: RequestProcessorConfig):
        self.config = config
        self.scenario_classifier = ScenarioClassifier()

    def process(self, request: Request, rcb: RequestControlBlock, params: RequestProcessParameters) -> RequestControlBlock:
        if rcb.request_metadata is None:
            return rcb
        if rcb.sampling_params is None:
            return rcb

        scenario_type = self.scenario_classifier.classify(
            n_text_tokens = rcb.request_metadata.n_text_tokens, 
            n_output_tokens = request.sampling_params.max_tokens, 
        )
        rcb.scenario_type = scenario_type
        return rcb

class OutputTokenProcessorComponent(RequestProcessorComponent):
    def __init__(self, config: RequestProcessorConfig):
        self.config = config

    def process(self, request: Request, rcb: RequestControlBlock, params: RequestProcessParameters) -> RequestControlBlock:
        rcb.request_id = request.request_id
        rcb.output_token_params = params.outout_token_parmas
        for output_token_processor in params.output_token_processors:
            rcb.register_output_token_processor(output_token_processor)
        return rcb

class SamplingParamsProcess(RequestProcessorComponent):
    def __init__(self, config: RequestProcessorConfig):
        self.config = config
        model_factory = getModelFactory(self.config.model, ModelFactoryContext())
        tokenizer = model_factory.getTokenizer()
        self.eos_token_id = tokenizer.eos_token_id

    def process(self, request: Request, rcb: RequestControlBlock, params: RequestProcessParameters) -> RequestControlBlock:
        rcb.sampling_params = request.sampling_params
        if not self.config.ignore_eos:
            rcb.sampling_params.eos_token_ids.append(self.eos_token_id)
        return rcb

class RequestProcessor:
    def __init__(self, config: RequestProcessorConfig):
        super().__init__()
        self.config = config

        if config.multi_thread_request_process:
            self.lock = threading.Lock()
            self.executor = ThreadPoolExecutor(max_workers=config.num_request_process_workers)

        self.request_process_components: list[RequestProcessorComponent] = [
            SamplingParamsProcess(config), 
            InstructionCreator(config), 
            ScenarioPredictor(config), 
            OutputTokenProcessorComponent(config), 
        ]
        self.request_process_observer: list[RequestProcessorObserver] = []

    def register_output_observer(self, observer: RequestProcessorObserver):
        self.request_process_observer.append(observer)

    def process(self, request: Request, params: RequestProcessParameters) -> None:
        if self.config.multi_thread_request_process:
            self.executor.submit(self._request_process_lock_wrapper, request, params)
        else:
            self._request_process(request, params)

    def _request_process_lock_wrapper(self, request: Request, params: RequestProcessParameters):
        try:
            with self.lock:
                self._request_process(request, params)
        except Exception as e:
            print(e)
    
    def _request_process(self, request: Request, params: RequestProcessParameters):
        rcb = RequestControlBlock()
        for component in self.request_process_components:
            rcb = component.process(request, rcb, params)
        for observer in self.request_process_observer:
            observer.update(rcb)