import io
import time
import base64
import asyncio
import threading
from dataclasses import dataclass, field, fields
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from dxz.request.request import Request
from dxz.request.rcb import RequestControlBlock, OutputTokenProcessor
from dxz.request.request_processor import RequestProcessor, RequestProcessorConfig, RequestProcessorContext, LanguageRequestProcessor, VisionRequestProcessor
from dxz.engine.engine import EngineConfig, Engine
from dxz.utils.zmq_utils import init_zmq_send


class ZeroMqStreamOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, zmq_send):
        self.zmq_send = zmq_send

    def append_token_id(self, token_id: int, is_last_token: bool=False):
        self.zmq_send.send_pyobj(token_id)


class ZeroMqNonStreamOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, zmq_send):
        self.token_ids: list[int] = []
        self.zmq_send = zmq_send

    def append_token_id(self, token_id: int, is_last_token: bool=False):
        self.token_ids.append(token_id)
        if is_last_token:
            self.zmq_send.send_pyobj(self.token_ids)


@dataclass
class RayEPDNodeConfig:
    request_processor_config: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    engine_config: EngineConfig = field(default_factory=EngineConfig)


@dataclass
class RayEPDNodeContext:
    zmq_url: str
    

class RayEPDNode:
    def __init__(self, config: RayEPDNodeConfig, context: RayEPDNodeContext):
        self.config = config
        self.context = context
        self.zmq_send = init_zmq_send(zmq_url=context.zmq_url)
        self.engine = Engine(config.engine_config)
        self.tokenizer = self.engine.tokenizer

        self.request_processor_context = RequestProcessorContext(
            tokenizer = self.engine.tokenizer, 
            processor = self.engine.processor, 
            image_token_id = self.engine.vision_model_config.image_token_id, 
            num_image_tokens = self.engine.vision_model_config.num_image_tokens, 
            n_layers = self.engine.language_model_config.n_layers,
        )
        self.language_request_processor = LanguageRequestProcessor(
            config = self.config.request_processor_config, 
            context = self.request_processor_context, 
        )
        self.vision_request_processor = VisionRequestProcessor(
            self.config.request_processor_config, 
            self.request_processor_context, 
        )

    def add_request(self, request: Request):
        output_processor = ZeroMqNonStreamOutputTokenProcessor(zmq_send = self.zmq_send)
        if request.image is None and request.image_base64 is not None:
            request.image = Image.open(io.BytesIO(base64.b64decode(request.image_base64)))
        arrival_time = time.perf_counter()

        if request.image is None and request.image_base64 is None:
            rcb = self.language_request_processor.process(request=request)
        else:
            rcb = self.vision_request_processor.process(request=request)

        rcb.metric.arrival_time = arrival_time
        rcb.output_token_processor = output_processor
        self.engine.schedule([rcb])

    async def step_loop(self):
        while True:
            self.engine.step()
            await asyncio.sleep(0)