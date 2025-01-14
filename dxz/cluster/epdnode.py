import io
import time
import base64
import threading
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from dxz.request.request import Request
from dxz.request.rcb import RequestControlBlock, OutputTokenProcessor
from dxz.request.request_processor import RequestProcessor, RequestProcessorConfig, RequestProcessorContext, RequestProcessOutput
from dxz.engine.engine import EngineConfig, Engine

class EPDNode:
    def __init__(self, config: EngineConfig):
        self.engine = Engine(config)
        self.config = config
        self.tokenizer = self.engine.tokenizer

        self.request_processor_context = RequestProcessorContext(
            tokenizer = self.engine.tokenizer, 
            processor = self.engine.processor, 
            image_token_id = self.engine.vision_model_config.image_token_id, 
            num_image_tokens = self.engine.vision_model_config.num_image_tokens, 
            n_layers = self.engine.language_model_config.n_layers,
        )
        self.request_processor = RequestProcessor(
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
        if request.image is None and request.image_base64 is not None:
            request.image = Image.open(io.BytesIO(base64.b64decode(request.image_base64)))
        arrival_time = time.perf_counter()

        rcb = self.request_processor.process(request=request)

        rcb.metric.arrival_time = arrival_time
        rcb.output_token_processor = output_processor
        self.engine.schedule([rcb])

    def step(self):
        self.engine.step()