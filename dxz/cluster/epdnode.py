import io
import time
import base64
import threading
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from dxz.engine.request import Request
from dxz.sequence.sequence import Sequence
from dxz.memory.compiler import CompilerConfig, Compiler, CompilerContext, CompileParameters
from dxz.engine.engine import EngineConfig, Engine

class EPDNode:
    def __init__(self, config: EngineConfig):
        self.engine = Engine(config)
        self.config = config
        self.tokenizer = self.engine.tokenizer

        self.sid_allocator = 0

        self.compiler_context = CompilerContext(
            tokenizer = self.engine.tokenizer, 
            processor = self.engine.processor, 
            image_token_id = self.engine.vision_model_config.image_token_id, 
            num_image_tokens = self.engine.vision_model_config.num_image_tokens, 
            n_layers = self.engine.language_model_config.n_layers,
        )
        self.compiler = Compiler(
            config = self.config.compiler_config, 
            context = self.compiler_context, 
        )

        self.add_request_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=32)

    def add_request(self, request: Request):
        if self.config.multi_thread_request_process:
            self.executor.map(self._add_request_async, [request])
        else:
            self._add_request(request)

    def _add_request_async(self, request: Request):
        with self.add_request_lock:
            self._add_request(request)
    
    def _add_request(self, request: Request):
        if request.image is None and request.image_base64 is not None:
            request.image = Image.open(io.BytesIO(base64.b64decode(request.image_base64)))
        arrival_time = time.perf_counter()

        static_info = self.compiler.compile(
            prompt = request.prompt, 
            images = request.image, 
            compile_params = CompileParameters(max_tokens = request.max_tokens)
            )
        sequence = Sequence(
            static_info=static_info, 
            sid = self.sid_allocator, 
            instructions = static_info.instructions, 
            virtual_kv_caches = self.engine.mmu.allocate_virtual_kv_caches(static_info.n_virtual_kv_caches), 
            max_tokens = request.max_tokens, 
            eos_token_id = None, 
            max_seq_len = self.engine.language_model_config.max_position_embeddings, 
            rid = request.request_id, 
        )
        sequence.metric.arrival_time = arrival_time
        self.sid_allocator += 1
        self.engine.scheduler.schedule_new([sequence])

    def step(self):
        self.engine.step()