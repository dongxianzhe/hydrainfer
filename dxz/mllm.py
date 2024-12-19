import threading
import torch
from PIL import Image
from torch import Tensor
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoProcessor
from dxz.model.downloader import download_hf_model
from dxz.engine.fast_engine import RequestOutput, SamplingParams, MemoryConfig, MemoryManagementPolicy, FastEngine, EngineConfig, BatchPriority


class MLLM:
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        num_blocks: int = 34000, 
        priority: str = 'prefill', 
        max_running_sequences: int = 20, 
        max_batch_fill_tokens: int = 1024, 
        max_batch_embed_images: int= 3, 
        batch_embed_fill: bool = True,
        debug_mode:bool = False,
        disaggregate_embed_prefill:bool = True, 
        batch_image_embed:bool = True, 
        ):
        model_path = download_hf_model(repo_id=model_name)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.lock = threading.Lock()

        config = EngineConfig()
        config.model_path = model_path
        config.memory_config.num_blocks = num_blocks
        config.scheduler_config.batch_embed_fill = batch_embed_fill
        config.scheduler_config.max_running_sequences = max_running_sequences
        config.scheduler_config.max_batch_fill_tokens = max_batch_fill_tokens
        config.scheduler_config.max_batch_embed_images = max_batch_embed_images
        config.scheduler_config.debug_mode = debug_mode
        config.scheduler_config.priority = BatchPriority.PREFILL if priority == "prefill" else BatchPriority.DECODE
        config.stage_config.disaggregate_embed_prefill = disaggregate_embed_prefill
        config.batch_image_embed_forward = batch_image_embed


        self.engine = FastEngine(config)
    
    def process_image(self, input):
        with self.lock:
            images = self.processor(
                text="", 
                images=input['multi_modal_data']['image'], 
                return_tensors="pt"
            )['pixel_values']
        return images

    def generate(self, inputs):
        """ 
        inputs example
        [{
            "prompt" : prompt, 
            "multi_modal_data":{
                "image": image
            },
            "max_tokens": 50,
        }, ...]
        """
        prompts: list[str] = []
        images: list[Tensor] = []
        samplings_params: list[SamplingParams] = []

        inputs_images = self.executor.map(self.process_image, inputs)
        for input_images in inputs_images:
            images.append(input_images)

        for input in inputs:
            prompts.append(input['prompt'])
            # images.append(self.process_image(input))
            sp = SamplingParams()
            sp.max_tokens = input['max_tokens']
            samplings_params.append(sp)

        outputs: list[RequestOutput] = [None] * len(prompts)
        
        # def output_callback(output: RequestOutput) -> bool:
        #     return True

        def batch_output_callback(index: int, output: RequestOutput) -> bool:
            outputs[index] = output
            return True

        # future = self.engine.add_request_async(prompts[0], images[0], samplings_params[0], False, output_callback)
        # future.wait()
        
        futures = self.engine.add_requests_async(prompts, images,  samplings_params, False, batch_output_callback)
        futures.wait()

        self.engine.run_until_complete()

        return outputs

if __name__ == '__main__':
    image_path = f'/home/xzd/projects/dxz/benchmark/dataset/cherry_blossom.jpg'
    image = Image.open(image_path)
    question = "What is the content of this image?"
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    batch_size = 10
    inputs = [{
        "prompt" : prompt, 
        "multi_modal_data":{
            "image": image
        },
        # "max_tokens":0, 
        # "max_tokens":random.randint(30, 70), 
        # "max_tokens": 10, 
        "max_tokens": i * 10,
    } for i in range(batch_size)]
    mllm = MLLM()

    import time
    start = time.perf_counter()
    outputs = mllm.generate(inputs)
    end = time.perf_counter()
    print(f'duration {end - start}')
    for output in outputs:
        print(output)
    mllm.engine.stop()
