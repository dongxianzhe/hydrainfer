from PIL import Image
import base64
import asyncio
from dataclasses import dataclass
from dxz.model.model_factory import getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.engine.scheduler import BatchRequest, BatchScheduler, SchedulerConfig
from dxz.engine.executor import InstructionExecutor, ExecutorConfig, ExecutorContext
from dxz.cluster.raynode import RayNode
from dxz.request.request import Request
from dxz.request.rcb import RequestControlBlock
from dxz.engine.isa import *


@dataclass
class ENodeConfig:
    pass


@dataclass
class ENodeContext:
    executor_config = ExecutorConfig
    scheduler_config: SchedulerConfig


class ENode(RayNode):
    def __init__(self, config: ENodeConfig, context: ENodeContext):
        model_factory = getModelFactory(ModelFactoryConfig, ModelFactoryContext)
        self.scheduler = BatchScheduler(context.scheduler_config)
        self.vision_model_config = model_factory.getVisionModelConfig()
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.tokenizer = model_factory.getTokenizer()
        self.processor = model_factory.getProcessor()
        self.executor = InstructionExecutor(context.executor_config, ExecutorContext(
            model_factory_config = context.model_factory_config, 
            block_size = context.memory_config.block_size, 
            mmu = self.mmu, 
            worker=self.worker, 
        ))
        self.nodes = []

    def insert_image_tokens(self, token_ids: list[int], num_image_tokens):
        # replace each image_token_id with num_image_tokens image_token_id
        inserted_token_ids: list[int] = []
        for token_id in token_ids:
            if token_id == self.vision_model_config.image_token_id:
                inserted_token_ids.extend([self.vision_model_config.image_token_id] * (num_image_tokens - 1))
            inserted_token_ids.append(token_id)
        return inserted_token_ids 

    async def process_request(self, request: Request) -> RequestControlBlock:
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
        n_token_ids_images = token_ids.count(self.vision_model_config.image_token_id)
        assert n_token_ids_images == n_pixel_values_images, f"image number is not equal between text and image list {n_token_ids_images} {n_pixel_values_images}"
        token_ids = self.insert_image_tokens(token_ids, self.vision_model_config.num_image_tokens)
        n_prompt_tokens = len(token_ids)
        token_ids = token_ids + [-1] * (request.sampling_params.max_tokens - 1) # -1 will be set when executing
        # 3. image_overwrite_mask
        image_overwrite_mask = [token_id == self.vision_model_config.image_token_id for token_id in token_ids]
        # 4. position_ids
        position_ids = list(range(len(token_ids)))
        # 5. cache_ids
        n_virtual_kv_caches: int = self.language_model_config.n_layers
        layer_virtual_kv_cache_ids = list(range(self.language_model_config.n_layers))
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
                    cache_ids = [cache_ids[:n_prompt_tokens] for _ in range(self.language_model_config.n_layers)], 
                    kv_cache_ids = layer_virtual_kv_cache_ids, 
                    sample = True, 
                    sample_dst = None, 
                )
                embed.image_features_dst = prefill
                builder.append(embed)
                builder.append(MigrateRequest())
                builder.append(prefill)
                builder.append(MigrateRequest())
            else:
                prefill = ImageFill(
                    pixel_values = images_tensor, 
                    token_ids = token_ids[:n_prompt_tokens], 
                    position_ids = position_ids[:n_prompt_tokens], 
                    cache_ids = [cache_ids[:n_prompt_tokens] for _ in range(self.language_model_config.n_layers)], 
                    kv_cache_ids = layer_virtual_kv_cache_ids, 
                    sample = True, 
                    sample_dst = None, 
                )
                builder.append(prefill)
                builder.append(MigrateRequest())
        else:
            prefill = TextFill(
                token_ids = token_ids[:n_prompt_tokens], 
                position_ids = position_ids[:n_prompt_tokens], 
                cache_ids = [cache_ids[:n_prompt_tokens] for _ in range(self.language_model_config.n_layers)], 
                kv_cache_ids = layer_virtual_kv_cache_ids, 
                sample = True, 
                sample_dst = None, 
            )
            builder.append(MigrateRequest())
            builder.append(prefill)
            builder.append(MigrateRequest())

        last_inst = prefill
        left = n_prompt_tokens     
        while left + 1 <= len(token_ids):
            right = left + 1
            decode = TextFill(
                token_ids = token_ids[left:right], 
                position_ids = position_ids[left:right], 
                cache_ids = [cache_ids[left:right] for _ in range(self.language_model_config.n_layers)], 
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

    async def add_request(self, request: Request):
        rcb = await self.process_request(request)
        curr = rcb.instructions.head
        while curr:
            print(curr)
            curr = curr.next
        self.scheduler.schedule_new([rcb])
        

    async def step(self):
        batch: BatchRequest = self.scheduler.step()

        batch_image_embed = BatchRequest()
        batch_migrate = BatchRequest()
        for rcb, inst in batch:
            if isinstance(inst, MigrateRequest):
                batch_migrate.append(rcb)
            elif isinstance(inst, ImageEmbed):
                batch_image_embed.append(rcb)
            else:
                raise Exception(f'{inst} is not supported in enode')
        self.executor.execute_image_embed()
        await self.execute_batch_migrate(batch_migrate)
        for rcb, inst in batch:
            if rcb.is_finished():
                pass
            else:
                self.scheduler.schedule_running(rcb)
        raise NotImplementedError

    async def step_loop(self):
        while True:
            await self.step()
            await asyncio.sleep(0.001)

    async def register_node(self, node: "RayNode"): 
        self.nodes.append(node)

    async def execute_batch_migrate(self, contexts: BatchRequest):
        if len(contexts) == 0:
            return
        node = self.nodes[0]
        for rcb, _ in contexts:
            rcb.step()
            node.migrate.remote(rcb)