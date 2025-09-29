import torch
from typing import Union, Optional
from torch import Tensor
from hydrainfer.model import LanguageModelParameters, LanguageModelConfig, VisionModelConfig
from hydrainfer.memory import TokenCacheBlockManager, TokenCache, KVCache
from hydrainfer.layer.causal_attention import AttentionParameters, AttentionParametersBuilder
from hydrainfer.engine.isa import ImageEmbedFill, Fill
from hydrainfer.engine.rcb import RequestControlBlock, BatchRequest


class LanguageModelParametersBuilder:
    def __init__(
        self, 
        image_block_manager: TokenCacheBlockManager, 
        kv_cache_block_manager: TokenCacheBlockManager, 
        vision_model_config: VisionModelConfig, 
        language_model_config: LanguageModelConfig, 
        dtype: torch.dtype, 
        device: torch.device, 
        batch_prefill_with_paged_kvcache_wrapper: Optional["flashinfer.BatchPrefillWithPagedKVCacheWrapper"], 
        batch_decode_with_paged_kvcache_wrapper: Optional["flashinfer.BatchDecodeWithPagedKVCacheWrapper"], 
    ):
        self.image_block_manager = image_block_manager
        self.kv_cache_block_manager = kv_cache_block_manager
        self.vision_model_config = vision_model_config
        self.language_model_config = language_model_config
        self.dtype = dtype
        self.device = device
        self.batch_prefill_with_paged_kvcache_wrapper = batch_prefill_with_paged_kvcache_wrapper
        self.batch_decode_with_paged_kvcache_wrapper = batch_decode_with_paged_kvcache_wrapper

        self.requests_image_tokens: list[Tensor] = []
        self.token_ids: list[int] = []
        self.position_ids: list[int] = []
        self.selected_token_ids: list[int] = []
        self.attention_params_builder = AttentionParametersBuilder(
            num_qo_heads = self.language_model_config.n_qo_heads,
            num_kv_heads = self.language_model_config.n_kv_heads,
            head_dim = self.language_model_config.head_dim, 
            block_size = self.kv_cache_block_manager.config.block_size, 
            device = self.device, 
            flash_infer_batch_prefill_handler = self.batch_prefill_with_paged_kvcache_wrapper, 
            flash_infer_batch_decode_handler = self.batch_decode_with_paged_kvcache_wrapper, 
        )

    def add(self, rcb: RequestControlBlock, inst: Union[ImageEmbedFill, Fill]):
        assert isinstance(inst, ImageEmbedFill) or isinstance(inst, Fill)
        if isinstance(inst, ImageEmbedFill):
            token_cache = self.image_block_manager.get_layer_cache(layer_id=0)
            slot_ids = self.image_block_manager.v2p(rcb.virtual_image_cache, inst.image_token_cache_ids)
            image_token_cache = token_cache.get_caches()[0]
            image_token_cache = image_token_cache.view(-1, self.language_model_config.n_qo_heads * self.language_model_config.head_dim)
            slot_ids = torch.tensor(slot_ids, dtype=torch.int, device=self.device)
            request_image_tokens = image_token_cache[slot_ids, :]
            self.requests_image_tokens.append(request_image_tokens)

        self.token_ids += inst.token_ids
        self.position_ids += inst.position_ids
        if inst.sample:
            self.selected_token_ids.append(len(self.token_ids) - 1)

        virtual_kv_cache = rcb.virtual_kv_cache
        slot_ids = self.kv_cache_block_manager.set(virtual_kv_cache, inst.cache_ids)
        self.attention_params_builder.add_request(
            q_seq_len = len(inst.token_ids), 
            kv_seq_len = virtual_kv_cache.n_cache_tokens, 
            new_cache_slots = slot_ids, 
            block_table = virtual_kv_cache.block_table
        )

    def add_batch(self, batch: BatchRequest):
        for rcb, inst, in batch:
            self.add(rcb, inst)

    def build_language_model_parameters(self) -> LanguageModelParameters:
        image_over_write_mask = [token_id == self.vision_model_config.image_token_id for token_id in self.token_ids]
        if len(self.requests_image_tokens) == 0:
            image_features = None
            assert sum(image_over_write_mask) == 0
        else:
            image_features = torch.cat(self.requests_image_tokens, dim=0).to(dtype=self.dtype, device=self.device)
            assert sum(image_over_write_mask) == image_features.shape[0]
        
        for layer_id in range(self.language_model_config.n_layers):
            self.attention_params_builder.add_kv_cache(KVCache.from_token_cache(self.kv_cache_block_manager.get_layer_cache(layer_id)))

        layers_attention_params = self.attention_params_builder.build_attention_parameters()
        model_params = LanguageModelParameters(
            input_ids_or_input_embeds = torch.tensor(self.token_ids, dtype=torch.int, device=self.device), 
            position_ids = torch.tensor(self.position_ids, dtype=torch.int, device=self.device), 
            image_features = image_features, 
            image_overwrite_mask = torch.tensor(image_over_write_mask, dtype=torch.bool, device=self.device), 
            attention_params = layers_attention_params, 
            all_sequences_decode = layers_attention_params[0].all_sequences_decode, 
            selected_token_ids = self.selected_token_ids
        )
        return model_params
        