import os
import torch
import safetensors
from PIL import Image
from typing import Optional
from torch import nn, Tensor
from transformers import LlavaConfig, LlavaNextConfig, AutoProcessor, AutoConfig, AutoTokenizer
from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image
from dxz.model.downloader import download_hf_model
from dxz.model import ModelFactory, ModelFactoryConfig, ModelFactoryContext, VisionModelConfig, LanguageModelConfig, LanguageModel, VisionModel, ImageTokenCaculator, Tokenizer
from dxz.model.parameters import LanguageModelParameters, LanguageModelOutput, VisionModelParameters, VisionModelOutput
from dxz.utils.torch_utils import str2device, str2dtype
from dxz.model.llava import LlavaVisionModel, LlavaLanguageModel, LlavaTokenizer


class LlavaNextImageTokenCaculator(ImageTokenCaculator):
    def __init__(self, path: str):
        self.config = LlavaNextConfig.from_pretrained(path)

    def get_unpad_image_size(self, current_size: tuple[int, int], original_size: tuple[int, int]) -> tuple[int, int]:
        original_height, original_width = original_size
        current_height, current_width = current_size
        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (current_height - new_height) // 2
            current_size = (current_height - padding * 2, current_width)
        else:
            scale_factor = current_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (current_width - new_width) // 2
            current_size = (current_height, current_width - padding * 2)

        return current_size

    def get_num_image_tokens(self, image_size: tuple[int, int]) -> int:
        # image_size (height, width)
        num_patch_height, num_patch_width = get_anyres_image_grid_shape(image_size, self.config.image_grid_pinpoints, self.config.vision_config.image_size)
        n_height_width_tokens_per_patch = self.config.vision_config.image_size // self.config.vision_config.patch_size
        current_size = (num_patch_height * n_height_width_tokens_per_patch, num_patch_width * n_height_width_tokens_per_patch)
        current_size = self.get_unpad_image_size(current_size, image_size)

        n_base_patch_tokens = self.config.image_seq_length
        n_patches_tokens = current_size[0] * (current_size[1] + 1)
        return n_base_patch_tokens + n_patches_tokens


class LlavaNextLanguageModel(LanguageModel):
    def __init__(self, path: str, dtype: torch.dtype, device: torch.device):
        self.model = LlavaLanguageModel(path, dtype, device)

    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        return self.model.forward(input_ids, image_features, position_ids, model_params)


class LlavaNextVisionModel(VisionModel):
    def __init__(self, path: str, dtype: torch.dtype, device: torch.device):
        self.config = LlavaNextConfig.from_pretrained(path)
        self.vision_model = LlavaVisionModel(path, dtype, device)
        self.image_newline = nn.Parameter(torch.empty(self.config.text_config.hidden_size, dtype=dtype, device=device), requires_grad=False)

        loaded_set = set()
        for entry in os.scandir(path):
            if entry.is_file and os.path.splitext(entry.name)[1] == '.safetensors':
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    if name == 'image_newline':
                        self.image_newline.copy_(weight)
                        loaded_set.add(name)
        assert len(loaded_set) == 1

    def forward(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        n_patches_list: list[int] = []
        pixel_values_list: list[Tensor] = []
        for pixel_values_per_request in pixel_values:
            # pixel_values_per_request (1, n_patches, 3, 336, 336)
            assert pixel_values_per_request.shape[0] == 1, f'only support one image per request, {pixel_values_per_request.shape}'
            n_images, n_patches, n_channels, height, width = pixel_values_per_request.shape
            pixel_values_per_request = pixel_values_per_request.view(-1, n_channels, height, width)
            pixel_values_list.append(pixel_values_per_request)
            n_patches_list.append(n_images * n_patches)

        output = self.vision_model.forward(pixel_values_list, model_params)
        image_features = output.image_features # (total_patches, 576, hidden_size)

        unpad_image_features_list: list[Tensor] = []
        image_features_id: int = 0
        for i, n_patches in enumerate(n_patches_list):
            patch_embeddings = image_features[image_features_id : image_features_id + n_patches] # patch_embeddings (n_patches, n_tokens_per_patch, hidden_size)
            unpad_image_features = self._merge_image_patch_embeddings(origin_image_size=model_params.original_image_sizes[i], patch_embeddings=patch_embeddings) # (n_this_image_tokens, hidden_size)
            unpad_image_features_list.append(unpad_image_features)
            image_features_id += n_patches

        image_features = torch.cat(unpad_image_features_list, dim=0)

        return VisionModelOutput(
            image_features = image_features, 
            scores = output.scores
        )

    def _merge_image_patch_embeddings(self, origin_image_size: tuple[int, int], patch_embeddings: torch.Tensor) -> torch.Tensor:
        # image_size (height, width)
        # patch_embeddings (n_patches, n_tokens_per_patch, hidden_size)
        n_height_tokens_per_patch = n_width_tokens_per_patch = self.config.vision_config.image_size // self.config.vision_config.patch_size
        base_patch_embeds = patch_embeddings[0] # (n_tokens_per_patch=576, hidden_size)
        other_patch_embeds = patch_embeddings[1:] # (n_other_patches, n_tokens_per_patch=576, hidden_size)
        assert n_height_tokens_per_patch * n_width_tokens_per_patch == base_patch_embeds.shape[0], f"The number of patches is not consistent with the image size. {n_height_tokens_per_patch} {n_width_tokens_per_patch} {base_patch_embeds.shape[0]}"

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(origin_image_size, self.config.image_grid_pinpoints, self.config.vision_config.image_size)
        num_patches = num_patch_height * num_patch_width
        # Image patches might be padded for batch processing
        other_patch_embeds = other_patch_embeds[:num_patches].view(num_patch_height, num_patch_width, n_height_tokens_per_patch, n_width_tokens_per_patch, -1) # (num_patch_height, num_patch_width, n_height_tokens_per_patch, n_width_tokens_per_patch, hidden_size)
        other_patch_embeds = other_patch_embeds.permute(4, 0, 2, 1, 3).contiguous().flatten(1, 2).flatten(2, 3)
        # (hidden_size, num_patch_height * n_height_tokens_per_patch, num_patch_width * n_width_tokens_per_patch)
        other_patch_embeds = unpad_image(other_patch_embeds, origin_image_size)
        # (hidden_size, n_tokens_height, n_tokens_width) n_tokens_height / n_tokens_width = height / width
        other_patch_embeds = torch.cat((
            other_patch_embeds, # (hidden_size, n_tokens_height, n_tokens_width)
            self.image_newline[:, None, None].expand(*other_patch_embeds.shape[:-1], 1).to(other_patch_embeds.device), # (hidden_size, ) -> (hidden_size, 1, 1) -> (hidden_size, n_tokens_height, 1)
        ), dim=-1)
        # (hidden_size, n_tokens_height, n_tokens_width + 1)
        other_patch_embeds = other_patch_embeds.flatten(1, 2).transpose(0, 1) # (n_tokens_height * (n_tokens_width + 1), hidden_size)
        merged_patch_embeddings = torch.cat((base_patch_embeds, other_patch_embeds), dim=0)

        return merged_patch_embeddings


class LlavaNextModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        self.name = config.name
        if config.path is None:
            self.path = download_hf_model(repo_id=config.name)
        else:
            self.path = config.path
        self.dtype = str2dtype(config.dtype)
        self.device = str2device(config.device)

    def getVisionModel(self) -> VisionModel:
        return LlavaNextVisionModel(self.path, self.dtype, self.device)

    def getLanguageModel(self) -> LanguageModel:
        return LlavaNextLanguageModel(self.path, self.dtype, self.device)

    def getVisionModelConfig(self) -> VisionModelConfig:
        config_ref = AutoConfig.from_pretrained(self.path)
        config = VisionModelConfig(
            image_token = "<image>", 
            image_token_id = config_ref.image_token_index, 
            image_token_caculator = LlavaNextImageTokenCaculator(self.path), 
        )
        return config

    def getLanguageModelConfig(self) -> LanguageModelConfig:
        config_ref = AutoConfig.from_pretrained(self.path)
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        config = LanguageModelConfig(
            n_layers = config_ref.text_config.num_hidden_layers, 
            max_position_embeddings = config_ref.text_config.max_position_embeddings, 
            n_qo_heads = config_ref.text_config.num_attention_heads, 
            n_kv_heads = config_ref.text_config.num_key_value_heads, 
            head_dim = config_ref.text_config.head_dim, 
            eos_token_id = tokenizer.eos_token_id, 
        )
        return config

    def getProcessor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.path)

    def getTokenizer(self) -> Tokenizer:
        return LlavaTokenizer(self.path)
