import os
import safetensors.torch
import torch
from torch import nn, Tensor
from transformers import LlavaConfig, AutoProcessor, AutoTokenizer, AutoConfig
from typing import Optional
from dxz.model.llama import LlamaForCausalLM
from dxz.model.clip import CLIPVisionModel
from dxz.model.parameters import VisionModelParameters, VisionModelOutput, LanguageModelParameters, LanguageModelOutput
from dxz.model.model_factory import VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.model.downloader import download_hf_model
from dxz.utils.torch_utils import str2device, str2dtype


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = nn.GELU()
    
    def forward(self, image_features: Tensor) -> Tensor:
        # image_features (n_images, n_tokens, vision_hidden_size)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaForConditionalGeneration(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = CLIPVisionModel(config.vision_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(config.text_config)
    
    @classmethod
    def from_safetensor(cls, model_weights_path: str, dtype: torch.dtype, device: torch.device):
        # 1. create model
        config = LlavaConfig.from_pretrained(model_weights_path)
        # creat model directly on the GPU with right module weight type to avoid slow data copy and cuda out of memory
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = cls(config)
        torch.set_default_dtype(torch.float)

        # 2. load weights
        state_dict = model.state_dict()
        loaded_set = set() # used to verify all weight are loaded
        for entry in os.scandir(model_weights_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
                print(f'load safetensor from {entry.path}')
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    state_dict[name].data.copy_(weight)
                    loaded_set.add(name)
        
        model.to(dtype) # to ensure that all tensor data type are correct such as non persistent rope inv freq, becuase it is created with float dtype specifically and will not be affected by set_default_dtype
        model.eval()

        # 3. verify
        assert len(loaded_set) == len(state_dict)

        return model


class LlavaVisionModel(VisionModel):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        # 1. config
        config = AutoConfig.from_pretrained(model_path)
        self.vision_feature_layer = config.vision_feature_layer
        # 2. create model
        torch.set_default_dtype(dtype)
        with torch.device(device):
            self.vision_tower = CLIPVisionModel(config.vision_config)
            self.multi_modal_projector = LlavaMultiModalProjector(config)
        torch.set_default_dtype(torch.float)
        # 3. load vision_tower state dict
        state_dict = self.vision_tower.state_dict()
        state_dict.update(self.multi_modal_projector.state_dict())
        loaded_set = set() # used to verify all weight are loaded
        for entry in os.scandir(model_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
                print(f'load safetensor from {entry.path}')
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    if name.startswith('vision_tower.'):
                        state_dict[name.removeprefix('vision_tower.')].copy_(weight)
                        loaded_set.add(name)
                    elif name.startswith('multi_modal_projector.'):
                        state_dict[name.removeprefix('multi_modal_projector.')].copy_(weight)
                        loaded_set.add(name)

        self.vision_tower.to(dtype)
        self.vision_tower.eval()
        self.multi_modal_projector.to(dtype)
        self.multi_modal_projector.eval()
        # 4. verify
        assert len(state_dict) == len(loaded_set), f'{len(state_dict)} {len(loaded_set)}'

    
    def forward(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        pixel_values = torch.cat(pixel_values, dim=0)
        assert pixel_values.dim() == 4, f'pixel value shape should be 4 dim but got {pixel_values.shape}'
        # pixel_values (n_images, n_channels, height, width)
        hidden_states, output = self.vision_tower(pixel_values, self.vision_feature_layer, model_params) # (n_images, num_tokens_per_images, hidden_size of vision model)
        selected_image_feature = hidden_states[:, 1:] # (n_images, num_tokens_per_images - 1, hidden_size of vision model) evict first class token of each image
        image_features = self.multi_modal_projector(selected_image_feature) # (n_images, num_tokens_per_images - 1, hidden_size of language model)
        output.image_features = image_features
        return output


class LlavaLanguageModel(LanguageModel):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        # 1. config
        config = AutoConfig.from_pretrained(model_path)
        self.image_token_id = config.image_token_index
        # 2. create model
        torch.set_default_dtype(dtype)
        with torch.device(device):
            self.language_model = LlamaForCausalLM(config.text_config)
        torch.set_default_dtype(torch.float)
        # 3. load vision_tower state dict
        state_dict = self.language_model.state_dict()
        loaded_set = set() # used to verify all weight are loaded
        for entry in os.scandir(model_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
                print(f'load safetensor from {entry.path}')
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    if name.startswith('language_model.'):
                        state_dict[name.removeprefix('language_model.')].copy_(weight)
                        loaded_set.add(name)

        # to ensure that all tensor data type are correct such as non persistent rope inv freq, becuase it is created with float dtype specifically and will not be affected by set_default_dtype
        self.language_model.to(dtype)
        self.language_model.eval()
        # 4. verify
        assert len(state_dict) == len(loaded_set), f'{len(state_dict)} {len(loaded_set)}'
    
    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        # input_ids      (n_text_tokens + n_image_tokens) n_text_tokens is number of text tokens, n_image_tokens is number of image tokens
        # image_features (n_text_tokens, hidden_size)
        # position_ids (n_text_tokens + n_image_tokens)
        input_embeds = self.language_model.model.embed_tokens(input_ids)
        if image_features is not None:
            image_overwrite_mask = input_ids == self.image_token_id
            input_embeds[image_overwrite_mask, :] = image_features.view(-1, input_embeds.shape[-1])
        sample_token_ids = self.language_model(input_embeds, position_ids, model_params) # (n_selected_tokens, )

        return LanguageModelOutput(
            sample_token_ids = sample_token_ids,
        )


class LlavaModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        self.name = config.name
        if config.path is None:
            self.path = download_hf_model(repo_id=config.name)
        else:
            self.path = config.path
        self.dtype = str2dtype(config.dtype)
        self.device = str2device(config.device)

    def getVisionModel(self) -> VisionModel:
        model = LlavaVisionModel(self.path, self.dtype, self.device)
        return model

    def getLanguageModel(self) -> LanguageModel:
        model = LlavaLanguageModel(self.path, self.dtype, self.device)
        return model

    def getVisionModelConfig(self) -> VisionModelConfig:
        config_ref = AutoConfig.from_pretrained(self.path)
        config = VisionModelConfig(
            image_token_id = config_ref.image_token_index, 
            num_image_tokens = config_ref.image_seq_length, 
        )
        return config

    def getLanguageModelConfig(self) -> LanguageModelConfig:
        config_ref = AutoConfig.from_pretrained(self.path)
        config = LanguageModelConfig(
            n_layers = config_ref.text_config.num_hidden_layers, 
            max_position_embeddings = config_ref.text_config.max_position_embeddings, 
            n_qo_heads = config_ref.text_config.num_attention_heads, 
            n_kv_heads = config_ref.text_config.num_key_value_heads, 
            head_dim = config_ref.text_config.head_dim, 
        )
        return config

    def getProcessor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.path)

    def getTokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.path)