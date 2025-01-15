import os
import safetensors.torch
import torch
from torch import nn, Tensor
from transformers import LlavaConfig, AutoProcessor, AutoTokenizer
from typing import Optional
from dxz.model.llama import LlamaForCausalLM
from dxz.model.clip import CLIPVisionModel
from dxz.model.parameters import VisionModelParameters, VisionModelOutput, LanguageModelParameters, LanguageModelOutput
from dxz.model.model_factory import VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactory
from dxz.model.downloader import download_hf_model

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
        model_ref = LlavaForConditionalGeneration.from_safetensor(model_path, dtype, device)
        config_ref = model_ref.config
        self.vision_feature_layer = config_ref.vision_feature_layer

        self.vision_tower = model_ref.vision_tower
        self.multi_modal_projector = model_ref.multi_modal_projector
        self.config = VisionModelConfig(
            image_token_id = config_ref.image_token_index, 
            num_image_tokens = config_ref.image_seq_length, 
        )
    
    def forward(self, pixel_values: Tensor, model_params: VisionModelParameters) -> VisionModelOutput:
        # pixel_values (n_images, n_channels, height, width)
        hidden_states, output = self.vision_tower(pixel_values, self.vision_feature_layer, model_params)
        selected_image_feature = hidden_states[:, 1:]
        image_features = self.multi_modal_projector(selected_image_feature)
        output.image_features = image_features
        return output

class LlavaLanguageModel(LanguageModel):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        model_ref = LlavaForConditionalGeneration.from_safetensor(model_path, dtype, device)
        config_ref = model_ref.config
        self.image_token_id = config_ref.image_token_index

        self.language_model = model_ref.language_model
        self.config = LanguageModelConfig(
            n_layers = config_ref.text_config.num_hidden_layers, 
            max_position_embeddings = config_ref.text_config.max_position_embeddings, 
            n_qo_heads = config_ref.text_config.num_attention_heads, 
            n_kv_heads = config_ref.text_config.num_key_value_heads, 
            head_dim = config_ref.text_config.head_dim, 
        )
    
    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        # input_ids      (n_tokens + m_tokens) n_tokens is text tokens, m_tokens is n_image * 576, n_image is number of image
        # image_features (m_tokens, hidden_size)
        # position_ids (n_tokens + m_tokens)
        input_embeds = self.language_model.model.embed_tokens(input_ids)
        if image_features is not None:
            image_overwrite_mask = input_ids == self.image_token_id
            input_embeds[image_overwrite_mask, :] = image_features.view(-1, input_embeds.shape[-1])
        sample_token_ids = self.language_model(input_embeds, position_ids, model_params)

        return LanguageModelOutput(
            sample_token_ids = sample_token_ids,
        )

class LlavaModelFactory(ModelFactory):
    def __init__(self, model_name: str, model_path: Optional[str] = None, dtype: torch.dtype=torch.half, device: torch.device=torch.device('cuda:0')):
        self.model_name = model_name
        if model_path is None:
            self.model_path = download_hf_model(repo_id=model_name)
        else:
            self.model_path = model_path
        self.dtype = dtype
        self.device = device

    def getVisionModel(self) -> tuple[VisionModel, VisionModelConfig]:
        model = LlavaVisionModel(self.model_path, self.dtype, self.device)
        return model, model.config

    def getLanguageModel(self) -> tuple[LanguageModel, LanguageModelConfig]:
        model = LlavaLanguageModel(self.model_path, self.dtype, self.device)
        return model, model.config

    def getProcessor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.model_path)

    def getTokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_path)