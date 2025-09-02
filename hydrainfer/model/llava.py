import os
import safetensors.torch
import torch
from jinja2 import Template
from torch import nn, Tensor
from transformers import LlavaConfig, AutoProcessor, AutoTokenizer, AutoConfig
from typing import Optional
from hydrainfer.model.llama import LlamaForCausalLM
from hydrainfer.model.clip import CLIPVisionModel
from hydrainfer.model.parameters import VisionModelParameters, VisionModelOutput, LanguageModelParameters, LanguageModelOutput
from hydrainfer.model.model_factory import VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactory, ModelFactoryConfig, ModelFactoryContext, ImageTokenCaculator, Tokenizer, ModelProfiler, ModelParamsConfig
from hydrainfer.model.model_profiler import VisionLanguageModelProfiler
from hydrainfer.model.downloader import download_hf_model
from hydrainfer.utils.torch_utils import str2device, str2dtype
from hydrainfer.utils.logger import getLogger
from hydrainfer.model.model_loader import load_safetensor
logger = getLogger(__name__)

class LlavaTokenCaculator(ImageTokenCaculator):
    def __init__(self, path: str):
        config = LlavaConfig.from_pretrained(path)
        self.num_image_tokens = config.image_seq_length 

    def get_num_image_tokens(self, image_size: tuple[int, int]) -> int:
        return self.num_image_tokens


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
        load_safetensor(model_with_prefix_list=[(model, '')], param_with_name_list=[], model_weights_path=model_weights_path),

        model.to(dtype) # to ensure that all tensor data type are correct such as non persistent rope inv freq, becuase it is created with float dtype specifically and will not be affected by set_default_dtype
        model.eval()

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
        load_safetensor(
            model_with_prefix_list=[
                (self.vision_tower, 'vision_tower.'), 
                (self.multi_modal_projector, 'multi_modal_projector.'), 
            ], 
            param_with_name_list=[], 
            model_weights_path=model_path, 
        )

        self.vision_tower.to(dtype)
        self.vision_tower.eval()
        self.multi_modal_projector.to(dtype)
        self.multi_modal_projector.eval()

    
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
        load_safetensor(model_with_prefix_list=[(self.language_model, 'language_model.')], param_with_name_list=[], model_weights_path=model_path)

        # to ensure that all tensor data type are correct such as non persistent rope inv freq, becuase it is created with float dtype specifically and will not be affected by set_default_dtype
        self.language_model.to(dtype)
        self.language_model.eval()
    
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


class LlavaTokenizer(Tokenizer):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        current_script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(current_script_path)
        template_path = os.path.join(script_dir, "chat_template", "template_llava.jinja")
        with open(template_path, 'r', encoding='utf-8') as file:
            self.chat_template = Template(file.read())

    def encode(self, prompt: str) -> list[int]:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return token_ids

    def decode(self, token_id: int) -> str:
        # note that this is not an ASCII U+005F _, it is U+2581
        if self.tokenizer.convert_ids_to_tokens([token_id])[0].startswith('▁'):
            text = " " + self.tokenizer.decode([token_id])
        else:
            text = self.tokenizer.decode([token_id])
        return text

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        prompt = self.chat_template.render(
            messages = messages, 
            bos_token = self.bos_token, 
            eos_token = self.eos_token,
            add_generation_prompt = True, 
        )
        return prompt


class LlavaModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
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
            image_token = "<image>", 
            image_token_id = config_ref.image_token_index, 
            image_token_caculator = LlavaTokenCaculator(self.path), 
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

    def getModelProfiler(self) -> ModelProfiler:
        return VisionLanguageModelProfiler(self.path, vision_model_prefixes=['vision_tower.', 'multi_modal_projector.'], language_model_prefixes=['language_model.'])
