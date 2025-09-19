import os
import math
import safetensors.torch
from jinja2 import Template
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from einops import rearrange, repeat
from torch import nn, Tensor
from typing import Optional
from hydrainfer.model.downloader import download_hf_model
from hydrainfer.model.parameters import LanguageModelParameters, LanguageModelOutput, VisionModelParameters, VisionModelOutput
from hydrainfer.model.model_factory import VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactory, ModelFactoryConfig, ModelFactoryContext, ImageTokenCaculator, Tokenizer, ImageProcessor
from hydrainfer.model.deepseek_v3 import DeepseekForCausalLM
from hydrainfer.utils.torch_utils import str2dtype, str2device
from hydrainfer.transformers_utils.deepseek_vl2_config import MlpProjectorConfig, DeepseekVLV2Config
from hydrainfer.transformers_utils.deepseek_vl2_processor import DeepseekVLV2Processor
from transformers import AutoProcessor, AutoTokenizer
from hydrainfer.model import ModelProfiler
from hydrainfer.model.model_profiler import VisionLanguageModelProfiler
from hydrainfer.utils.logger import getLogger
from hydrainfer.model.model_loader import load_safetensor
logger = getLogger(__name__)

class DeepSeekVL2ImageTokenCaculator(ImageTokenCaculator):
    def __init__(self, model_path: str):
        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.image_size = self.processor.image_size
        self.patch_size = self.processor.patch_size
        self.downsample_ratio = self.processor.downsample_ratio

    def get_num_image_tokens(self, image_size: tuple[int, int], cropping: bool = True) -> int:
        height, width = image_size
        image_size = self.image_size
        patch_size = self.patch_size
        downsample_ratio = self.downsample_ratio

        if cropping:
            best_width, best_height = self.processor.select_best_resolution((width, height))
            num_width_tiles, num_height_tiles = (best_width // image_size, best_height // image_size)
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        local_views_tokens = (num_height_tiles * h) * (num_width_tiles * w + 1)
        return global_views_tokens + local_views_tokens + 1

class MlpProjector(nn.Module):
    def __init__(self, cfg: MlpProjectorConfig):
        super().__init__()

        self.cfg = cfg
        assert not cfg.token_pooling, (
            "Token pooling is not supported currently.")

        if cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [
                nn.Linear(
                    cfg.input_dim * cfg.downsample_ratio *
                    cfg.downsample_ratio, cfg.n_embed * mlp_ratio)
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(cfg.n_embed * mlp_ratio,
                              cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)
        else:
            raise NotImplementedError(
                f"Unsupported projector type: {cfg.projector_type}")

        self.layers = modules

    def forward(self, x):
        bs, hw, input_dim = x.shape
        h = w = int((hw)**0.5)
        """compute padding"""
        if h % self.cfg.downsample_ratio:
            pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
        else:
            pad = 0
        x = x.reshape(bs, h, w, input_dim)
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)
        """4 to 1 concat"""
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = F.unfold(x,
                     kernel_size=self.cfg.downsample_ratio,
                     stride=self.cfg.downsample_ratio,
                     padding=0)  # B, C*4, HW // 4
        x = x.permute(0, 2, 1)

        return self.layers(x)

class DeepSeekVL2VisionModel(VisionModel):
    def __init__(self, path: str, dtype: torch.dtype, device: torch.device):
        self.config = DeepseekVLV2Config.from_pretrained(path)
        self.processor = DeepseekVLV2Processor.from_pretrained(path)
        torch.set_default_dtype(dtype)
        with torch.device(device):
            self.vision = timm.create_model(
                "vit_so400m_patch14_siglip_384.webli",
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=True,
            )
            self.projector = MlpProjector(self.config.projector_config)
            self.projector_config = self.config.projector_config
        torch.set_default_dtype(torch.float)
        self.tile_tag = self.config.tile_tag
        self.global_view_pos = self.config.global_view_pos

        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(
            torch.tensor(self.projector_config.n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            with torch.device(device):
                # <|view_separator|>, <|\n|>
                self.image_newline = torch.randn(self.projector_config.n_embed) * embed_std
                # This is a typo in original implementation
                self.view_seperator = torch.randn(self.projector_config.n_embed) * embed_std

        load_safetensor(
            model_with_prefix_list=[
                (self.vision, 'vision.'),  
                (self.projector, 'projector.'), 
            ], 
            param_with_name_list=[
                (self.image_newline, 'image_newline'), 
                (self.view_seperator, 'view_seperator'), 
            ],
            model_weights_path=path, 
        )
        self.vision.to(dtype)
        self.vision.eval()
        self.projector.to(dtype)
        self.projector.eval()


    def forward(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        # [batch_all_tiles, 3, height, width]
        pixel_values = torch.cat(pixel_values, dim=0)
        
        # [batch_all_tiles, vit_seq_len, c]
        images_feature = self.vision.forward_features(pixel_values)

        # [batch_all_tiles, hw, D]
        images_embeds = self.projector(images_feature)

        # images_spatial_crop
        images_spatial_crop = []
        for height, width in model_params.original_image_sizes:
            best_width, best_height = self.processor.select_best_resolution((width, height))
            num_width_tiles, num_height_tiles = (best_width // self.processor.image_size,
                                                best_height // self.processor.image_size)
            images_spatial_crop.append((num_width_tiles, num_height_tiles))
        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)

        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        # fill image token based on self.tile_tag & self.global_view_pos
        tile_index = 0
        vision_embeddings = []
        for jdx in range(images_spatial_crop.size(0)):
            # extra global & local features
            num_width_tiles, num_height_tiles = images_spatial_crop[jdx]
            if num_width_tiles == 0 or num_height_tiles == 0:
                break
            num_tiles_in_image = num_width_tiles * num_height_tiles

            # [hw, D]
            global_features = images_embeds[tile_index]

            # [num_height_tiles * num_width_tiles, hw, D]
            local_features = images_embeds[tile_index + 1:tile_index + 1 +
                                           num_tiles_in_image]
            tile_index += num_tiles_in_image + 1

            # format global and local features
            # ----------------- global view add newline -----------------
            # [hw, D] -> [h, w, D]
            global_features = global_features.view(h, w, n_dim)

            # [D]     -> [h, 1, D]
            new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)

            # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
            global_features = torch.cat([global_features, new_lines_in_global],
                                        dim=1)

            # [h, w + 1, D] -> [h * (w + 1), D]
            global_features = global_features.view(-1, n_dim)

            # ----------------- local view add newline -----------------
            # [num_height_tiles * num_width_tiles, h * w, D] ->
            # [num_height_tiles * h, num_width_tiles * w, D]
            local_features = rearrange(local_features,
                                       "(th tw) (h w) d -> (th h) (tw w) d",
                                       th=num_height_tiles,
                                       tw=num_width_tiles,
                                       h=h,
                                       w=w)

            # [D] -> [num_height_tiles * h, 1, D]
            new_lines_in_local = repeat(self.image_newline,
                                        "d -> (th h) 1 d",
                                        th=num_height_tiles,
                                        h=h)

            # [num_height_tiles * h, num_width_tiles * w + 1, D]
            local_features = torch.cat([local_features, new_lines_in_local],
                                       dim=1)

            # [num_height_tiles * h, num_width_tiles * w + 1, D]
            #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
            local_features = local_features.view(-1, n_dim)

            # merge global and local tiles
            if self.global_view_pos == "head":
                global_local_features = torch.cat([
                    global_features,
                    self.view_seperator[None, :],
                    local_features,
                ])
            else:
                global_local_features = torch.cat([
                    local_features,
                    self.view_seperator[None, :],
                    global_features,
                ])

            vision_embeddings.append(global_local_features)

        output = VisionModelOutput()
        output.image_features = torch.cat(vision_embeddings, dim=0).to(dtype=torch.float16)
        return output

class DeepSeekVL2LanguageModel(LanguageModel):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.model = DeepseekForCausalLM.from_safetensor(model_path, dtype, device)

    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        sample_token_ids = self.model(input_ids, image_features, position_ids, model_params)
        return LanguageModelOutput(sample_token_ids=sample_token_ids)

class DeepSeekVL2Tokenizer(Tokenizer):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        current_script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(current_script_path)
        template_path = os.path.join(script_dir, "chat_template", "template_deepseek_vl2.jinja")
        with open(template_path, 'r', encoding='utf-8') as file:
            self.chat_template = Template(file.read())

    def encode(self, prompt: str) -> list[int]:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return token_ids

    def decode(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        prompt = self.chat_template.render(
            messages = messages, 
            bos_token = self.bos_token, 
            eos_token = self.eos_token,
            add_generation_prompt = True, 
        )
        return prompt

class DeepSeekVL2ImageProccsor(ImageProcessor):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.processor = DeepseekVLV2Processor.from_pretrained(path)

    def process(self, image: Image.Image) -> Tensor:
        images_tensor: Tensor = self.processor(
            text="", 
            images = image, 
            return_tensors="pt"
        )['pixel_values']
        return images_tensor

class DeepSeekVL2ModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        self.path = config.path
        self.dtype = str2dtype(config.dtype)
        self.device = str2device(config.device)
        
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.image_token_id = tokenizer.vocab["<image>"]

    def getVisionModel(self) -> VisionModel:
        return DeepSeekVL2VisionModel(self.path, self.dtype, self.device)

    def getLanguageModel(self) -> LanguageModel:
        return DeepSeekVL2LanguageModel(self.path, self.dtype, self.device)

    def getVisionModelConfig(self) -> VisionModelConfig:
        config = VisionModelConfig(
            image_token = "<image>",
            image_token_id = self.image_token_id,
            image_token_caculator = DeepSeekVL2ImageTokenCaculator(self.path),
        )
        return config

    def getLanguageModelConfig(self) -> LanguageModelConfig:
        config_ref = DeepseekVLV2Config.from_pretrained(self.path).text_config

        n_layers = config_ref.num_hidden_layers
        max_position_embeddings = config_ref.max_position_embeddings
        hidden_size = config_ref.hidden_size
        n_qo_heads = config_ref.num_attention_heads
        n_kv_heads = config_ref.num_key_value_heads
        assert hidden_size % n_qo_heads == 0
        head_dim = hidden_size // n_qo_heads

        tokenizer = AutoTokenizer.from_pretrained(self.path)
        config = LanguageModelConfig(
            n_layers = n_layers,
            max_position_embeddings = max_position_embeddings, 
            n_qo_heads = n_qo_heads,
            n_kv_heads = n_kv_heads,
            head_dim = head_dim,
            eos_token_id = tokenizer.eos_token_id, 
        )
        return config

    def getProcessor(self) -> ImageProcessor:
        from hydrainfer.model.processor import TransformersAutoProcessorAdapter
        return TransformersAutoProcessorAdapter(self.path)
        # return DeepSeekVL2ImageProccsor(self.path)

    def getTokenizer(self) -> Tokenizer:
        return DeepSeekVL2Tokenizer(self.path)

    def getModelProfiler(self) -> ModelProfiler:
        return VisionLanguageModelProfiler(self.path, vision_model_prefixes=['vision.', 'projector.', 'image_newline', 'view_seperator'], language_model_prefixes=['language.'])