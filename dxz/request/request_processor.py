import random
import argparse
from torch import Tensor
from dataclasses import dataclass, fields
from transformers import AutoTokenizer, AutoProcessor
from dxz.engine.isa import Instruction, TextFill, ImageFill, Mov, ReAlloc, EmptyInstruction, ImageEmbed, ImageEmbedFill, InstructionList, InstructionListBuilder
from PIL import Image
from typing import Literal, Optional
from dxz.request.request import Request
from dxz.request.rcb import RequestControlBlock


@dataclass
class RequestProcessorConfig:
    disaggregate_embed_prefill: bool = True

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'RequestProcessorConfig':
        attrs = [attr.name for attr in fields(cls)]
        config = cls(**{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--disaggregate-embed-prefill', action='store_true', help='Enable disaggregation of embedding prefill.')
        return parser


@dataclass
class RequestProcessorContext:
    tokenizer: AutoTokenizer
    processor: AutoProcessor
    image_token_id: int
    num_image_tokens: int
    n_layers: int


class RequestProcessor:
    def process(self, request: Request) -> RequestControlBlock:
        raise Exception('interface not implemented')