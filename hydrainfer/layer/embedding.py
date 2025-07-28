import torch
from torch import nn, Tensor
from hydrainfer.model_parallel.process_group import ProcessGroup

class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, process_group: ProcessGroup):
        super().__init__()
        self.process_group = process_group
        assert embed_dim % self.process_group.world_size == 0, f"output_features {embed_dim} is not divisble by world_size {self.process_group.world_size}"
        self.weight = nn.Parameter(torch.empty(size=(vocab_size, embed_dim // self.process_group.world_size)), requires_grad=False)
    
    def forward(self, x: Tensor) -> Tensor:
        o = nn.functional.embedding(x, self.weight)
        if self.process_group.world_size > 1:
            o = self.process_group.gather_from_tensor_parallel_region(o)
        return o