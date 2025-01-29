import torch
from torch import nn, Tensor
from dxz.model_parallel.process_group import ProcessGroup, ParallelConfig


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size=(out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.randn(size=(out_features, )))
        else:
            self.bias = None
    
    def forward(self, h: Tensor) -> Tensor: 
        return torch.nn.functional.linear(h, self.weight, bias=self.bias)


class RowParallelLinear(nn.Module):
    """
        Linear layer with row parallelism.
            The linear layer is defined as Y = XA + b. A is parallelized along
            its first dimension and X along its second dimension as:
                    -   -
                    | A_1 |
                    | .   |
                A = | .   |       X = [X_1, ..., X_p]
                    | .   |
                    | A_p |
                    -   -
    """
    def __init__(self, in_features: int, out_features: int, bias: bool, input_is_parallelized: bool, process_group: ProcessGroup):
        super().__init__()
        self.tp_size = process_group.world_size
        self.tp_rank = process_group.rank
        self.process_group = process_group

        assert in_features % self.tp_size == 0, f"in_features {in_features} is not divisible by tp_size {self.tp_size}"
        in_features_per_partition = in_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(size=(out_features, in_features_per_partition)), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(size=(out_features, )), requires_grad=False)
        else:
            self.bias = None
        self.input_is_parallelized = input_is_parallelized
        

    def forward(self, input: Tensor) -> Tensor:
        if not self.input_is_parallelized:
            input = self.process_group.scatter_to_tensor_parallel_region(input)
        output = torch.nn.functional.linear(input, self.weight, bias=None)
        if self.tp_size > 1:
            output = self.process_group.reduce_from_tensor_parallel_region(output)
        # N.B. need to apply bias after the reduce
        if self.bias is not None:
            output.add_(self.bias)
        return output

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        weight = state_dict['weight']
        assert weight.dim() == 2, f'weight dim should be 2 but got {weight.shape}'
        assert weight.shape[1] % self.tp_size == 0, f'weight {weight.shape} second dim is not divisible by tp_size {self.tp_size}'
        chunks = weight.chunk(self.tp_size, dim=1)
        self.weight.copy_(chunks[self.tp_rank])

        if self.bias is not None:
            bias = state_dict['bias']
            assert bias.dim() == 1, f'bias dim should be 1 but got {bias.shape}'
            self.bias.copy_(bias)


class ColumnParallelLinear(nn.Module):
    """
        Linear layer with column parallelism.
        The linear layer is defined as Y = XA + b. A is parallelized along
        its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, in_features: int, out_features: int, bias: bool, gather_output: bool, process_group: ProcessGroup):
        super().__init__()
        self.tp_rank = process_group.rank
        self.tp_size = process_group.world_size
        self.process_group = process_group
        assert out_features % self.tp_size == 0, f"out_features {out_features} is not divisible by tp_size {self.tp_size}"
        out_features_per_partition = out_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(size=(out_features_per_partition, in_features)), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(size=(out_features_per_partition, )), requires_grad=False)
        else:
            self.bias = None
        self.gather_output = gather_output
        
    def forward(self, input: Tensor) -> Tensor:
        output = torch.nn.functional.linear(input, self.weight, bias=self.bias)
        if self.tp_size > 1 and self.gather_output:
            output = self.process_group.gather_from_tensor_parallel_region(output)
        return output

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        weight = state_dict['weight']
        assert weight.dim() == 2, f'weight dim should be 2 but got {weight.shape}'
        assert weight.shape[0] % self.tp_size == 0, f'weight {weight.shape} first dim is not divisible by tp_size {self.tp_size}'
        chunks = weight.chunk(self.tp_size, dim=0)
        self.weight.copy_(chunks[self.tp_rank])

        if self.bias is not None:
            bias = state_dict['bias']
            assert bias.dim() == 1, f'bias dim should be 1 but got {bias.shape}'
            assert bias.shape[0] % self.tp_size == 0, f'bias {bias.shape} is not disibile by tp_size {self.tp_size}'
            chunks = bias.chunk(self.tp_size, dim=0)
            self.bias.copy_(chunks[self.tp_rank])