from torch import nn, Tensor


class UpDownMLP:
    def __init__(
        self, 
        up_proj: nn.Linear, 
        down_proj: nn.Linear, 
        activation: nn.Module
    ):
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.activation = activation

    def forward(self, h: Tensor) -> Tensor:
        # h (n_tokens, hidden_size)
        return self.down_proj(self.activation(self.up_proj(h)))


class GateUpDownMLP:
    def __init__(
        self, 
        gate_proj: nn.Linear, 
        up_proj: nn.Linear, 
        down_proj: nn.Linear, 
        activation: nn.Module
    ):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.activation = activation
    def forward(self, h: Tensor) -> Tensor:
        # h (n_tokens, hidden_size)
        return self.down_proj(self.activation(self.gate_proj(h)) * self.up_proj(h))
