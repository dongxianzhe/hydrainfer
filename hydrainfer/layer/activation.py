import torch
from torch import nn, Tensor
import math
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)

try:
    from hydrainfer._C.kernel.activation import silu as silu_kernel
except ImportError:
    logger.warning('silu kernel import failed')
    silu_kernel = None

class NewGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)

def silu(h: Tensor):
    if h.device == torch.device('cpu'):
        return torch.nn.functional.silu(h)
    if silu_kernel is not None:
        return silu_kernel(h)
    return torch.nn.functional.silu(h)

try:
    import flashinfer
except ImportError as e:
    logger.warning('import flashinfer activation kernel failed')
    flashinfer = None

if flashinfer is not None:
    try:
        device = torch.device('cuda:0')
        n_tokens = 1
        hidden_size = 128
        x = torch.randn(size=(1, hidden_size * 2), dtype=torch.half, device=device)
        _ = flashinfer.activation.silu_and_mul(x)
    except Exception as e:
        logger.warning('flashinfer activation kernel pre forward failed')
        flashinfer = None

class SiluAndMul(nn.Module):
    def __init__(
        self, 
        enable_flashinfer: bool = True, 
    ):
        super().__init__()
        self.enable_flashinfer = enable_flashinfer
    
    def forward(self, x: Tensor) -> Tensor:
        # x (n_tokens, hidden_size * 2)
        hidden_size = x.shape[1] // 2
        if x.is_cuda:
            if self.enable_flashinfer and flashinfer is not None:
                return flashinfer.activation.silu_and_mul(x)
            return silu(x[:, :hidden_size]) * x[:, hidden_size:]
        else:
            return nn.functional.silu(x[:, :hidden_size]) * x[:, hidden_size:]