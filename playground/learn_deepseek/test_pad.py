import torch
import torch.nn.functional as F

# 假设 h 是一个已定义的张量
h = torch.randn(5, 3)  # 假设 h 是一个形状为 (5, 3) 的张量
max_num_token = 10

# 使用 F.pad 进行填充
if h is not None:
    padded_h = F.pad(
        h,
        (0, 0, 0, max_num_token - h.size(0)),  # pad (left, right, top, bottom)
        "constant",  # 填充方式为常数填充（默认为0）
        0            # 填充值为 0
    )

print(padded_h)
