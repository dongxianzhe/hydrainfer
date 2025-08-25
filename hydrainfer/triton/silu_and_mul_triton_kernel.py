import torch
import triton
import triton.language as tl


@triton.jit
def silu_and_mul_triton_kernel(
    gateup_output,
    down_input,
    hidden_size,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2

    pid = tl.program_id(0)
    expert_id = tl.load(reorder_topk_ids + pid)
    if expert_id >= start_expert_id and expert_id <= end_expert_id:
        gateup_output_ptr = gateup_output + pid * hidden_size
        gate_output_ptr = gateup_output_ptr
        up_output_ptr = gateup_output_ptr + half_hidden_size
        down_input_ptr = down_input + pid * half_hidden_size

        if scales is not None:
            scale = tl.load(scales + expert_id - start_expert_id)
            scale = (1 / scale).to(InDtype)
        else:
            scale = 1

        for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < half_hidden_size

            gate_output = tl.load(gate_output_ptr + offset,
                                  mask=mask).to(tl.float32)
            up_output = tl.load(up_output_ptr + offset, mask=mask)

            # silu & mul & quantize
            gate_output = gate_output * tl.sigmoid(gate_output)
            gate_output = gate_output.to(InDtype)

            silu_mul_output = gate_output * up_output * scale
            silu_mul_output = silu_mul_output.to(OutDtype)
            tl.store(down_input_ptr + offset, silu_mul_output, mask=mask)


def silu_and_mul(gateup_output: torch.Tensor,
                 down_input: torch.Tensor,
                 hidden_size: int,
                 reorder_topk_ids: torch.Tensor,
                 scales: torch.Tensor,
                 start_expert_id: int,
                 end_expert_id: int,
                 BLOCK_SIZE=128) -> None:
    ret_kernel = silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
        gateup_output,
        down_input,
        hidden_size,
        reorder_topk_ids,
        scales,
        start_expert_id,
        end_expert_id,
        BLOCK_SIZE,
    )
    # print(f'ret_kernel.metadata.shared: {ret_kernel.metadata.shared}')

    return gateup_output, ret_kernel.metadata.shared


float_dtype = torch.float32
half_dtype = torch.bfloat16
int_dtype = torch.int32
start_expert_id = 0
end_expert_id = 32
topk = 6
hidden_size = 2816

gateup_output = torch.full((186, 2816), 1.5, dtype=half_dtype, device='cuda')
down_input = torch.full((186, 1408), 11.5, dtype=torch.float8_e4m3fn, device='cuda')

reorder_topk_values = [
    0, 2, 2, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8,
    8, 8, 8, 8, 8, 9, 10, 10, 10, 13, 13, 13, 13, 13, 13, 13, 14, 15, 15, 15,
    16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 18, 19, 19, 19, 20, 20, 20, 20, 21,
    22, 22, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26,
    27, 27, 27, 27, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33,
    34, 34, 34, 35, 35, 35, 35, 36, 36, 38, 38, 39, 39, 39, 39, 39, 39, 40, 40,
    40, 41, 43, 43, 43, 44, 44, 44, 44, 45, 46, 46, 46, 50, 50, 50, 50, 51, 51,
    52, 52, 52, 52, 52, 53, 53, 55, 55, 55, 55, 55, 55, 55, 55, 55, 57, 57, 57,
    57, 57, 57, 57, 57, 59, 59, 59, 59, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61,
    62, 62, 62, 62, 62, 62, 62, 63
]

reorder_topk_ids = torch.tensor(reorder_topk_values,
                                dtype=int_dtype,
                                device='cuda')

scales = torch.rand(32, dtype=float_dtype, device='cuda')

output_tensor, shared_mem_bytes = silu_and_mul(gateup_output, down_input,
                                               hidden_size, reorder_topk_ids,
                                               scales, start_expert_id,
                                               end_expert_id)

print(f'output_tensor.shape {output_tensor.shape}')