from dataclasses import dataclass

@dataclass
class ParallelInfo:
    group_size: int
    num_group: int
    current_group_id: int
    rank_per_group: list[list[int]]
    rank: int
    domain: str
    buffer_size: int
    backend: str

@dataclass
class ParallelInfos:
    embed_tp: ParallelInfo
    embed_dp: ParallelInfo
    attn_tp: ParallelInfo
    attn_o_proj_tp: ParallelInfo
    attn_dp: ParallelInfo
    attn_o_proj_dp: ParallelInfo
    mlp_tp: ParallelInfo
    mlp_dp: ParallelInfo
    moe_tp: ParallelInfo
    moe_dp: ParallelInfo
    lm_head_tp: ParallelInfo
    lm_head_dp: ParallelInfo
    attn_inter_sp: ParallelInfo


# def map_devices(world_size: int, rank: int, dp_size: int, tp_size: int, moe_ep_size: int, moe_tp_size: int, pp_size: int, sp_size: int):
#     # num_machines
#     # world_size
#     # machine_world_size
#     # 

if __name__ == '__main__':
    print('hello world')
