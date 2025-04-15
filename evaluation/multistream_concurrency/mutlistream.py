import time
import torch
import hydra
import asyncio
import argparse
from dxz.engine import RequestControlBlock, InstructionListBuilder, ImageEmbed, TextFill, OutputTokenProcessor
from dxz.request import SamplingParameters
from dxz.cluster.async_epdnode import AsyncEPDNode, NodeConfig, NodeContext

async def main(args):
    config = NodeConfig()
    print(config)
    context = NodeContext(world_size=1, rank=0)
    node = AsyncEPDNode(config, context)
    await node.init()

    print('init finished')
    n_encode_request = args.n_encode_request
    n_decode_request = args.n_decode_request
    # create n_encode_request requests, each request has one image embed task
    n_iter = 100
    rcbs = []
    for i in range(n_encode_request):
        inst_builder = InstructionListBuilder()
        for j in range(n_iter):
            inst_builder.append(ImageEmbed(
                pixel_values=torch.randn(size=(1, 3, 336, 336), dtype=torch.half, device=torch.device('cuda:0')), 
                cache_ids = list(range(576)), 
                token_pruning_params=None
            ))
        rcb = RequestControlBlock(instructions=inst_builder.build_instruction_list(), sampling_params=SamplingParameters(), request_id=0, output_token_params=OutputTokenProcessor())
        rcbs.append(rcb)
    
    # create n_decode_request requests, each request has one decode task
    for i in range(n_decode_request):
        inst_builder = InstructionListBuilder()
        for j in range(n_iter):
            inst_builder.append(TextFill(
                token_ids = [0], position_ids=[576], cache_ids=[576], sample=False, sample_dst=None
            ))
        rcb = RequestControlBlock(instructions=inst_builder.build_instruction_list(), sampling_params=SamplingParameters(), request_id=0, output_token_params=OutputTokenProcessor())
        rcbs.append(rcb)

    for rcb in rcbs:
        node.batch_scheduler.schedule_new(rcb)

    # warm up
    n_warm_up_iter = 3
    for j in range(n_warm_up_iter):
        await node.step()

    total_dur = 0.
    n_test_iter = n_iter - n_warm_up_iter - 1
    for j in range(n_test_iter):
        start = time.perf_counter()
        await node.step()
        end = time.perf_counter()
        dur = end - start
        total_dur += dur
        print(f'dur {dur}')
    print(f'avg {total_dur / n_test_iter}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="offlne epd disaggregate test", conflict_handler='resolve')
    parser.add_argument(f'--n-encode-request', type=int, default=8, help='number of encode request')
    parser.add_argument(f'--n-decode-request', type=int, default=8, help='number of decode request')

    args = parser.parse_args()
    asyncio.run(main(args))