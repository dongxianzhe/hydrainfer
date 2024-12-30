import torch
from dxz.model.mixtral import MixtralSparseMoeBlock
from dxz.utils.profiler import profile
from transformers import MixtralConfig


device=torch.device('cuda:0')
streams = [torch.cuda.Stream() for i in range(2)]

config = MixtralConfig()
model = MixtralSparseMoeBlock(config)
model.to(device=device)
model.half()

n_tokens = 1024
hidden_states = torch.randn(size=(n_tokens, config.hidden_size), dtype=torch.half, device=device)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--multistream', action='store_true', default=False)
    args = parser.parse_args()

    if args.multistream:
        with profile('multi stream forward many times'):
            for i in range(100):
                model(hidden_states, streams)
    else:
        with profile('singles stream forward many times'):
            for i in range(100):
                model(hidden_states)

