from typing import Union, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch import Tensor
import numpy
import seaborn

def histogram(data: Union[list[float], Tensor], range:Optional[tuple[int, int]]=None, fig_size: int=5, bins: int=30, name: str='histogram'):
    if isinstance(data, Tensor):
        # torch.type() return a str, like torch.cuda.FloatTensor, torch.cuda.IntTensor, torch.FloatTensor, torch.IntTensor
        if data.type().endswith('FloatTensor'):
            data = data.view(-1).to(torch.float).to(torch.device('cpu')).detach()
        elif data.type().endswith('torch.IntTensor'):
            data = data.view(-1).to(torch.int).to(torch.device('cpu')).detach()
        else:
            raise RuntimeError(f'invalid data.type(): {data.type()}')

    plt.figure(figsize=(fig_size, fig_size))
    plt.hist(data, range=range, bins=bins, color='skyblue', edgecolor='black')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(name)
    plt.savefig(name)
    plt.close()

def lognorm_attention_score_heatmap(score: Tensor, name: str='attention_score_heatmap', fig_size:int=5, vmin=None, vmax=None):
    assert score.dim() == 2, "score should be a matrix"
    score = score.to(torch.float).to(torch.device('cpu')).detach()
    cmap = plt.get_cmap("Reds")
    plt.figure(figsize=(fig_size, fig_size))

    ax = seaborn.heatmap(score,
                cmap=cmap,  # custom color map
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cbar_kws={'label': name},
                )

    # change the x tinks font size
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # make y label vertical
    plt.yticks(rotation=0)
    # plt.xticks(rotation=90)     

    # tight layout
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def attention_score_heatmap(score: Tensor, name: str='attention_score_heatmap', fig_size:int=5):
    assert score.dim() == 2, "score should be a matrix"
    score = score.to(torch.float).to(torch.device('cpu')).detach()
    cmap = plt.get_cmap("Reds")
    plt.figure(figsize=(fig_size, fig_size))

    ax = seaborn.heatmap(score,
                cmap=cmap,  # custom color map
                cbar_kws={'label': name},
                )

    # change the x tinks font size
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # make y label vertical
    plt.yticks(rotation=0)
    # plt.xticks(rotation=90)     

    # tight layout
    plt.savefig(name, bbox_inches='tight')
    plt.close()