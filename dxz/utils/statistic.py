import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch import Tensor
import numpy
import seaborn

def histogram(data: list[float]):
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.savefig('histogram.png')

def attention_score_heatmap(score: Tensor, name: str='attention_score_heatmap', fig_size:int=5):
    assert score.dim() == 2, "score should be a matrix"
    score = score.to(torch.float).to(torch.device('cpu')).detach()
    cmap = plt.get_cmap("Reds")
    plt.figure(figsize=(fig_size, fig_size))

    log_norm = LogNorm(vmin=0.000001, vmax=score.max())
    
    ax = seaborn.heatmap(score,
                cmap=cmap,  # custom color map
                norm=log_norm,
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