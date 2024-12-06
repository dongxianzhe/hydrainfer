import matplotlib.pyplot as plt

def bar_chart(x, y, figsize=(8, 6), xlabel="", ylabel="", title="", filename="bar_chart"):
    plt.figure(figsize=figsize)
    bars = plt.bar(x, y, edgecolor='black', linewidth=2, color='#6fa7a9')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 100, int(yval), ha='center', va='bottom')

    plt.savefig(filename)

    plt.show()


def pie_chart(data: list[float], labels: list[str], title: str, filename: str):
    colors = ['#a35faf', '#4b8ac2', '#60b75d', '#c75d66']
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(data, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
    ax.axis('equal')
    plt.figtext(0.5, 0.01, title, ha='center', fontsize=12)
    plt.savefig(filename)

def distribution_bar_chart(xlabels: list[int], y: list[list[float]], y_labels: list[str], xlabel: str, ylabel: str, title: str, filename: str):
    # y is list of y values for each x label
    # eg.
    #     sequence_lengths = [128, 256, 512, 1000, 2000]
    #     times = [(20, 10, 20), (25, 5, 20), (40, 5, 15), (70, 5, 20), (120, 10, 20)]
    #     distribution_bar_chart(
    #         xlabels = sequence_lengths, 
    #         y = times,
    #         y_labels = ['linear', 'attention', 'others'], 
    #         xlabel = 'Sequence Length', 
    #         ylabel = 'Time (ms)', 
    #         title = 'Prefill', 
    #         filename = 'bar_chart.png', 
    #         )
    fig, ax = plt.subplots()
    n_xlabels = len(xlabels)
    colors = ['#6fa7a9', '#c6c6c6', '#e6938d']
    hatchs = ['\\'     , ''       , '/']

    bars = ax.bar(range(n_xlabels), [0 for i in range(n_xlabels)])
    for i, bar in enumerate(bars):
        x = bar.get_x() + bar.get_width() / 2
        start_height = 0
        width = bar.get_width()
        for j, h in enumerate(y[i]):
            ax.bar(x, h, width, start_height, color=colors[j], edgecolor='black', hatch=hatchs[j])
            start_height += h

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticks(range(n_xlabels))
    
    ax.set_xticklabels(xlabels)

    ax.set_ylim(0, 150)
    ax.set_yticks(np.arange(0, int(1.1 * max([sum(heights) for heights in y])), 50))

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 创建图例条目
    patchs = [plt.Rectangle((0, 0), 1, 1, fc=colors[i], edgecolor='black', hatch=hatchs[i]) for i in range(len(y_labels))]

    # 添加图例
    ax.legend(patchs, y_labels, loc='upper left')

    plt.savefig(filename)
