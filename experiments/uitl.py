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