import matplotlib.pyplot as plt

def histogram(data: list[float]):
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.savefig('histogram.png')
