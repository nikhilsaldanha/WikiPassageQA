import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("./data/extracted/webap_queries.csv")
    freq = data['QID'].value_counts().values
    print(freq)
    plt.bar(range(80), freq)
    plt.savefig("mygraph.png")