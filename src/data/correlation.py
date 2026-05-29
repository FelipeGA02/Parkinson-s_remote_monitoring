import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo


parkinsons_telemonitoring = fetch_ucirepo(id=189)

X = parkinsons_telemonitoring.data.features
y = parkinsons_telemonitoring.data.targets

df = pd.concat([X, y], axis=1)

# heatmap
def correl():
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Matriz de correlação")
    plt.show()

# único
def correl_target(target):
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr[[target]].sort_values(by=target, ascending=False),
        annot=True,
        cmap="coolwarm"
    )
    plt.show()
for col in df.columns:
    correl_target(col)

