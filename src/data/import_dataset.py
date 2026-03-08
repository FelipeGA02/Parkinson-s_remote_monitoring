from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

parkinsons_telemonitoring = fetch_ucirepo(id=189)

X = parkinsons_telemonitoring.data.features
y = parkinsons_telemonitoring.data.targets

df = pd.concat([X, y], axis=1)

path = Path("data/raw")
path.mkdir(parents=True, exist_ok=True)

df.to_csv(path / "parkinsons_telemonitoring.csv", index=False)

print(parkinsons_telemonitoring.metadata)

print(parkinsons_telemonitoring.variables)