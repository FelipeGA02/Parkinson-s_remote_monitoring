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

#print(parkinsons_telemonitoring.metadata)
#print(parkinsons_telemonitoring.variables)

# =========================
# VISÃO GERAL DO DATASET
# =========================
print("="*50)
print("📊 VISÃO GERAL DO DATASET")
print("="*50)

num_registros, num_atributos = df.shape

print(f"🔢 Registros (linhas): {num_registros}")
print(f"🧩 Atributos (colunas): {num_atributos}")


# =========================
# LISTA DE ATRIBUTOS
# =========================
print("\n" + "="*50)
print("📌 LISTA DE ATRIBUTOS")
print("="*50)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:02d}. {col}")


# =========================
# TIPOS E NULOS
# =========================
print("\n" + "="*50)
print("🧪 TIPOS DE DADOS E VALORES NULOS")
print("="*50)

info_df = pd.DataFrame({
    "Atributo": df.columns,
    "Tipo": df.dtypes.values,
    "Valores Nulos": df.isnull().sum().values,
    "Não Nulos": df.notnull().sum().values
})

print(info_df.to_string(index=False))


# =========================
# ESTATÍSTICAS DESCRITIVAS
# =========================
print("\n" + "="*50)
print("📈 ESTATÍSTICAS DESCRITIVAS")
print("="*50)

print(df.describe().T)