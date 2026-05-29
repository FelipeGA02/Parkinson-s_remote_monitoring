"""
Pré-processamento — Oxford Parkinson's Disease Telemonitoring Dataset
Classificação MULTICLASSE da progressão da Doença de Parkinson via voz.

IMPORTANTE — avaliação correta:
  O dataset possui múltiplas gravações de 31 pacientes distintos.
  O split DEVE ser feito por paciente (GroupKFold), nunca por registro,
  para evitar data leakage e resultados artificialmente inflados.

Etapas:
  1. Carregamento do CSV bruto
  2. Discretização da variável-alvo (total_UPDRS → 3 classes clínicas)
  3. Seleção apenas dos 16 atributos vocais (sem age, sex, test_time)
  4. Tratamento de outliers (IQR × 3)
  5. Normalização (StandardScaler)
  6. Split por paciente: GroupShuffleSplit 80/20
  7. Balanceamento com SMOTE (apenas no treino)
  8. Exportação para data/processed/
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RAW_PATH      = "data/raw/parkinsons_telemonitoring.csv"
PROCESSED_DIR = "data/processed"
RANDOM_STATE  = 42

BINS   = [0, 25, 40, 100]
LABELS = {0: "Leve", 1: "Moderado", 2: "Grave"}

# Apenas atributos biomédicos de voz — age, sex e test_time excluídos
# - age/sex: identificam o paciente → vazam identidade, não generalizam
# - test_time: índice temporal, não é biomarcador vocal
VOCAL_FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "PPE",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] {df.shape[0]} registros × {df.shape[1]} atributos")
    print(f"       Valores nulos: {df.isnull().sum().sum()}")
    return df


def create_multiclass_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["progressao"] = pd.cut(df["total_UPDRS"], bins=BINS,
                               labels=[0, 1, 2]).astype(int)
    df["patient_id"] = df["age"].astype(str) + "_" + df["sex"].astype(str)
    print("\n[target] Distribuição das classes:")
    counts = df["progressao"].value_counts().sort_index()
    for k, v in counts.items():
        print(f"  {LABELS[k]} ({k}): {v} ({v/len(df):.1%})")
    print(f"  Pacientes distintos (age+sex): {df['patient_id'].nunique()}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, features: list,
                        factor: float = 3.0) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col in features:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask &= df[col].between(Q1 - factor * IQR, Q3 + factor * IQR)
    print(f"\n[outliers] {(~mask).sum()} registros removidos (IQR × {factor})")
    return df[mask].reset_index(drop=True)


def preprocess(raw_path: str = RAW_PATH,
               output_dir: str = PROCESSED_DIR,
               apply_smote: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(raw_path)
    df = create_multiclass_target(df)
    df = remove_outliers_iqr(df, VOCAL_FEATURES)

    X = df[VOCAL_FEATURES]
    y = df["progressao"]
    groups = df["patient_id"]

    # Split por PACIENTE para evitar data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_pts = df.iloc[train_idx]["patient_id"].nunique()
    test_pts  = df.iloc[test_idx]["patient_id"].nunique()
    print(f"\n[split] treino: {len(X_train)} registros / {train_pts} pacientes")
    print(f"        teste:  {len(X_test)} registros / {test_pts} pacientes")

    # Verificação de sobreposição — deve ser vazio
    overlap = (set(df.iloc[train_idx]["patient_id"]) &
               set(df.iloc[test_idx]["patient_id"]))
    assert len(overlap) == 0, f"Leakage detectado: {overlap}"
    print("        Sobreposição de pacientes: nenhuma ✓")

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=VOCAL_FEATURES)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=VOCAL_FEATURES)

    if apply_smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_sc, y_train = sm.fit_resample(X_train_sc, y_train)
        print(f"\n[SMOTE] treino após reamostagem: {len(X_train_sc)}")
        counts = pd.Series(y_train).value_counts().sort_index()
        for k, v in counts.items():
            print(f"  {LABELS[k]}: {v}")

    train_df = X_train_sc.copy()
    train_df["progressao"] = y_train.values if hasattr(y_train, "values") else y_train
    test_df  = X_test_sc.copy()
    test_df["progressao"]  = y_test.values

    train_df.to_csv(f"{output_dir}/train.csv",     index=False)
    test_df.to_csv( f"{output_dir}/test.csv",      index=False)
    test_df.to_csv( f"{output_dir}/processed.csv", index=False)

    print(f"\n[save] {output_dir}/train.csv | test.csv | processed.csv")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, VOCAL_FEATURES


if __name__ == "__main__":
    preprocess()
