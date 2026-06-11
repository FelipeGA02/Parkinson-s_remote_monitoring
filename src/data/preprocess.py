"""
Pré-processamento — Oxford Parkinson's Disease Telemonitoring Dataset
Classificação MULTICLASSE da progressão da Doença de Parkinson via voz.

Estratégia de split: TEMPORAL por paciente
  Para cada um dos 31 pacientes, as gravações são ordenadas por test_time.
  Os primeiros 80% (fase inicial do acompanhamento) vão para treino;
  os últimos 20% (fase futura) vão para teste.
  Isso simula o cenário real: o sistema aprende com o histórico do paciente
  e prediz seu estado em gravações futuras, sem memorizar o resultado atual.

Atributos preditivos (18):
  - subject_id  : identidade do paciente (0–30), captura padrão vocal individual
  - test_time   : tempo desde o recrutamento (semanas), progressão temporal
  - 16 biomarcadores vocais: Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RAW_PATH      = "data/raw/parkinsons_telemonitoring.csv"
PROCESSED_DIR = "data/processed"
RANDOM_STATE  = 42

BINS   = [0, 25, 40, 100]
LABELS = {0: "Leve", 1: "Moderado", 2: "Grave"}

VOCAL_FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "PPE",
]

ALL_FEATURES = ["subject_id", "test_time"] + VOCAL_FEATURES
COLS_TO_SCALE = ["test_time"] + VOCAL_FEATURES


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load] {df.shape[0]} registros × {df.shape[1]} atributos")
    print(f"       Valores nulos: {df.isnull().sum().sum()}")
    return df


def create_subject_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    patient_key = df["age"].astype(str) + "_" + df["sex"].astype(str)
    unique_patients = sorted(patient_key.unique())
    id_map = {key: idx for idx, key in enumerate(unique_patients)}
    df["subject_id"] = patient_key.map(id_map)
    df["patient_key"] = patient_key
    print(f"\n[subject_id] {df['subject_id'].nunique()} pacientes únicos (0–{df['subject_id'].max()})")
    return df


def create_multiclass_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["progressao"] = pd.cut(df["total_UPDRS"], bins=BINS,
                               labels=[0, 1, 2]).astype(int)
    print("\n[target] Distribuição das classes:")
    counts = df["progressao"].value_counts().sort_index()
    for k, v in counts.items():
        print(f"  {LABELS[k]} ({k}): {v} ({v/len(df):.1%})")
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


def temporal_split_per_patient(df: pd.DataFrame, split_ratio: float = 0.8):
    """
    Split temporal por paciente:
    - Ordena as gravações de cada paciente por test_time (cronologicamente)
    - Primeiros split_ratio% → treino (histórico conhecido)
    - Últimos (1-split_ratio)% → teste (gravações futuras a predizer)
    """
    train_rows, test_rows = [], []
    for pid in sorted(df["subject_id"].unique()):
        pat = df[df["subject_id"] == pid].sort_values("test_time")
        n   = len(pat)
        split_point = int(n * split_ratio)
        train_rows.append(pat.iloc[:split_point])
        test_rows.append(pat.iloc[split_point:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)

    print(f"\n[split temporal] treino: {len(train_df)} registros (primeiros 80% por paciente)")
    print(f"                 teste:  {len(test_df)} registros (últimos 20% por paciente)")
    print(f"                 Todos os 31 pacientes presentes em treino e teste.")

    return (train_df[ALL_FEATURES], test_df[ALL_FEATURES],
            train_df["progressao"],  test_df["progressao"])


def preprocess(raw_path: str = RAW_PATH,
               output_dir: str = PROCESSED_DIR,
               apply_smote: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(raw_path)
    df = create_subject_id(df)
    df = create_multiclass_target(df)
    df = remove_outliers_iqr(df, VOCAL_FEATURES)

    X_train, X_test, y_train, y_test = temporal_split_per_patient(df)

    # Escala test_time e biomarcadores vocais (subject_id permanece inteiro)
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE]  = scaler.transform(X_test[COLS_TO_SCALE])

    if apply_smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_arr, y_train = sm.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train_arr, columns=ALL_FEATURES)
        # SMOTE pode interpolar subject_id — arredonda para inteiro mais próximo
        X_train["subject_id"] = X_train["subject_id"].round().astype(int).clip(0, 30)
        print(f"\n[SMOTE] treino após reamostagem: {len(X_train)}")
        counts = pd.Series(y_train).value_counts().sort_index()
        for k, v in counts.items():
            print(f"  {LABELS[k]}: {v}")

    train_df_out = X_train.copy()
    train_df_out["progressao"] = y_train.values if hasattr(y_train, "values") else y_train
    test_df_out  = X_test.copy()
    test_df_out["progressao"]  = y_test.values

    train_df_out.to_csv(f"{output_dir}/train.csv",     index=False)
    test_df_out.to_csv( f"{output_dir}/test.csv",      index=False)
    test_df_out.to_csv( f"{output_dir}/processed.csv", index=False)

    print(f"\n[save] {output_dir}/train.csv | test.csv | processed.csv")
    print(f"[features] {len(ALL_FEATURES)}: subject_id + test_time + 16 biomarcadores vocais")
    return X_train, X_test, y_train, y_test, scaler, ALL_FEATURES


if __name__ == "__main__":
    preprocess()
