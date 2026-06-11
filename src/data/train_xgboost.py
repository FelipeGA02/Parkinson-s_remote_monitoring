"""
Treinamento e avaliação — XGBoost Multiclasse
Doença de Parkinson · Telemonitoramento de Voz

Metodologia: Split Temporal por Paciente
  Para cada paciente, as gravações são ordenadas cronologicamente (test_time).
  Primeiros 80% → treino (histórico do acompanhamento).
  Últimos 20%  → teste (gravações futuras, nunca vistas pelo modelo).

  O subject_id é incluído como feature legítima: em telemonitoramento,
  o sistema JÁ conhece o paciente (viu suas gravações históricas).
  O split temporal garante que o modelo prediz estados FUTUROS, não memoriza
  estados já vistos — eliminando o problema de memorização por identidade.

Features (18): subject_id + test_time + 16 biomarcadores vocais.
"""

import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score,
                              accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from preprocess import (load_data, create_subject_id, create_multiclass_target,
                         remove_outliers_iqr, temporal_split_per_patient,
                         VOCAL_FEATURES, ALL_FEATURES, COLS_TO_SCALE)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

FIG_DIR      = "notebooks/data"
RANDOM_STATE = 42
LABELS_MAP   = {0: "Leve", 1: "Moderado", 2: "Grave"}
COLORS       = ["#4C72B0", "#DD8452", "#55A868"]
os.makedirs(FIG_DIR, exist_ok=True)


# ── 1. Dados e Split Temporal ─────────────────────────────────────────────────
df = load_data("data/raw/parkinsons_telemonitoring.csv")
df = create_subject_id(df)
df = create_multiclass_target(df)
df = remove_outliers_iqr(df, VOCAL_FEATURES)

X_train, X_test, y_train, y_test = temporal_split_per_patient(df, split_ratio=0.8)

print(f"\n[features] {len(ALL_FEATURES)}: subject_id + test_time + 16 biomarcadores vocais")

# Escala
scaler = StandardScaler()
X_train = X_train.copy()
X_test  = X_test.copy()
X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
X_test[COLS_TO_SCALE]  = scaler.transform(X_test[COLS_TO_SCALE])

# SMOTE apenas no treino
sm = SMOTE(random_state=RANDOM_STATE)
X_train_arr, y_train_bal = sm.fit_resample(X_train, y_train)
X_train_bal = pd.DataFrame(X_train_arr, columns=ALL_FEATURES)
X_train_bal["subject_id"] = X_train_bal["subject_id"].round().astype(int).clip(0, 30)

print(f"\n[SMOTE] treino após balanceamento: {len(X_train_bal)} registros")
counts = pd.Series(y_train_bal).value_counts().sort_index()
for k, v in counts.items():
    print(f"  {LABELS_MAP[k]}: {v}")


# ── 2. Treinamento ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("[XGBoost] Treinando com histórico dos pacientes...")
print("="*65)

model_params = dict(
    objective="multi:softmax", num_class=3,
    eval_metric="mlogloss",
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=3, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)

model = XGBClassifier(**model_params)
model.fit(X_train_bal, y_train_bal)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)


# ── 3. Métricas ───────────────────────────────────────────────────────────────
acc     = accuracy_score(y_test, y_pred)
auc_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")

report_str  = classification_report(y_test, y_pred,
                                     target_names=list(LABELS_MAP.values()))
report_dict = classification_report(y_test, y_pred,
                                     target_names=list(LABELS_MAP.values()),
                                     output_dict=True)

print(f"\n[Resultado — Split Temporal por Paciente]")
print(f"  Acurácia:  {acc:.3f}  ({acc*100:.1f}%)")
print(f"  AUC-ROC:   {auc_ovr:.3f}")
print(f"\n[Classification Report]")
print(report_str)

# Salva métricas
metrics = {
    "method"          : "Split Temporal por Paciente (80% histórico / 20% futuro)",
    "n_patients"      : int(df["subject_id"].nunique()),
    "n_features"      : len(ALL_FEATURES),
    "features_used"   : ALL_FEATURES,
    "accuracy"        : round(acc,     4),
    "auc_roc_weighted": round(auc_ovr, 4),
    "classification_report": report_dict,
    "model_params"    : model_params,
}
with open("data/processed/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("[save] data/processed/metrics.json")


# ── 4. Figura — Matriz de Confusão ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=list(LABELS_MAP.values()))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(
    "Matriz de Confusão — XGBoost Multiclasse\n"
    "(Split Temporal por Paciente · 80% treino / 20% teste)",
    fontsize=11, fontweight="bold", pad=12
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig07_confusion_matrix.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig07_confusion_matrix.png")


# ── 5. Figura — Importância das Features ─────────────────────────────────────
importances = model.feature_importances_
feat_imp_df = (
    pd.DataFrame({"feature": ALL_FEATURES, "importance": importances})
    .sort_values("importance", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 8))
colors_bar = ["#C44E52" if f == "subject_id" else
              "#DD8452" if f == "test_time"  else "#4C72B0"
              for f in feat_imp_df["feature"]]
bars = ax.barh(feat_imp_df["feature"], feat_imp_df["importance"],
               color=colors_bar, edgecolor="white", height=0.7)
ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=4)
ax.set_xlabel("Importância (Gain normalizado)", fontsize=11)
ax.set_title(
    "Importância dos Atributos — XGBoost · Progressão do Parkinson\n"
    "(vermelho = subject_id | laranja = test_time | azul = biomarcadores vocais)",
    fontsize=12, fontweight="bold", pad=12
)
ax.tick_params(axis="y", labelsize=10)
ax.set_xlim(0, feat_imp_df["importance"].max() * 1.18)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig08_feature_importance.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig08_feature_importance.png")


# ── 6. Figura — Métricas por Classe ──────────────────────────────────────────
classes     = list(LABELS_MAP.values())
f1_scores   = [report_dict[c]["f1-score"]  for c in classes]
prec_scores = [report_dict[c]["precision"] for c in classes]
rec_scores  = [report_dict[c]["recall"]    for c in classes]

x = np.arange(len(classes)); width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, prec_scores, width, label="Precisão",  color="#4C72B0")
ax.bar(x,         rec_scores,  width, label="Recall",    color="#DD8452")
ax.bar(x + width, f1_scores,   width, label="F1-Score",  color="#55A868")
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=12)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
ax.set_title(
    "Métricas por Classe — XGBoost (Split Temporal)\n"
    "Precisão, Recall e F1-Score",
    fontsize=13, fontweight="bold"
)
ax.legend(fontsize=11)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=9, padding=2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig09_metricas_por_classe.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig09_metricas_por_classe.png")


# ── 7. Figura — Curvas ROC ────────────────────────────────────────────────────
y_bin = label_binarize(y_test, classes=[0, 1, 2])
fig, ax = plt.subplots(figsize=(8, 6))
for i, (label, color) in enumerate(zip(classes, COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2)
ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
ax.set_title(
    "Curvas ROC — One-vs-Rest (OVR)\n"
    "XGBoost · Split Temporal por Paciente",
    fontsize=13, fontweight="bold"
)
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig10_roc_curves.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig10_roc_curves.png")


# ── 8. Figura — Acurácia por Paciente ────────────────────────────────────────
X_test_orig = X_test.copy()
X_test_orig["subject_id_int"] = X_test_orig["subject_id"].copy()
X_test_orig["y_true"] = y_test.values
X_test_orig["y_pred"] = y_pred

patient_acc = (
    X_test_orig.groupby("subject_id_int")
    .apply(lambda g: accuracy_score(g["y_true"], g["y_pred"]))
    .reset_index()
    .rename(columns={0: "accuracy", "subject_id_int": "paciente"})
    .sort_values("accuracy")
)

fig, ax = plt.subplots(figsize=(12, 5))
bar_colors = ["#55A868" if a >= 0.8 else "#DD8452" if a >= 0.6 else "#C44E52"
              for a in patient_acc["accuracy"]]
ax.bar(patient_acc["paciente"].astype(str), patient_acc["accuracy"],
       color=bar_colors, edgecolor="white", width=0.7)
ax.axhline(acc, color="#4C72B0", linestyle="--", linewidth=2,
           label=f"Acurácia global = {acc:.1%}")
ax.set_xlabel("Paciente (subject_id)", fontsize=11)
ax.set_ylabel("Acurácia", fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title(
    "Acurácia por Paciente — Split Temporal\n"
    "(verde ≥ 80% | laranja ≥ 60% | vermelho < 60%)",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig11_acuracia_por_paciente.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig11_acuracia_por_paciente.png")


print(f"\n{'='*65}")
print(f"✓ Treinamento concluído.")
print(f"  Acurácia:  {acc:.3f}  ({acc*100:.1f}%)")
print(f"  AUC-ROC:   {auc_ovr:.3f}")
print(f"  Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
