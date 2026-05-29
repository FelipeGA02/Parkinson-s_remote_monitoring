"""
Treinamento e avaliação — XGBoost Multiclasse (avaliação correta por GroupKFold)
Doença de Parkinson · Telemonitoramento de Voz

Metodologia de validação: GroupKFold(n_splits=5) por paciente.
Cada fold testa o modelo em pacientes NUNCA vistos no treino,
refletindo performance real de generalização clínica.
"""

import os, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score,
                              accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from preprocess import load_data, create_multiclass_target, remove_outliers_iqr, VOCAL_FEATURES

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

FIG_DIR      = "notebooks/data"
RANDOM_STATE = 42
LABELS_MAP   = {0: "Leve", 1: "Moderado", 2: "Grave"}
COLORS       = ["#4C72B0", "#DD8452", "#55A868"]
os.makedirs(FIG_DIR, exist_ok=True)


# ── 1. Dados ──────────────────────────────────────────────────────────────────
df = load_data("data/raw/parkinsons_telemonitoring.csv")
df = create_multiclass_target(df)
df = remove_outliers_iqr(df, VOCAL_FEATURES)

X      = df[VOCAL_FEATURES]
y      = df["progressao"]
groups = df["patient_id"]

print(f"\n[features] {len(VOCAL_FEATURES)} atributos vocais")
print(f"[pacientes] {groups.nunique()} — split por paciente (sem leakage)")


# ── 2. GroupKFold — avaliação honesta ─────────────────────────────────────────
print("\n[GroupKFold] 5 folds por paciente...")
gkf = GroupKFold(n_splits=5)

model_cv = XGBClassifier(
    objective="multi:softmax", num_class=3,
    eval_metric="mlogloss",
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)

fold_results = []
y_test_all, y_pred_all, y_proba_all = [], [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_sc, y_tr)

    model_cv.fit(X_tr_bal, y_tr_bal)
    yp    = model_cv.predict(X_te_sc)
    yprob = model_cv.predict_proba(X_te_sc)

    acc = accuracy_score(y_te, yp)
    auc_s = roc_auc_score(y_te, yprob, multi_class="ovr", average="weighted")
    n_pts = df.iloc[test_idx]["patient_id"].nunique()

    print(f"  Fold {fold+1}: acc={acc:.3f} | AUC={auc_s:.3f} | {n_pts} pacientes no teste")
    fold_results.append({"fold": fold+1, "accuracy": acc, "auc_roc": auc_s})

    y_test_all.extend(y_te.tolist())
    y_pred_all.extend(yp.tolist())
    y_proba_all.extend(yprob.tolist())

mean_acc  = np.mean([r["accuracy"] for r in fold_results])
std_acc   = np.std( [r["accuracy"] for r in fold_results])
mean_auc  = np.mean([r["auc_roc"]  for r in fold_results])
std_auc   = np.std( [r["auc_roc"]  for r in fold_results])

y_test_all  = np.array(y_test_all)
y_pred_all  = np.array(y_pred_all)
y_proba_all = np.array(y_proba_all)

report_str  = classification_report(y_test_all, y_pred_all,
                                     target_names=list(LABELS_MAP.values()))
report_dict = classification_report(y_test_all, y_pred_all,
                                     target_names=list(LABELS_MAP.values()),
                                     output_dict=True)

print(f"\n[Resultado Consolidado (5 folds)]")
print(f"  Acurácia:  {mean_acc:.3f} ± {std_acc:.3f}")
print(f"  AUC-ROC:   {mean_auc:.3f} ± {std_auc:.3f}")
print(f"\n[Classification Report — predições consolidadas]")
print(report_str)

# Salva métricas
metrics = {
    "note"             : "Avaliação GroupKFold(5) por paciente — sem data leakage",
    "n_patients"       : int(groups.nunique()),
    "n_vocal_features" : len(VOCAL_FEATURES),
    "accuracy_mean"    : round(mean_acc, 4),
    "accuracy_std"     : round(std_acc,  4),
    "auc_roc_mean"     : round(mean_auc, 4),
    "auc_roc_std"      : round(std_auc,  4),
    "fold_results"     : fold_results,
    "classification_report": report_dict,
}
with open("data/processed/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("[save] data/processed/metrics.json")


# ── 3. Treino final (para figuras de feature importance) ──────────────────────
# Treino em 80% dos pacientes para gerar figuras representativas
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_idx, te_idx = next(gss.split(X, y, groups))

X_tr_f = X.iloc[tr_idx].copy()
X_te_f = X.iloc[te_idx].copy()
y_tr_f = y.iloc[tr_idx]
y_te_f = y.iloc[te_idx]

scaler_f = StandardScaler()
X_tr_sc_f = scaler_f.fit_transform(X_tr_f)
X_te_sc_f = scaler_f.transform(X_te_f)

sm_f = SMOTE(random_state=RANDOM_STATE)
X_tr_bal_f, y_tr_bal_f = sm_f.fit_resample(X_tr_sc_f, y_tr_f)

best_model = XGBClassifier(
    objective="multi:softmax", num_class=3,
    eval_metric="mlogloss",
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)
best_model.fit(X_tr_bal_f, y_tr_bal_f)


# ── 4. Matriz de Confusão (consolidada de todos os folds) ─────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cm   = confusion_matrix(y_test_all, y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=list(LABELS_MAP.values()))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(
    "Matriz de Confusão — XGBoost Multiclasse\n"
    "(Consolidada · GroupKFold 5-fold · Avaliação por paciente)",
    fontsize=12, fontweight="bold", pad=12
)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig07_confusion_matrix.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig07_confusion_matrix.png")


# ── 5. Importância das Features ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
plot_importance(best_model, ax=ax, max_num_features=16,
                importance_type="gain", height=0.7,
                color="#4C72B0", edgecolor="white")
ax.set_title(
    "Importância dos Atributos Vocais (Gain)\n"
    "XGBoost · Progressão do Parkinson",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlabel("Ganho Médio (Gain)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig08_feature_importance.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig08_feature_importance.png")


# ── 6. Métricas por Classe (consolidadas) ─────────────────────────────────────
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
    "Métricas por Classe — XGBoost (GroupKFold 5-fold)\nPrecisão, Recall e F1-Score",
    fontsize=13, fontweight="bold"
)
ax.legend(fontsize=11)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=9, padding=2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig09_metricas_por_classe.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig09_metricas_por_classe.png")


# ── 7. Curvas ROC (consolidadas) ──────────────────────────────────────────────
y_bin   = label_binarize(y_test_all, classes=[0, 1, 2])
fig, ax = plt.subplots(figsize=(8, 6))
for i, (label, color) in enumerate(zip(classes, COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_all[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc_val:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1.2)
ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)
ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)
ax.set_title(
    "Curvas ROC — One-vs-Rest (OVR)\nXGBoost · GroupKFold 5-fold",
    fontsize=13, fontweight="bold"
)
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig10_roc_curves.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig10_roc_curves.png")


# ── 8. Desempenho por Fold (barplot) ──────────────────────────────────────────
folds_df = pd.DataFrame(fold_results)
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(5); w = 0.35
ax.bar(x - w/2, folds_df["accuracy"], w, label="Acurácia",  color="#4C72B0")
ax.bar(x + w/2, folds_df["auc_roc"],  w, label="AUC-ROC",   color="#DD8452")
ax.axhline(mean_acc,  color="#4C72B0", linestyle="--", linewidth=1.5,
           label=f"Média acc = {mean_acc:.3f}")
ax.axhline(mean_auc, color="#DD8452", linestyle="--", linewidth=1.5,
           label=f"Média AUC = {mean_auc:.3f}")
ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
ax.set_ylim(0, 0.85); ax.set_ylabel("Score")
ax.set_title("Desempenho por Fold — GroupKFold 5-fold\n(avaliação independente por paciente)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig11_desempenho_por_fold.png", dpi=150)
plt.close()
print(f"[save] {FIG_DIR}/fig11_desempenho_por_fold.png")

print(f"\n✓ Concluído.")
print(f"  Acurácia: {mean_acc:.3f} ± {std_acc:.3f}")
print(f"  AUC-ROC:  {mean_auc:.3f} ± {std_auc:.3f}")
print(f"\n  Nota: resultados mais baixos que split aleatório são ESPERADOS e CORRETOS.")
print(f"  O split aleatório causava leakage de paciente → resultados irrealistas de 98%.")
