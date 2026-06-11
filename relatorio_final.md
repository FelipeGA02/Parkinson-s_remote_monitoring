# Relatório Final de Resultados
## Classificação da Progressão da Doença de Parkinson por Telemonitoramento de Voz com XGBoost

**Centro Universitário Dom Helder — Disciplina de Ciência de Dados**
**Orientador:** Prof. Dr. Marcos W. Rodrigues
**Data:** Junho de 2026

---

## 1. Síntese Executiva

O modelo XGBoost treinado com **split temporal por paciente** atingiu **acurácia de 95,7%** e **AUC-ROC de 0,968** na classificação multiclasse (Leve / Moderado / Grave) da progressão da Doença de Parkinson a partir de 18 atributos vocais extraídos do Oxford Parkinson's Disease Telemonitoring Dataset (UCI, 5.875 registros, 31 pacientes).

---

## 2. Dataset

| Característica | Valor |
|---|---|
| Fonte | Oxford Parkinson's Disease Telemonitoring Dataset (UCI) |
| Registros totais | 5.875 |
| Registros após remoção de outliers (IQR × 3,0) | 5.504 |
| Pacientes distintos | 31 |
| Gravações por paciente (média) | ~190 |
| Período de coleta por paciente | ~6 meses |
| Features preditivas | 18 |
| Variável-alvo original | `total_UPDRS` (contínua) |
| Variável-alvo final | 3 classes (Leve / Moderado / Grave) |

**Nota sobre `subject_id`:** A versão pública do CSV não contém a coluna `subject#`. O identificador foi reconstruído deterministicamente a partir dos pares únicos `(age, sex)` — 31 combinações distintas, uma por paciente.

---

## 3. Discretização da Variável-Alvo

A variável `total_UPDRS` foi discretizada com base em limiares clínicos (Goetz *et al.*, 2008):

| Classe | Faixa UPDRS | Registros | Proporção |
|---|---|---|---|
| **Leve** | ≤ 25 | 2.188 | 37,2% |
| **Moderado** | 26 – 40 | 2.681 | 45,6% |
| **Grave** | > 40 | 1.006 | 17,1% |

A coluna `motor_UPDRS` foi excluída por correlação de Pearson de **0,947** com `total_UPDRS` (data leakage direto).

---

## 4. Features Preditivas (18)

| # | Feature | Categoria |
|---|---|---|
| 1 | `subject_id` (0–30) | Identidade do paciente |
| 2 | `test_time` | Temporal (semanas desde recrutamento) |
| 3–7 | `Jitter(%)`, `Jitter(Abs)`, `Jitter:RAP`, `Jitter:PPQ5`, `Jitter:DDP` | Variabilidade de frequência |
| 8–13 | `Shimmer`, `Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `Shimmer:APQ11`, `Shimmer:DDA` | Variabilidade de amplitude |
| 14 | `NHR` | Ruído vocal |
| 15 | `HNR` | Relação harmônicos/ruído |
| 16 | `RPDE` | Complexidade dinâmica (caos glótico) |
| 17 | `DFA` | Autocorrelação de longo alcance |
| 18 | `PPE` | Entropia de período de pitch |

---

## 5. Metodologia: Split Temporal por Paciente

Para cada um dos 31 pacientes, as gravações foram ordenadas cronologicamente por `test_time`:

```
Gravações do paciente (ordenadas por test_time)
├── Primeiros 80%  →  TREINO  (histórico: ~0 a ~136 semanas)
└── Últimos 20%    →  TESTE   (futuro:    ~136 a ~215 semanas)
```

| Conjunto | Registros |
|---|---|
| Treino (pré-SMOTE) | 4.391 |
| Teste | 1.113 |
| Total | 5.504 |

Todos os 31 pacientes estão presentes em ambos os conjuntos.

**Justificativa:** O `subject_id` é feature legítima porque o sistema já conhece o paciente — aprendeu seu histórico vocal no treino. O modelo prediz estados **futuros**, não memoriza estados já vistos. Essa estratégia simula fielmente o fluxo real de uma plataforma de telemonitoramento.

**Pipeline completo:**
1. Reconstrução de `subject_id` a partir dos pares únicos `(age, sex)`
2. Discretização de `total_UPDRS` em 3 classes
3. Remoção de outliers (IQR × 3,0) nos 16 biomarcadores vocais
4. Split temporal por paciente (80% / 20%)
5. `StandardScaler` em `test_time` e 16 biomarcadores (fit apenas no treino)
6. SMOTE aplicado exclusivamente no treino (balancear classe Grave, 17%)

---

## 6. Hiperparâmetros do XGBoost

| Hiperparâmetro | Valor |
|---|---|
| `objective` | `multi:softmax` |
| `num_class` | 3 |
| `eval_metric` | `mlogloss` |
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0,05 |
| `subsample` | 0,85 |
| `colsample_bytree` | 0,85 |
| `min_child_weight` | 3 |
| `gamma` | 0,1 |
| `reg_alpha` | 0,1 |
| `reg_lambda` | 1,0 |
| `random_state` | 42 |

---

## 7. Resultados

### 7.1 Métricas Globais

| Métrica | Valor |
|---|---|
| **Acurácia** | **95,7%** (0,9569) |
| **AUC-ROC (OVR weighted)** | **0,968** |
| **F1-Score Macro** | **0,961** |
| **F1-Score Weighted** | **0,957** |
| Baseline aleatório (3 classes) | ~33,3% / AUC 0,500 |

### 7.2 Resultados por Classe

| Classe | Precisão | Recall | F1-Score | Suporte |
|---|---|---|---|---|
| **Leve** | 0,979 | 0,907 | **0,942** | 420 |
| **Moderado** | 0,925 | 0,985 | **0,954** | 452 |
| **Grave** | 0,984 | 0,992 | **0,988** | 241 |
| **Macro avg** | **0,963** | **0,961** | **0,961** | 1.113 |
| **Weighted avg** | **0,958** | **0,957** | **0,957** | 1.113 |

### 7.3 Observações sobre os Resultados

- **Desempenho homogêneo entre classes:** F1 de 0,94 a 0,99 — sem viés de classe majoritária.
- **Classe Grave com melhor F1 (0,988):** resultado contra-intuitivo explicado pelo SMOTE (balanceou a classe minoritária de 17%) e pelo fato de que os pacientes graves têm padrão vocal mais distinto e consistente.
- **Classe Leve com menor Recall (0,907):** a fronteira Leve/Moderado (UPDRS ≤ 25 vs. 26–40) gera as confusões mais frequentes — clinicamente esperado, dado que a transição entre esses estágios é gradual.
- **Matriz de confusão:** os erros do modelo são quase exclusivamente entre classes adjacentes (Leve ↔ Moderado), sem confusões entre Leve e Grave.

---

## 8. Importância das Features (Gain Normalizado)

| Rank | Feature | Categoria | Interpretação |
|---|---|---|---|
| 1° | **subject_id** | Identidade | Assinatura vocal individual do paciente |
| 2° | **PPE** | Complexidade | Entropia do período de pitch — instabilidade glótica |
| 3° | **HNR** | Ruído vocal | Relação harmônicos/ruído — qualidade fonatória |
| 4° | **DFA** | Complexidade | Autocorrelação de longo alcance — coordenação neuromotora |
| 5° | **RPDE** | Complexidade | Dinâmica caótica da glote |
| 6° | **test_time** | Temporal | Posição na trajetória de acompanhamento |
| 7°+ | Jitter / Shimmer | Locais | Variabilidade ciclo-a-ciclo |

**Destaques:**
- `subject_id` é a feature mais importante: confirma que o aprendizado personalizado por paciente é o principal vetor de desempenho.
- Biomarcadores de complexidade dinâmica (PPE, DFA, RPDE) superam as métricas locais (Jitter, Shimmer) no poder discriminativo — alinhado com a literatura recente sobre caracterização do estado neuromotor global.
- `test_time` no top-6 valida a captura da progressão temporal pelo modelo.

---

## 9. Análise por Paciente

- **> 85% dos pacientes** apresentam acurácia ≥ 80% nas gravações futuras (conjunto de teste).
- Os pacientes com desempenho inferior são aqueles com **maior variabilidade intra-individual do UPDRS** — transições de classe que ocorrem justamente no período de teste.
- Esse comportamento é clinicamente esperado: o modelo erra nos mesmos pontos em que um clínico teria dificuldade de predizer a mudança de estágio.

---

## 10. Análise Exploratória — Destaques

| Achado | Valor |
|---|---|
| Correlação Pearson com `total_UPDRS` — HNR | r = −0,47 (maior absoluto) |
| Correlação Pearson com `total_UPDRS` — PPE | r = +0,41 |
| Correlação Pearson com `total_UPDRS` — DFA | r = −0,38 |
| Multicolinearidade Jitter | r > 0,95 entre variantes |
| Multicolinearidade Shimmer | r > 0,92 entre variantes |
| Kruskal-Wallis (todos atributos × classe) | p < 0,001 para todos |
| Correlação `motor_UPDRS` × `total_UPDRS` | 0,947 → excluído (data leakage) |

A análise de progressão temporal confirmou que a maioria dos 31 pacientes apresenta tendência crescente de UPDRS ao longo do acompanhamento, validando `test_time` como feature relevante.

---

## 11. Comparação com Literatura

| Trabalho | Método | Dataset | Tarefa | Resultado |
|---|---|---|---|---|
| Little *et al.* (2009) | SVM RBF | UCI binário (197 reg.) | Binária | Sensib. 99%, Espec. 91% |
| Grover *et al.* (2018) | Random Forest | UCI binário | Binária | Acurácia 94,8% |
| Dinesh & Bhavanam (2022) | XGBoost | UCI binário (197 reg.) | Binária | Acurácia 97,4% |
| **Este trabalho** | **XGBoost + split temporal** | **UCI telemon. (5.875 reg., 31 pac.)** | **Multiclasse (3 estágios)** | **Acurácia 95,7%, AUC 0,968** |

**Avanços sobre a literatura:**
1. Classificação **multiclasse** (3 estágios de progressão) vs. binária dos trabalhos anteriores.
2. Dataset de **telemonitoramento longitudinal** (5.875 reg.) vs. dataset estático pequeno (197 reg.).
3. **Split temporal por paciente** — metodologia correta para séries temporais; evita vazamento de informação do futuro para o treino.
4. `subject_id` como feature legítima — permite personalização sem modelos separados.

---

## 12. Limitações

1. **Pacientes conhecidos:** o modelo requer histórico prévio de gravações. Para novos pacientes (sem histórico), o desempenho seria inferior.
2. **Tamanho do dataset:** apenas 31 pacientes distintos — pequeno para generalização a populações diversas.
3. **Modelagem temporal implícita:** o XGBoost não modela a sequência temporal explicitamente (ao contrário de LSTM/GRU); `test_time` captura a posição temporal, mas não a dinâmica da série.
4. **Variabilidade de estágio no período de teste:** pacientes em transição de classe têm predição mais difícil — inerente ao fenômeno clínico.

---

## 13. Trabalhos Futuros

1. **Few-shot para novos pacientes:** calibrar o modelo com as primeiras gravações de um paciente novo.
2. **Modelos de séries temporais:** LSTM ou GRU para modelar explicitamente a trajetória temporal de progressão.
3. **SHAP Values:** interpretabilidade por gravação e por paciente — útil para aplicação clínica.
4. **Dataset maior:** replicar em datasets com mais pacientes (ex.: mPower, iDoPa).
5. **Fusão de modalidades:** combinar voz com dados de wearables (acelerômetros, giroscópios).
6. **Validação cross-dataset:** treinar em Oxford, testar em outro dataset de voz para Parkinson.

---

## 14. Figuras Geradas

| Arquivo | Conteúdo |
|---|---|
| `fig01_distribuicao_classes.png` | Distribuição de frequência das 3 classes |
| `fig02_heatmap_correlacao.png` | Heatmap de correlação entre features |
| `fig03_boxplots_por_classe.png` | Boxplots dos biomarcadores por classe |
| `fig04_correlacao_target.png` | Correlação de Pearson de cada feature com `total_UPDRS` |
| `fig05_updrs_discretizacao.png` | Histograma de `total_UPDRS` com os limiares de discretização |
| `fig07_confusion_matrix.png` | Matriz de confusão — XGBoost multiclasse |
| `fig08_feature_importance.png` | Importância das features (Gain normalizado) |
| `fig09_metricas_por_classe.png` | Precisão, Recall e F1 por classe |
| `fig10_roc_curves.png` | Curvas ROC (OVR) por classe |
| `fig11_acuracia_por_paciente.png` | Acurácia individual dos 31 pacientes |
| `fig12_comparativo_metodologias.png` | Comparativo de metodologias de split |

Todas as figuras estão em `notebooks/data/` (PNG, 150 dpi).

---

## 15. Arquivos do Projeto

| Arquivo | Descrição |
|---|---|
| `data/raw/parkinsons_telemonitoring.csv` | Dataset original (5.875 × 21) |
| `data/processed/metrics.json` | Métricas salvas em JSON |
| `src/data/preprocess.py` | Pipeline: subject_id, split temporal, SMOTE |
| `src/data/train_xgboost.py` | Treinamento + geração das figuras |
| `notebooks/01_eda.ipynb` | EDA com análise temporal por paciente |
| `notebooks/02_modelagem_xgboost.ipynb` | Modelagem completa com split temporal |
| `artigo_parkinson_xgboost.md` | Artigo completo (Introdução → Conclusão) |
| `roteiro_apresentacao.md` | Roteiro de apresentação (15 min) |
| `relatorio_final.md` | Este documento |

---

## 16. Como Reproduzir os Resultados

```bash
# Na raiz do projeto
source venv/bin/activate

# 1. Pré-processamento
python3 src/data/preprocess.py

# 2. Treinamento + geração de figuras
PYTHONPATH=src/data python3 src/data/train_xgboost.py

# Saídas:
#   data/processed/metrics.json   → métricas
#   notebooks/data/fig07–fig11    → figuras PNG
```

---

## 17. Referências Chave

- TSANAS, A. *et al.* Accurate telemonitoring of Parkinson's disease progression by noninvasive speech tests. **IEEE Trans. Biomed. Eng.**, v. 57, n. 4, p. 884-893, 2010.
- LITTLE, M. A. *et al.* Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. **IEEE Trans. Biomed. Eng.**, v. 56, n. 4, p. 1015-1022, 2009.
- GOETZ, C. G. *et al.* MDS-UPDRS. **Movement Disorders**, v. 23, n. 15, p. 2129-2170, 2008.
- DINESH, K. N.; BHAVANAM, G. R. Parkinson's Disease Prediction using XGBoost. **IJARET**, v. 13, n. 3, p. 211-220, 2022.
- UCI MACHINE LEARNING REPOSITORY. **Parkinsons Telemonitoring Data Set**. 2009.
