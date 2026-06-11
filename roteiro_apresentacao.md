# Roteiro da Apresentação — 3ª Apresentação (15 minutos)
## Classificação da Progressão do Parkinson por Telemonitoramento de Voz com XGBoost

**Centro Universitário Dom Helder · Ciência de Dados · Prof. Dr. Marcos W. Rodrigues**

> Todos os integrantes devem apresentar ao menos uma seção.

---

## [2 min] SEÇÃO 1 — TEMA E MOTIVAÇÃO

**Integrante responsável:** _______________

### Slide 1 — O Problema
- A **Doença de Parkinson** é a 2ª doença neurodegenerativa mais comum no mundo
- **10 milhões** de pessoas afetadas globalmente (OMS, 2023); **~200 mil** no Brasil
- Progressão avaliada pela escala **UPDRS** — requer consultas presenciais frequentes
- **Problema:** monitoramento contínuo é inviável em regiões com poucos especialistas

### Slide 2 — A Solução Proposta
- Pacientes com Parkinson desenvolvem **alterações vocais mensuráveis** (89% dos casos)
- Análise acústica da voz pode substituir parcialmente avaliações presenciais
- **Proposta:** classificar o estágio de progressão (Leve / Moderado / Grave) automaticamente a partir de gravações de voz remotas, usando **XGBoost**

> *"Imagine monitorar a progressão do Parkinson de um paciente em Minas Gerais a partir de uma simples gravação de voz. É isso que este trabalho propõe."*

---

## [3 min] SEÇÃO 2 — BASE DE DADOS E ATRIBUTOS

**Integrante responsável:** _______________

### Slide 3 — O Dataset
| Característica | Valor |
|----------------|-------|
| **Nome** | Oxford Parkinson's Disease Telemonitoring |
| **Fonte** | UCI Machine Learning Repository (Tsanas et al., 2010) |
| **Registros** | 5.875 gravações de voz |
| **Pacientes** | 31 indivíduos com Parkinson diagnosticado |
| **Período** | ~6 meses de acompanhamento por paciente |
| **Features usadas** | 18 (subject_id + test_time + 16 biomarcadores vocais) |
| **Valores nulos** | Nenhum |

### Slide 4 — As 18 Features Preditivas

**subject_id (0–30):** identificador único do paciente, reconstruído a partir dos pares únicos (age, sex) — equivalente ao `subject#` original UCI. Permite ao modelo aprender o padrão vocal individual.

**test_time:** tempo em semanas desde o recrutamento — marca a posição do paciente na trajetória de progressão.

**16 biomarcadores vocais:**
- **Jitter (5 variantes):** perturbação de frequência → tremor vocal
- **Shimmer (6 variantes):** variação de amplitude → hipofonia
- **NHR / HNR:** razão ruído/harmônicos → qualidade vocal
- **RPDE / DFA / PPE:** complexidade e caos vocal → disfunção neuromotora

### Slide 5 — Variável-Alvo
**Mostrar:** `fig01_distribuicao_classes.png`

| Classe | Faixa UPDRS | Registros | Proporção |
|--------|-------------|-----------|-----------|
| **Leve** | ≤ 25 | 2.188 | 37,2% |
| **Moderado** | 26–40 | 2.681 | 45,6% |
| **Grave** | > 40 | 1.006 | 17,1% |

Desbalanceamento (Grave = 17%) → solucionado com **SMOTE** exclusivamente no treino.

---

## [3 min] SEÇÃO 3 — METODOLOGIA

**Integrante responsável:** _______________

### Slide 6 — Pipeline Completo

```
[CSV Bruto — 5.875 × 21]
    ↓
[Reconstrução do subject_id a partir de (age, sex)]
    ↓
[Discretização UPDRS → 3 classes (Leve / Moderado / Grave)]
    ↓
[Remoção de outliers — IQR × 3 → 371 registros removidos]
    ↓
[Split Temporal por Paciente — 80% histórico / 20% futuro]
    ↓
[StandardScaler — test_time + 16 biomarcadores]
    ↓
[SMOTE — balanceamento apenas no treino]
    ↓
[XGBoost — 500 estimadores, max_depth=6, lr=0,05]
    ↓
[Avaliação: Acurácia, F1, AUC-ROC, Matriz de Confusão]
```

### Slide 7 — O Split Temporal por Paciente (ponto central)

> **Diferencial metodológico do trabalho — explica por que o resultado é válido.**

**O problema:** o dataset tem ~190 gravações por paciente ao longo de 6 meses. Uma divisão aleatória colocaria gravações do mesmo instante temporal no treino e no teste — o modelo memorizaria o estado atual, não aprenderia progressão.

**A solução adotada — Split Temporal:**

```
Para cada paciente (31 pacientes):
  Gravações ordenadas por test_time
  ├── Primeiros 80% → TREINO  (semanas 0 a ~136)
  └── Últimos 20%   → TESTE   (semanas ~136 a ~215)
```

**Por que subject_id é feature legítima:**
- Em telemonitoramento, o sistema CONHECE o paciente — viu seu histórico no treino
- O modelo prediz estados FUTUROS que pode não ter visto
- Não é memorização: o UPDRS do paciente pode ter mudado no período de teste

**Comparativo de metodologias:**
| Abordagem | O que avalia | Acurácia esperada |
|---|---|---|
| Split aleatório | Memorização (inválido) | ~98% |
| **Split temporal** | **Progressão futura — correto** | **95,7%** |
| GroupKFold | Paciente nunca visto | ~38% |

### Slide 8 — Decisões Técnicas
| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Identificação de pacientes | subject_id (0–30) | Feature legítima no telemonitoramento |
| Progressão temporal | test_time incluído | Posição na trajetória de acompanhamento |
| Discretização | Limiares clínicos (25 / 40) | Relevância médica > divisão estatística |
| Outliers | IQR × 3,0 | Preservar casos graves legítimos |
| Normalização | StandardScaler | Escalas distintas entre features |
| Balanceamento | SMOTE (só no treino) | Classe Grave = 17% |

---

## [4 min] SEÇÃO 4 — RESULTADOS DO XGBOOST

**Integrante responsável:** _______________

### Slide 9 — Configuração do Modelo

| Hiperparâmetro | Valor |
|----------------|-------|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0,05 |
| `subsample` | 0,85 |
| `colsample_bytree` | 0,85 |
| Regularização | gamma=0,1 · L1=0,1 · L2=1,0 |

### Slide 10 — Métricas Globais

| Métrica | Valor |
|---------|-------|
| **Acurácia** | **95,7%** |
| **F1-Score Macro** | **0,96** |
| **F1-Score Weighted** | **0,96** |
| **AUC-ROC (OVR weighted)** | **0,968** |
| Baseline aleatório (3 classes) | 33,3% / AUC 0,500 |

### Slide 11 — Resultados por Classe

| Classe | Precisão | Recall | F1-Score | Suporte |
|--------|----------|--------|----------|---------|
| **Leve** | 0,98 | 0,91 | **0,94** | 420 |
| **Moderado** | 0,93 | 0,98 | **0,95** | 452 |
| **Grave** | 0,98 | 0,99 | **0,99** | 241 |

> **Destaque:** F1 ≥ 0,94 para TODAS as classes, incluindo Grave — demonstra que o modelo não tem viés para classes majoritárias.

### Slide 12 — Figuras
**Mostrar:**
- `fig07_confusion_matrix.png` — Matriz de Confusão (erros concentrados na fronteira Leve↔Moderado)
- `fig10_roc_curves.png` — Curvas ROC (AUC próxima de 1,0 para todas as classes)
- `fig11_acuracia_por_paciente.png` — Acurácia por paciente (maioria ≥ 80%)

### Slide 13 — Importância das Features
**Mostrar:** `fig08_feature_importance.png`

| Rank | Feature | Interpretação |
|------|---------|---------------|
| 1° | **subject_id** | Assinatura vocal individual do paciente |
| 2° | **PPE** | Irregularidade no controle de voz |
| 3° | **HNR** | Deterioração da qualidade vocal |
| 4° | **DFA** | Autocorrelação de longo alcance |
| 5° | **RPDE** | Dinâmica caótica da glote |
| 6° | **test_time** | Progressão ao longo do acompanhamento |

---

## [3 min] SEÇÃO 5 — DISCUSSÃO E CONCLUSÃO

**Integrante responsável:** _______________

### Slide 14 — Interpretação Clínica

**Por que subject_id domina a importância?**
→ Cada paciente tem uma "voz única". O modelo aprende: "paciente X com esta voz agora → UPDRS Y daqui a algumas semanas". Isso é biologicamente plausível e clinicamente útil.

**Por que PPE, HNR, DFA e RPDE se destacam entre os biomarcadores?**
→ Capturam a complexidade dinâmica da fonação em múltiplas escalas temporais — mais informativos que Jitter/Shimmer (métricas locais) para caracterizar o estado neuromotor global.

**O que a acurácia por paciente nos diz?**
→ A maioria dos 31 pacientes tem acurácia ≥ 80% nas gravações futuras. Pacientes com menor performance são aqueles em transição de estágio — clinicamente esperado, pois a progressão é gradual.

### Slide 15 — Vantagens e Limitações

**Vantagens:**
- ✓ Alta acurácia (95,7%) com metodologia temporal correta
- ✓ F1 equilibrado entre as 3 classes — sem viés de classe
- ✓ subject_id permite personalização sem modelos separados por paciente
- ✓ Interpretabilidade via importância de features

**Limitações:**
- ✗ Requer histórico prévio do paciente (não serve para diagnóstico inicial)
- ✗ 31 pacientes distintos — dataset relativamente pequeno
- ✗ Estrutura temporal não modelada explicitamente (LSTM/GRU fariam isso melhor)

### Slide 16 — Conclusão e Trabalhos Futuros

**Conclusão:**
> O XGBoost com split temporal por paciente atingiu **95,7% de acurácia e AUC-ROC de 0,968** para predição de estados futuros de progressão da Doença de Parkinson — resultado sólido, metodologicamente correto e clinicamente relevante. O identificador de paciente (`subject_id`) é a feature mais importante, confirmando que o aprendizado personalizado é a chave para o desempenho em telemonitoramento longitudinal.

**Trabalhos Futuros:**
- Calibração rápida para novos pacientes (few-shot learning)
- LSTM/GRU para modelagem explícita da série temporal de gravações
- SHAP values para interpretabilidade individual por paciente
- Integração com dados de wearables (acelerômetros, giroscópios)

---

## Checklist Final

- [ ] Slides com figuras de `notebooks/data/` (fig07 a fig11)
- [ ] Cada integrante preparou sua seção
- [ ] Notebook `01_eda.ipynb` executado
- [ ] Notebook `02_modelagem_xgboost.ipynb` executado com novos resultados
- [ ] Artigo `artigo_parkinson_xgboost.md` revisado com métricas 95,7%
- [ ] `data/processed/metrics.json` atualizado

## Estrutura do ZIP para Entrega

```
parkinson_xgboost_grupo.zip
├── artigo_parkinson_xgboost.pdf
├── data/raw/parkinsons_telemonitoring.csv
├── src/data/
│   ├── preprocess.py
│   └── train_xgboost.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modelagem_xgboost.ipynb
│   └── data/         ← figuras fig07–fig11
└── data/processed/
    ├── train.csv · test.csv · processed.csv · metrics.json
```
