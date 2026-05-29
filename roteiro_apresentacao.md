# Roteiro da Apresentação — 3ª Apresentação (15 minutos)
## Classificação da Progressão do Parkinson por Telemonitoramento de Voz com XGBoost

**Centro Universitário Dom Helder · Ciência de Dados · Prof. Dr. Marcos W. Rodrigues**

> Todos os integrantes devem apresentar ao menos uma seção.  
> Sugestão: distribuir as 5 seções entre os membros do grupo.

---

## [2 min] SEÇÃO 1 — TEMA E MOTIVAÇÃO

**Integrante responsável:** _______________

### Slide 1 — O Problema
- A **Doença de Parkinson** é a 2ª doença neurodegenerativa mais comum no mundo
- **10 milhões** de pessoas afetadas globalmente (OMS, 2023)
- **~200 mil** pacientes diagnosticados no Brasil
- Progressão avaliada pela escala **UPDRS** — requer consultas presenciais frequentes
- **Problema:** monitoramento contínuo é inviável em regiões com poucos especialistas

### Slide 2 — A Solução Proposta
- Pacientes com Parkinson desenvolvem **alterações vocais mensuráveis** (em até 89% dos casos)
- Análise acústica da voz pode substituir parcialmente avaliações presenciais
- **Proposta:** classificar o estágio de progressão (Leve / Moderado / Grave) automaticamente via machine learning a partir de gravações de voz remotas
- **Algoritmo:** XGBoost — estado da arte em dados tabulares estruturados

> **Fala sugerida:** *"Imagine poder monitorar a progressão do Parkinson de um paciente em Minas Gerais a partir de uma simples ligação. É isso que este trabalho propõe."*

---

## [3 min] SEÇÃO 2 — BASE DE DADOS E ATRIBUTOS

**Integrante responsável:** _______________

### Slide 3 — O Dataset
| Característica | Valor |
|----------------|-------|
| **Nome** | Oxford Parkinson's Disease Telemonitoring |
| **Fonte** | UCI Machine Learning Repository (Tsanas et al., 2010) |
| **Registros** | 5.875 gravações de voz |
| **Pacientes** | 42 indivíduos com Parkinson diagnosticado |
| **Atributos preditivos** | 19 medidas biomédicas de voz |
| **Período** | ~6 meses de telemonitoramento por paciente |
| **Valores nulos** | Nenhum |

### Slide 4 — Os Atributos de Voz
- **Jitter (5 variantes):** perturbação ciclo-a-ciclo na frequência fundamental → tremor vocal
- **Shimmer (6 variantes):** variação de amplitude → hipofonia
- **NHR / HNR:** razão ruído/harmônicos → qualidade vocal
- **RPDE / DFA / PPE:** medidas de complexidade e caos → disfunção neuromotora
- **Idade, sexo, test_time:** dados demográficos e temporais

### Slide 5 — Variável-Alvo: Discretização do UPDRS
**Mostrar gráficos:** `fig01_distribuicao_classes.png` + `fig05_updrs_discretizacao.png`

| Classe | Faixa UPDRS | Registros | Proporção |
|--------|-------------|-----------|-----------|
| **Leve** | ≤ 25 | 2.188 | 37,2% |
| **Moderado** | 26–40 | 2.681 | 45,6% |
| **Grave** | > 40 | 1.006 | 17,1% |

*Limiares baseados em: Goetz et al. (2008) — MDS-UPDRS*  
→ Desbalanceamento detectado (Grave = 17%) → solução: **SMOTE**

---

## [3 min] SEÇÃO 3 — METODOLOGIA E PRÉ-PROCESSAMENTO

**Integrante responsável:** _______________

### Slide 6 — Fluxograma do Pipeline

```
[CSV Bruto]
    ↓
[Discretização UPDRS → 3 classes]
    ↓
[Remoção de outliers — IQR × 3 → −371 registros]
    ↓
[StandardScaler — normalização]
    ↓
[Divisão treino/teste estratificada 80/20]
    ↓
[SMOTE — balanceamento do treino]
    ↓
[XGBoost + GridSearchCV (5-fold)]
    ↓
[Avaliação: Acurácia, F1, AUC-ROC, Matriz de Confusão]
```

### Slide 7 — Decisões Técnicas
| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Discretização | Limiares clínicos (25 / 40) | Significado clínico > divisão estatística |
| Outliers | IQR × 3,0 | Preservar casos graves legítimos |
| Normalização | StandardScaler | Atributos em escalas muito diferentes |
| Features | 16 vocais (sem age/sex/test_time) | age+sex identificam o paciente → leakage |
| Balanceamento | SMOTE (apenas no treino) | Classe Grave = 17% — risco de viés |
| **Validação** | **GroupKFold(5) por paciente** | **Evita data leakage — único método correto** |

### Slide 7b — O Problema do Data Leakage ⚠️ (slide-chave)

> **Descoberta metodológica importante do trabalho:**

| Método de split | Acurácia | AUC-ROC | É válido? |
|----------------|----------|---------|-----------|
| Split aleatório de registros (errado) | **98,00%** | **0,999** | ❌ Não — leakage |
| GroupKFold por paciente (correto) | **38,5%** | **0,535** | ✓ Sim |

- O dataset tem 31 pacientes com ~178 gravações cada
- Split aleatório coloca gravações do mesmo paciente em treino E teste
- O modelo memoriza o padrão UPDRS de cada indivíduo, não generaliza
- **Este é o erro mais comum na literatura com este dataset**

---

## [4 min] SEÇÃO 4 — RESULTADOS DO XGBOOST

**Integrante responsável:** _______________

### Slide 8 — Configuração do XGBoost

| Hiperparâmetro | Valor |
|----------------|-------|
| `n_estimators` | 200 |
| `max_depth` | 5 |
| `learning_rate` | 0,10 |
| `subsample` | 0,8 |
| `colsample_bytree` | 0,8 |
| **Validação** | **GroupKFold(5) por paciente** |

### Slide 9 — Métricas de Avaliação (DESTACAR)

| Métrica | Valor |
|---------|-------|
| **Acurácia** | **38,5% ± 2,7%** |
| **F1-Score Macro** | **0,34** |
| **F1-Score Weighted** | **0,39** |
| **AUC-ROC (OVR)** | **0,535 ± 0,024** |
| Baseline aleatório (referência) | ~33,3% / AUC 0,500 |

**Tabela completa por classe (predições consolidadas dos 5 folds):**
| Classe | Precisão | Recall | F1 | Suporte |
|--------|----------|--------|----|---------|
| Leve | 0,45 | 0,46 | 0,45 | 2.140 |
| Moderado | 0,46 | 0,41 | 0,43 | 2.412 |
| Grave | 0,13 | 0,16 | 0,14 | 952 |

### Slide 10 — Matriz de Confusão
**Mostrar imagem:** `notebooks/data/fig07_confusion_matrix.png`

- Resultado consolidado dos 5 folds (31 pacientes)
- Classe Grave é a mais difícil — menor F1 (0,14)
- Erros predominantes: confusão entre Leve e Moderado
- Resultado honesto: o modelo erra bastante em pacientes não vistos

### Slide 11 — Desempenho por Fold
**Mostrar imagem:** `notebooks/data/fig11_desempenho_por_fold.png`

- Acurácia varia de 34,6% a 42,3% entre folds
- Variância ± 2,7% — relativamente estável
- Confirma que o modelo generaliza de forma consistente (mas modesta)

### Slide 12 — Importância das Features
**Mostrar imagem:** `notebooks/data/fig08_feature_importance.png`

Top 5 atributos mais importantes (mesmo com baixa acurácia, o modelo prioriza os features corretos clinicamente):
1. **HNR** — relação harmônicos/ruído
2. **PPE** — entropia do período de pitch
3. **DFA** — flutuação sem tendência
4. **RPDE** — densidade de recorrência
5. **Shimmer** — variação de amplitude

---

## [3 min] SEÇÃO 5 — DISCUSSÃO E CONCLUSÃO

**Integrante responsável:** _______________

### Slide 13 — Interpretação Clínica dos Resultados

**Por que HNR é o mais importante?**
→ Pacientes com Parkinson avançado têm perda progressiva do controle motor laríngeo
→ Isso aumenta o componente de ruído e reduz HNR — sinal claro de deterioração

**Por que PPE e RPDE são relevantes?**
→ Capturam irregularidades em escalas temporais longas no sinal de voz
→ Mais robustos a variações de gravação que Jitter/Shimmer

**O modelo erra mais onde?**
→ Na fronteira Leve ↔ Moderado — clinicamente compreensível dado que a progressão é contínua

### Slide 14 — Vantagens e Limitações

**Vantagens do XGBoost:**
- ✓ Robusto a outliers — adequado para dados biomédicos
- ✓ Importância de features clinicamente interpretável
- ✓ Identifica corretamente HNR, PPE, DFA como features-chave
- ✓ Eficiente — treinamento completo em poucos minutos

**Limitações (honestas):**
- ✗ Generalização cross-paciente é fraca — acurácia de 38,5%
- ✗ Apenas 31 pacientes distintos — dataset pequeno para este problema
- ✗ Não explora a estrutura temporal (longitudinal)
- ✗ Classe Grave sub-representada mesmo após SMOTE

### Slide 15 — Conclusão e Trabalhos Futuros

**Conclusão:**
> Com avaliação metodologicamente correta (GroupKFold por paciente), o XGBoost atingiu **38,5% de acurácia** — modesto, porém honesto. A principal contribuição é demonstrar que **split aleatório neste dataset inflaciona artificialmente a acurácia para 98%**, um erro metodológico com impacto direto na validade científica do trabalho.

**Trabalhos Futuros:**
- Modelos **personalizados por paciente** (within-patient) — abordagem mais promissora
- LSTM/GRU para modelagem da trajetória temporal de progressão
- SHAP values para explicabilidade individual
- Dataset maior (mais pacientes distintos)
- Integração com dados de wearables

---

## Checklist Final para a Apresentação

- [ ] Slides prontos com todas as figuras geradas (notebooks/data/)
- [ ] Cada integrante preparou sua seção
- [ ] Notebook `01_eda.ipynb` executado completamente
- [ ] Notebook `02_modelagem_xgboost.ipynb` executado completamente
- [ ] Figuras disponíveis: fig01, fig02, fig03, fig05, fig07, fig08, fig09, fig10
- [ ] Artigo `artigo_parkinson_xgboost.md` revisado e formatado
- [ ] ZIP com entregáveis preparado até 01/06

## Estrutura do ZIP para Entrega

```
parkinson_xgboost_grupo.zip
├── artigo_parkinson_xgboost.pdf   ← exportar o .md para PDF
├── data/
│   └── raw/parkinsons_telemonitoring.csv
├── src/
│   └── data/
│       ├── preprocess.py
│       └── train_xgboost.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modelagem_xgboost.ipynb
│   └── data/                      ← todas as figuras .png
└── data/processed/
    ├── processed.csv
    ├── train.csv
    ├── test.csv
    └── metrics.json
```
