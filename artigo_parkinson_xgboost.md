# Classificação da Progressão da Doença de Parkinson por Telemonitoramento de Voz Utilizando XGBoost

**Centro Universitário Dom Helder — Disciplina de Ciência de Dados**  
**Orientador:** Prof. Dr. Marcos W. Rodrigues

---

## RESUMO

A Doença de Parkinson (DP) é a segunda doença neurodegenerativa mais prevalente no mundo, afetando aproximadamente 10 milhões de pessoas, sendo cerca de 200 mil no Brasil. O monitoramento contínuo da progressão da doença é um desafio clínico, especialmente em regiões com acesso limitado a especialistas. Este trabalho propõe a classificação do estágio de progressão da DP em três níveis — leve, moderado e grave — a partir de 18 atributos preditivos (identificador do paciente, tempo de monitoramento e 16 biomarcadores vocais), utilizando o algoritmo XGBoost. A base de dados utilizada é o Oxford Parkinson's Disease Telemonitoring Dataset (UCI), composta por 5.875 registros de 31 pacientes distintos. A contribuição metodológica central é o emprego de **split temporal por paciente**: para cada paciente, os primeiros 80% das gravações (ordenadas cronologicamente por `test_time`) compõem o conjunto de treino, e os últimos 20% o conjunto de teste. Essa estratégia simula fielmente o cenário clínico de telemonitoramento, no qual o modelo aprende com o histórico vocal do paciente e prediz seu estado em gravações futuras — sem memorizar o estado corrente. O identificador de paciente (`subject_id`) é incluído como feature legítima, pois o sistema conhece previamente os indivíduos monitorados. O XGBoost atingiu **acurácia de 95,7%** e **AUC-ROC de 0,968**, com F1-Score macro de 0,96 — desempenho homogêneo entre as três classes, incluindo a classe Grave (F1 = 0,99).

**Palavras-chave:** Doença de Parkinson. Telemonitoramento. Voz. XGBoost. Classificação Multiclasse. Split Temporal. UPDRS. subject_id.

---

## 1. INTRODUÇÃO

A Doença de Parkinson (DP) é uma enfermidade neurodegenerativa progressiva causada pela perda de neurônios dopaminérgicos na substância negra do cérebro. Segundo a Organização Mundial da Saúde (OMS, 2023), mais de 10 milhões de pessoas são afetadas mundialmente, com prevalência crescente em função do envelhecimento populacional. No Brasil, estima-se a existência de aproximadamente 200 mil pacientes diagnosticados (ABEN, 2022). A doença manifesta-se clinicamente por tremores de repouso, rigidez muscular, bradicinesia e instabilidade postural, além de sintomas não motores como alterações de voz e fala, que surgem em estágios precoces (Braak *et al.*, 2003).

O monitoramento clínico convencional baseia-se na aplicação presencial da Unified Parkinson's Disease Rating Scale (UPDRS). Entretanto, consultas frequentes nem sempre são viáveis, motivando o desenvolvimento de tecnologias de telemonitoramento que permitam a avaliação remota e contínua da evolução da doença. Estudos recentes demonstram que as alterações vocais — detectáveis por medidas acústicas como jitter, shimmer, NHR, HNR, RPDE, DFA e PPE — constituem biomarcadores confiáveis para a avaliação do estado neurológico de pacientes com DP (Little *et al.*, 2009; Tsanas *et al.*, 2010).

O presente trabalho aborda a classificação da progressão da DP como um problema multiclasse de três estágios (leve, moderado e grave). O XGBoost foi adotado como algoritmo principal por sua robustez a outliers, capacidade de modelar relações não lineares e por fornecer importância de features para interpretação clínica. A contribuição metodológica central é o **split temporal por paciente**, que respeita a natureza longitudinal do dataset e simula o fluxo real de uma plataforma de telemonitoramento.

---

## 2. TRABALHOS RELACIONADOS

### 2.1 Contextualização Clínica da Doença de Parkinson

A DP é uma doença de curso progressivo e irreversível. A escala UPDRS (MDS-UPDRS) quantifica a gravidade dos sintomas motores e não motores, sendo amplamente utilizada em ensaios clínicos e no monitoramento longitudinal (Goetz *et al.*, 2008). As alterações vocais estão presentes em até 89% dos pacientes (Ramig *et al.*, 2004), tornando a voz um biomarcador não invasivo e de baixo custo para o telemonitoramento.

### 2.2 Aplicação de Técnicas de Machine Learning ao Diagnóstico por Voz

**Little *et al.* (2009)** utilizaram SVM com kernel RBF e 22 atributos de voz, obtendo sensibilidade de 99% e especificidade de 91%. O estudo estabeleceu jitter, shimmer, NHR e HNR como marcadores diagnósticos fundamentais.

**Tsanas *et al.* (2010)** coletaram o Oxford Parkinson's Disease Telemonitoring Dataset e desenvolveram sistema de predição remota de pontuações UPDRS. Os autores demonstraram que modelos treinados com o histórico individual de cada paciente (within-patient) superam substancialmente modelos genéricos (cross-patient) — resultado replicado e formalizado neste trabalho pela metodologia de split temporal.

**Grover *et al.* (2018)** compararam múltiplos algoritmos para classificação binária de Parkinson. O Random Forest apresentou melhor desempenho com acurácia de 94,8%, sem abordar a multiclasse nem o acompanhamento longitudinal.

**Dinesh e Bhavanam (2022)** aplicaram XGBoost ao dataset UCI Parkinsons (binário, 197 registros), reportando acurácia de 97,4%. O presente trabalho avança sobre esse resultado ao abordar a classificação multiclasse em três estágios de progressão e ao empregar o dataset de telemonitoramento longitudinal (5.875 registros, 31 pacientes).

---

## 3. METODOLOGIA

### 3.1 Materiais

A base de dados é o **Oxford Parkinson's Disease Telemonitoring Dataset** (Tsanas *et al.*, 2010), disponível no UCI Machine Learning Repository. O conjunto contém **5.875 registros** coletados de **31 pacientes** com diagnóstico confirmado de Parkinson, ao longo de aproximadamente seis meses de telemonitoramento. Cada registro representa uma medição de voz individual, descrito por 21 colunas: age, test_time, 16 biomarcadores vocais, sex, motor_UPDRS e total_UPDRS.

**Nota sobre subject_id:** A versão pública do CSV não contém a coluna `subject#` original. Entretanto, cada par (age, sex) é único no dataset — 31 combinações distintas correspondendo exatamente aos 31 pacientes — permitindo a reconstrução determinística do identificador de paciente.

### 3.2 Atributos Preditivos

18 features preditivas foram utilizadas:

| Feature | Tipo | Justificativa |
|---------|------|---------------|
| `subject_id` (0–30) | Identidade | Captura o padrão vocal individual; legítimo no telemonitoramento |
| `test_time` | Temporal | Posição na trajetória de acompanhamento |
| 16 biomarcadores vocais | Biomédico | Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE |

A variável `motor_UPDRS` foi excluída por correlação de Pearson de 0,947 com `total_UPDRS`, o que constituiria data leakage direto.

### 3.3 Discretização da Variável-Alvo

A variável `total_UPDRS` foi discretizada em três classes com base em limiares clínicos (Goetz *et al.*, 2008):

| Classe | Faixa UPDRS | Registros | Proporção |
|--------|-------------|-----------|-----------|
| Leve | ≤ 25 | 2.188 | 37,2% |
| Moderado | 26–40 | 2.681 | 45,6% |
| Grave | > 40 | 1.006 | 17,1% |

### 3.4 Pipeline de Pré-processamento

1. Reconstrução do `subject_id` a partir dos pares únicos (age, sex)
2. Discretização do `total_UPDRS` em 3 classes
3. Remoção de 371 registros com valores extremos (IQR × 3,0)
4. **Split temporal por paciente** (80% histórico → treino / 20% futuro → teste)
5. `StandardScaler` em `test_time` e 16 biomarcadores (parâmetros estimados no treino)
6. SMOTE aplicado exclusivamente no treino para balancear a classe Grave (17%)

### 3.5 Split Temporal por Paciente

Esta é a decisão metodológica central do trabalho. Para cada um dos 31 pacientes:

```
Gravações ordenadas por test_time (cronologicamente)
├── Primeiros 80% → TREINO  (histórico: ~0 a ~136 semanas)
└── Últimos 20%   → TESTE   (futuro:    ~136 a ~215 semanas)
```

**Resultado:** 4.391 registros no treino / 1.113 no teste. Todos os 31 pacientes estão presentes em ambos os conjuntos.

**Justificativa:** O `subject_id` é feature legítima porque o sistema **já conhece** o paciente — ele viu as gravações históricas no treino. O modelo aprende o padrão vocal individual e a trajetória de progressão; ao predizer o teste, enfrenta gravações **futuras** com UPDRS que pode ter mudado, exigindo generalização temporal genuína — não memorização.

### 3.6 Configuração do XGBoost

| Hiperparâmetro | Valor |
|----------------|-------|
| `objective` | `multi:softmax` |
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0,05 |
| `subsample` | 0,85 |
| `colsample_bytree` | 0,85 |
| `min_child_weight` | 3 |
| `gamma` | 0,1 |
| `reg_alpha` | 0,1 |
| `reg_lambda` | 1,0 |

---

## 4. EXPERIMENTOS E ANÁLISE DE RESULTADOS

### 4.1 Dimensões da Base de Dados

| Característica | Valor |
|----------------|-------|
| Registros totais | 5.875 |
| Registros após remoção de outliers | 5.504 |
| Pacientes distintos (subject_id) | 31 |
| Gravações por paciente (média) | ~190 |
| Features preditivas | 18 |
| Período de coleta | ~6 meses por paciente |

### 4.2 Análise Exploratória

A matriz de correlação revelou alta multicolinearidade entre as variantes de Jitter (r > 0,95) e Shimmer (r > 0,92). As correlações mais expressivas com a variável-alvo foram observadas para HNR (r = −0,47), PPE (r = +0,41) e DFA (r = −0,38). O teste de Kruskal-Wallis indicou diferença estatisticamente significativa entre as classes para todos os atributos (p < 0,001).

A análise da progressão temporal (test_time × total_UPDRS por paciente) confirmou a natureza longitudinal do dataset: a maioria dos pacientes apresenta tendência crescente de UPDRS ao longo do período de acompanhamento, validando a relevância do `test_time` como feature preditiva.

### 4.3 Resultados do XGBoost

**Tabela 1 — Resultados por Classe (Split Temporal por Paciente)**

| Classe | Precisão | Recall | F1-Score | Suporte |
|--------|----------|--------|----------|---------|
| Leve (0) | 0,98 | 0,91 | **0,94** | 420 |
| Moderado (1) | 0,93 | 0,98 | **0,95** | 452 |
| Grave (2) | 0,98 | 0,99 | **0,99** | 241 |
| **Macro avg** | **0,96** | **0,96** | **0,96** | 1.113 |
| **Weighted avg** | **0,96** | **0,96** | **0,96** | 1.113 |

**Tabela 2 — Métricas Globais**

| Métrica | Valor |
|---------|-------|
| Acurácia | **95,7%** |
| F1-Score Macro | **0,96** |
| F1-Score Weighted | **0,96** |
| AUC-ROC (OVR weighted) | **0,968** |
| Baseline aleatório (3 classes) | ~33,3% / AUC 0,500 |

Os resultados expressivos refletem a capacidade do XGBoost de aprender a trajetória vocal individual de cada paciente (via `subject_id`) e a progressão temporal (via `test_time`). O desempenho homogêneo entre as três classes — incluindo a classe Grave (F1 = 0,99), historicamente a mais difícil — demonstra que o SMOTE foi eficaz e que o modelo generaliza bem para estados futuros de progressão severa.

### 4.4 Importância das Features

| Rank | Atributo | Categoria | Interpretação |
|------|----------|-----------|---------------|
| 1° | **subject_id** | Identidade | Padrão vocal individual do paciente |
| 2° | **PPE** | Complexidade | Entropia do período de pitch |
| 3° | **HNR** | Ruído vocal | Relação harmônicos/ruído |
| 4° | **DFA** | Complexidade | Autocorrelação de longo alcance |
| 5° | **RPDE** | Complexidade | Dinâmica caótica da glote |
| 6° | **test_time** | Temporal | Posição na trajetória de acompanhamento |

A proeminência do `subject_id` confirma que o aprendizado personalizado por paciente é o principal vetor de desempenho. Os biomarcadores de complexidade dinâmica (PPE, DFA, RPDE) superam as métricas locais (Jitter, Shimmer) no poder discriminativo — alinhado com a literatura sobre caracterização do estado neuromotor global.

### 4.5 Análise por Paciente

A avaliação individual revelou que a maioria dos pacientes (> 85%) apresenta acurácia superior a 80% nas gravações futuras. Pacientes com menor desempenho tendem a ser aqueles com maior variabilidade intra-individual no UPDRS — transições de classe que ocorrem justamente no período de teste. Esse comportamento é clinicamente esperado e reforça a validade do split temporal.

---

## 5. CONCLUSÃO

### 5.1 Síntese dos Resultados

Este trabalho demonstrou que a classificação da progressão da Doença de Parkinson a partir de biomarcadores vocais, com avaliação metodologicamente correta por **split temporal por paciente**, atinge **acurácia de 95,7%** e **AUC-ROC de 0,968** com o XGBoost. O desempenho é viabilizado pela combinação de: (1) `subject_id` como feature — que permite ao modelo aprender a assinatura vocal individual; (2) `test_time` — que captura a progressão temporal; e (3) normalização, SMOTE e regularização adequadas do XGBoost.

O split temporal é a metodologia correta para o problema de telemonitoramento longitudinal: o modelo aprende com o histórico do paciente e prediz estados futuros, simulando o fluxo real de uma plataforma clínica.

### 5.2 Vantagens e Desvantagens do Método

**Vantagens:**
- Alta acurácia (95,7%) em gravações futuras de pacientes conhecidos
- F1 equilibrado entre as 3 classes, sem viés de classe majoritária
- Metodologia temporal correta para séries longitudinais
- `subject_id` como feature permite personalização sem modelos separados por paciente
- Interpretabilidade via importância de features (gain)

**Desvantagens e Limitações:**
- Requer que o paciente seja previamente cadastrado com histórico de gravações
- Para novos pacientes (sem histórico), o desempenho seria inferior
- Apenas 31 pacientes distintos — dataset relativamente pequeno
- A natureza longitudinal não é modelada explicitamente (sequência temporal das gravações)

### 5.3 Trabalhos Futuros

- **Adaptação a novos pacientes:** calibração rápida com as primeiras gravações (few-shot learning)
- **Modelos de séries temporais (LSTM, GRU):** modelar explicitamente a trajetória temporal de progressão
- **SHAP Values:** interpretabilidade por gravação e por paciente
- **Dataset maior:** mais pacientes para reforçar a robustez da avaliação
- **Integração com wearables:** combinar dados de voz com acelerômetros e outros sensores

---

## REFERÊNCIAS

BRAAK, H. *et al.* Staging of brain pathology related to sporadic Parkinson's disease. **Neurobiology of Aging**, v. 24, n. 2, p. 197-211, 2003.

DINESH, K. N.; BHAVANAM, G. R. Parkinson's Disease Prediction using XGBoost Ensemble Learning Method. **International Journal of Advanced Research in Engineering and Technology**, v. 13, n. 3, p. 211-220, 2022.

GOETZ, C. G. *et al.* Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS). **Movement Disorders**, v. 23, n. 15, p. 2129-2170, 2008.

GROVER, S. *et al.* Various Machine Learning Methods and Their Comparative Analysis for the Diagnosis of Parkinson's Disease. In: **ICAICR**, 2018.

HUGHES, A. J. *et al.* Accuracy of clinical diagnosis of idiopathic Parkinson's disease. **Journal of Neurology, Neurosurgery and Psychiatry**, v. 55, n. 3, p. 181-184, 1992.

LITTLE, M. A. *et al.* Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. **IEEE Transactions on Biomedical Engineering**, v. 56, n. 4, p. 1015-1022, 2009.

ORGANIZAÇÃO MUNDIAL DA SAÚDE (OMS). **Parkinson disease**. Genebra: OMS, 2023.

RAMIG, L. O. *et al.* Speech treatment for Parkinson's disease. **Expert Review of Neurotherapeutics**, v. 4, n. 2, p. 299-311, 2004.

TSANAS, A. *et al.* Accurate telemonitoring of Parkinson's disease progression by noninvasive speech tests. **IEEE Transactions on Biomedical Engineering**, v. 57, n. 4, p. 884-893, 2010.

UCI MACHINE LEARNING REPOSITORY. **Parkinsons Telemonitoring Data Set**. Irvine: University of California, 2009.
