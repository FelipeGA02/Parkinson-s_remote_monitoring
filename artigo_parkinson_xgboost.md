# Classificação da Progressão da Doença de Parkinson por Telemonitoramento de Voz Utilizando XGBoost

**Centro Universitário Dom Helder — Disciplina de Ciência de Dados**  
**Orientador:** Prof. Dr. Marcos W. Rodrigues

---

## RESUMO

A Doença de Parkinson (DP) é a segunda doença neurodegenerativa mais prevalente no mundo, afetando aproximadamente 10 milhões de pessoas, sendo cerca de 200 mil no Brasil. O acompanhamento contínuo da progressão da doença é um desafio clínico, especialmente em regiões com acesso limitado a especialistas. Este trabalho propõe a classificação do estágio de progressão da DP em três níveis — leve, moderado e grave — a partir de 16 medidas biomédicas vocais, utilizando o algoritmo XGBoost. A base de dados utilizada é o Oxford Parkinson's Disease Telemonitoring Dataset (UCI), composta por 5.875 registros de 31 pacientes distintos. Uma contribuição metodológica central deste trabalho é a adoção de validação cruzada por paciente (`GroupKFold`, n=5), que evita o *data leakage* decorrente da presença de múltiplas gravações de um mesmo indivíduo em treino e teste — falha que, conforme demonstrado, infla artificialmente a acurácia de 38% para 98%. Com avaliação correta, o XGBoost atingiu **acurácia de 38,5% ± 2,7%** e **AUC-ROC de 0,535 ± 0,024**, evidenciando que a generalização entre pacientes não vistos é um desafio substancial e que modelos personalizados são necessários para uso clínico efetivo.

**Palavras-chave:** Doença de Parkinson. Telemonitoramento. Voz. XGBoost. Classificação Multiclasse. Data Leakage. GroupKFold. UPDRS.

---

## 1. INTRODUÇÃO

A Doença de Parkinson (DP) é uma enfermidade neurodegenerativa progressiva causada pela perda de neurônios dopaminérgicos na substância negra do cérebro. Segundo a Organização Mundial da Saúde (OMS, 2023), mais de 10 milhões de pessoas são afetadas mundialmente, com prevalência crescente em função do envelhecimento populacional. No Brasil, estima-se a existência de aproximadamente 200 mil pacientes diagnosticados (ABEN, 2022). A doença manifesta-se clinicamente por tremores de repouso, rigidez muscular, bradicinesia e instabilidade postural, além de sintomas não motores como alterações de voz e fala, que surgem em estágios precoces (Braak *et al.*, 2003).

O monitoramento clínico convencional da DP baseia-se na aplicação presencial da Unified Parkinson's Disease Rating Scale (UPDRS), uma escala padronizada que avalia múltiplos domínios da doença. Entretanto, consultas frequentes nem sempre são viáveis, motivando o desenvolvimento de tecnologias de telemonitoramento que permitam a avaliação remota e contínua da evolução da doença. Estudos recentes demonstram que as alterações vocais — detectáveis por medidas acústicas como jitter, shimmer, NHR, HNR, RPDE, DFA e PPE — constituem biomarcadores confiáveis para a avaliação do estado neurológico de pacientes com DP (Little *et al.*, 2009; Tsanas *et al.*, 2010).

Entre os trabalhos relacionados, destaca-se o estudo seminal de Little *et al.* (2009), que demonstrou a viabilidade do uso de atributos de voz para discriminar pacientes com e sem Parkinson, obtendo acurácia superior a 91% com SVM. Tsanas *et al.* (2010) propuseram o uso de regressão CART para predizer pontuações UPDRS a partir de medidas vocais, obtendo correlação de Pearson de até 0,85. Mais recentemente, Dinesh e Bhavanam (2022) aplicaram XGBoost ao diagnóstico binário de Parkinson e reportaram acurácia de 97,4%.

O presente trabalho avança sobre esses estudos ao reformular o problema como **classificação multiclasse de três estágios de progressão** (leve, moderado e grave), o que tem maior aplicabilidade clínica ao orientar decisões terapêuticas diferenciadas. Utiliza-se o XGBoost como algoritmo principal pela sua robustez a outliers, capacidade de modelar relações não lineares e pela disponibilidade nativa de importância de features, essencial para a interpretação clínica.

Este trabalho está organizado em cinco seções: a Seção 2 apresenta os trabalhos relacionados e a fundamentação teórica; a Seção 3 descreve os materiais e métodos; a Seção 4 apresenta os experimentos e análise dos resultados; e a Seção 5 conclui o trabalho e propõe direções futuras.

---

## 2. TRABALHOS RELACIONADOS

### 2.1 Contextualização Clínica da Doença de Parkinson

A DP é uma doença idiopática de curso progressivo e irreversível. O diagnóstico definitivo é histopatológico, mas clinicamente é estabelecido por critérios do UK Parkinson's Disease Society Brain Bank (Hughes *et al.*, 1992). A escala UPDRS (Unified Parkinson's Disease Rating Scale), em sua versão revisada pela Movement Disorder Society (MDS-UPDRS), quantifica a gravidade dos sintomas motores e não motores em uma escala de 0 a 199, sendo amplamente utilizada em ensaios clínicos e no monitoramento longitudinal (Goetz *et al.*, 2008).

As alterações vocais são manifestações frequentes da DP, presentes em até 89% dos pacientes (Ramig *et al.*, 2004). Caracterizam-se por redução da intensidade, hipofonia, monotonia de tom e disfonia, resultantes do comprometimento do controle motor da laringe. Essas características são mensuráveis por análise acústica computadorizada, tornando a voz um biomarcador não invasivo e de baixo custo para o telemonitoramento.

### 2.2 Aplicação de Técnicas de Data Mining ao Diagnóstico por Voz

**Little *et al.* (2009)** desenvolveram um sistema de triagem para DP baseado em medidas de disfonias sustentadas. Utilizando SVM com kernel RBF e 22 atributos de voz, os autores obtiveram sensibilidade de 99% e especificidade de 91% em validação deixar-um-de-fora. O estudo estabeleceu as bases para o uso de jitter, shimmer, NHR e HNR como marcadores diagnósticos.

**Tsanas *et al.* (2010)** coletaram os dados do Oxford Parkinson's Disease Telemonitoring Dataset e desenvolveram um sistema para predição remota de pontuações UPDRS. Aplicando regressão CART e seleção de features por LASSO, alcançaram MAE de 5,88 pontos para o motor_UPDRS, demonstrando viabilidade do monitoramento remoto baseado em voz.

**Grover *et al.* (2018)** compararam múltiplos algoritmos (SVM, Random Forest, KNN, Naive Bayes) para classificação binária de Parkinson. O Random Forest apresentou melhor desempenho com acurácia de 94,8%. Contudo, o estudo não abordou a multiclasse nem empregou técnicas de balanceamento.

**Dinesh e Bhavanam (2022)** aplicaram XGBoost ao diagnóstico de Parkinson com o dataset UCI Parkinsons, reportando acurácia de 97,4% e F1-Score de 0,97. Os autores destacaram a importância do tuning de hiperparâmetros e do tratamento do desbalanceamento de classes como fatores determinantes para o desempenho.

O presente trabalho diferencia-se desses estudos por três aspectos: (1) abordagem de **classificação multiclasse em três estágios**, ao invés do diagnóstico binário; (2) uso do dataset de telemonitoramento longitudinal, com múltiplas gravações por paciente; e (3) combinação de SMOTE para balanceamento com GridSearchCV para otimização, produzindo um pipeline reproduzível e clinicamente interpretável.

---

## 3. METODOLOGIA

### 3.1 Materiais

A base de dados utilizada é o **Oxford Parkinson's Disease Telemonitoring Dataset** (Tsanas *et al.*, 2010), disponível no repositório UCI Machine Learning Repository. O conjunto contém **5.875 registros** coletados de **42 pacientes** com diagnóstico confirmado de Parkinson, ao longo de aproximadamente seis meses de telemonitoramento. Cada registro representa uma medição de voz individual e é descrito por **19 atributos de entrada** e duas variáveis UPDRS contínuas (motor_UPDRS e total_UPDRS). Não há registros de indivíduos saudáveis; toda a base corresponde a pacientes com Parkinson em diferentes níveis de progressão.

Os 19 atributos preditivos incluem: medidas de perturbação de frequência (Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP), medidas de perturbação de amplitude (Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA), relação harmônicos-ruído (NHR, HNR), e medidas de complexidade e caos vocal (RPDE, DFA, PPE), além de idade, sexo e tempo de teste.

### 3.2 Métodos

As etapas metodológicas são enumeradas a seguir:

**Etapa 1 — Entendimento do domínio:** Revisão da literatura sobre DP, escala UPDRS e biomarcadores vocais. Definição do problema como classificação multiclasse de progressão da doença.

**Etapa 2 — Definição e discretização da variável-alvo:** A variável `total_UPDRS` (escala contínua de 7 a 55 pontos) foi discretizada em três classes com base em limiares clínicos da literatura (Goetz *et al.*, 2008): Leve (total_UPDRS ≤ 25, n=2.188, 37,2%), Moderado (26–40, n=2.681, 45,6%) e Grave (> 40, n=1.006, 17,1%). Essa binarização transforma o problema em classificação multiclasse, possibilitando a orientação de decisões terapêuticas diferenciadas.

**Etapa 3 — Pré-processamento:** Verificação e confirmação da ausência de valores nulos. Remoção de 371 registros com valores extremos (IQR × 3,0), resultando em 5.504 registros. Normalização dos atributos numéricos com `StandardScaler`. Balanceamento com SMOTE (Synthetic Minority Oversampling Technique), igualando as três classes no conjunto de treino a 1.929 amostras cada (total 5.787). Divisão treino/teste estratificada 80/20 (4.403 treino / 1.101 teste antes do SMOTE).

**Etapa 4 — Análise exploratória (EDA):** Geração de heatmap de correlação, boxplots por classe, distribuição da variável-alvo, teste de Kruskal-Wallis e pairplot dos atributos mais discriminativos.

**Etapa 5 — Seleção de atributos:** Avaliação da importância via `feature_importances_` do XGBoost (métrica gain). Identificação dos atributos de maior poder preditivo.

**Etapa 6 — Treinamento do XGBoost:** Treinamento do classificador com objetivo `multi:softmax`, otimização de 5 hiperparâmetros (n_estimators, max_depth, learning_rate, subsample, colsample_bytree) via GridSearchCV com validação cruzada estratificada de 5 folds, avaliação por F1-weighted (48 combinações × 5 folds = 240 ajustes).

**Etapa 7 — Validação e métricas:** Avaliação no conjunto de teste com acurácia, precisão, recall, F1-Score por classe, F1 macro, F1 weighted, AUC-ROC OVR e matriz de confusão.

**Etapa 8 — Interpretação:** Análise da importância das features e implicações clínicas dos resultados.

---

## 4. EXPERIMENTOS E ANÁLISE DE RESULTADOS

### 4.1 Entendimento do Domínio e Modelo Conceitual

A doença de Parkinson afeta progressivamente os neurônios dopaminérgicos, comprometendo o controle fino dos movimentos, incluindo a fonação. Clinicamente, a progressão da doença é avaliada pela escala UPDRS, cujos escores refletem diretamente a gravidade dos sintomas motores. O telemonitoramento vocal propõe que medidas acústicas — obtidas de gravações de fonemas sustentados — possam substituir parcialmente as avaliações presenciais, fornecendo informação contínua e de baixo custo sobre a evolução da doença.

O modelo conceitual adotado neste trabalho define que as características acústicas da voz de um paciente em um dado momento codificam informação suficiente para classificar seu nível atual de progressão da DP. O XGBoost foi escolhido por sua capacidade de capturar relações não lineares entre os atributos vocais e o estágio clínico, sua robustez a outliers e a escalas distintas de atributos, e por fornecer, nativamente, métricas de importância de features que auxiliam a interpretação clínica.

### 4.2 Montagem e Dimensões da Base de Dados

A base de dados carregada de `data/raw/parkinsons_telemonitoring.csv` apresenta as seguintes dimensões:

| Característica | Valor |
|----------------|-------|
| Registros totais | 5.875 |
| Atributos originais | 21 |
| Atributos preditivos (após remoção das UPDRS) | 19 |
| Valores nulos | 0 |
| Sujeitos distintos | 42 |
| Período de coleta | ~6 meses por paciente |

A variável `total_UPDRS` apresentou distribuição assimétrica positiva (média = 29,02; mediana = 27,58; desvio-padrão = 10,70), com valores entre 7,0 e 55,0. A discretização em três classes resultou em distribuição moderadamente desbalanceada: Leve 37,2%, Moderado 45,6%, Grave 17,1% (Figura 1).

> **Figura 1** — Distribuição das classes de progressão (ver `notebooks/data/fig01_distribuicao_classes.png`)  
> **Figura 2** — Distribuição do total_UPDRS com limiares de discretização (ver `notebooks/data/fig05_updrs_discretizacao.png`)

### 4.3 Pré-processamento: Decisões e Justificativas

#### 4.3.0 Identificação e Correção de Data Leakage

Uma análise crítica da metodologia revelou um problema de *data leakage* presente em versões anteriores do experimento e comum na literatura sobre este dataset: o split aleatório de registros individuais (80/20) colocava gravações do mesmo paciente simultaneamente no treino e no teste. Como o dataset contém em média ~178 gravações por paciente ao longo de 6 meses, o modelo aprendia padrões UPDRS individuais (e não biomarcadores vocais generalizáveis), resultando em acurácia artificial de 98%.

A correção adotada foi o uso de `GroupShuffleSplit` para a separação treino/teste e `GroupKFold(n_splits=5)` para a validação cruzada, garantindo que cada fold avalia o modelo em pacientes **nunca vistos** durante o treino. Este é o único procedimento metodologicamente correto para avaliar a capacidade de generalização clínica do modelo.

Adicionalmente, as colunas `age`, `sex` e `test_time` foram excluídas do conjunto de features preditivas: `age` e `sex` são constantes por paciente e funcionariam como identificadores implícitos do indivíduo; `test_time` é um índice temporal, não um biomarcador vocal. Foram mantidos apenas os **16 atributos biomédicos de voz** (Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE).

#### 4.3.1 Discretização da Variável-Alvo

Os limiares adotados (≤25 = Leve, 26–40 = Moderado, >40 = Grave) foram baseados nas recomendações da Movement Disorder Society (Goetz *et al.*, 2008), que classifica escores totais da UPDRS III em faixas de comprometimento clinicamente significativas. Esta discretização é preferível ao uso da mediana (abordagem puramente estatística) por preservar relevância clínica.

#### 4.3.2 Tratamento de Outliers

Foram removidos 371 registros (6,3% do total) com valores além de 3 IQR nas features numéricas. O fator multiplicador 3,0 (em vez do convencional 1,5) foi escolhido para preservar casos de progressão severa legítimos, que naturalmente apresentam valores extremos em Jitter e Shimmer.

#### 4.3.3 Normalização

O `StandardScaler` foi aplicado após a divisão treino/teste, com os parâmetros (µ, σ) estimados exclusivamente no conjunto de treino para evitar data leakage. A normalização foi necessária dado que os atributos possuem escalas distintas (ex.: Jitter(%) em torno de 0,006 versus age em torno de 64).

#### 4.3.4 Balanceamento com SMOTE

A classe Grave representava apenas 17,1% dos dados, gerando risco de viés do modelo para as classes majoritárias. O SMOTE foi aplicado exclusivamente no conjunto de treino, gerando amostras sintéticas por interpolação entre vizinhos mais próximos na classe minoritária, igualando todas as classes a 1.929 amostras.

#### 4.3.5 Divisão Treino/Teste

Divisão estratificada 80/20 com `random_state=42`, preservando as proporções originais das classes no conjunto de teste (sem SMOTE), garantindo avaliação em distribuição realista.

### 4.4 Análise Exploratória

A matriz de correlação (Figura 3) revelou alta multicolinearidade entre os atributos de Jitter (todos com correlações > 0,95 entre si) e entre os de Shimmer (> 0,92), sugerindo redundância de informação. A correlação de Pearson com a variável-alvo foi mais expressiva para HNR (r = −0,47), DFA (r = −0,38) e PPE (r = +0,41), indicando esses como os atributos mais informativos para a discriminação dos estágios de progressão.

Os boxplots (Figura 4) confirmam padrões clinicamente coerentes: pacientes na classe Grave apresentam valores sistematicamente menores de HNR (mais ruído vocal) e maiores de PPE (maior irregularidade de pitch) em comparação à classe Leve. O teste de Kruskal-Wallis indicou diferença estatisticamente significativa entre as classes para todos os 19 atributos (p < 0,001).

> **Figura 3** — Matriz de correlação (ver `notebooks/data/fig02_heatmap_correlacao.png`)  
> **Figura 4** — Boxplots dos atributos-chave por classe (ver `notebooks/data/fig03_boxplots_por_classe.png`)

### 4.5 Seleção de Atributos

A análise de importância do XGBoost treinado identificou os atributos de maior ganho médio na construção das árvores (Figura 8). Os cinco principais foram:

| Rank | Atributo | Interpretação Biomédica |
|------|----------|------------------------|
| 1° | **HNR** | Relação harmônicos/ruído — queda indica deterioração vocal |
| 2° | **PPE** | Entropia do período de pitch — irregularidade no controle de voz |
| 3° | **DFA** | Análise de flutuação sem tendência — autocorrelação de longo alcance |
| 4° | **RPDE** | Densidade de recorrência — dinâmica caótica da glote |
| 5° | **Shimmer** | Variação de amplitude ciclo-a-ciclo |

Esses atributos capturam, em conjunto, tanto a perturbação de frequência (jitter e PPE), a amplitude (shimmer) quanto a complexidade dinâmica (RPDE, DFA) do sinal vocal, fornecendo uma representação multidomínio do estado neuromotor do paciente.

### 4.6 Treinamento do XGBoost: Configuração e Validação

O modelo XGBoost foi configurado com objetivo `multi:softmax` e avaliado com `GroupKFold(n_splits=5)`, garantindo que cada fold utilize pacientes completamente distintos entre treino e teste. A Tabela 1 apresenta a configuração do modelo:

**Tabela 1 — Configuração do XGBoost**

| Hiperparâmetro | Valor | Justificativa |
|----------------|-------|---------------|
| `objective` | `multi:softmax` | Classificação multiclasse |
| `n_estimators` | **200** | Convergência adequada com taxa de aprendizado de 0,1 |
| `max_depth` | **5** | Equilíbrio entre complexidade e generalização cross-paciente |
| `learning_rate` | **0,1** | Taxa padrão com 200 estimadores |
| `subsample` | **0,8** | Subsampling para regularização (evita overfitting por paciente) |
| `colsample_bytree` | **0,8** | Uso de 80% das features por árvore |

### 4.7 Validação: Métricas Completas

Os resultados são apresentados de forma consolidada sobre as predições de todos os 5 folds do `GroupKFold`, totalizando 5.504 registros de 31 pacientes (nenhum paciente avaliado em seu próprio fold de treino).

**Tabela 2 — Resultados do XGBoost por Classe (GroupKFold 5-fold consolidado)**

| Classe | Precisão | Recall | F1-Score | Suporte |
|--------|----------|--------|----------|---------|
| Leve (0) | 0,45 | 0,46 | **0,45** | 2.140 |
| Moderado (1) | 0,46 | 0,41 | **0,43** | 2.412 |
| Grave (2) | 0,13 | 0,16 | **0,14** | 952 |
| **Macro avg** | **0,35** | **0,34** | **0,34** | 5.504 |
| **Weighted avg** | **0,40** | **0,38** | **0,39** | 5.504 |

**Tabela 3 — Métricas Globais de Desempenho**

| Métrica | Valor |
|---------|-------|
| Acurácia (média ± std) | **38,5% ± 2,7%** |
| F1-Score Macro | **0,34** |
| F1-Score Weighted | **0,39** |
| AUC-ROC (OVR weighted, média ± std) | **0,535 ± 0,024** |
| Baseline aleatório (3 classes) | ~33,3% / AUC 0,500 |

Os resultados são modestos, porém academicamente honestos. A acurácia de 38,5% supera marginalmente o baseline aleatório (33,3%), e o AUC-ROC de 0,535 indica discriminação ligeiramente acima do acaso. A classe Grave apresentou o pior desempenho (F1=0,14), reflexo da dificuldade de generalização para padrões vocais de progressão severa em pacientes não vistos.

Esses resultados contrastam radicalmente com os 98% obtidos com split aleatório de registros, reforçando que a metodologia de avaliação é determinante para a validade científica do trabalho.

> **Figura 6** — Matriz de Confusão consolidada (ver `notebooks/data/fig07_confusion_matrix.png`)  
> **Figura 7** — Curvas ROC OVR (ver `notebooks/data/fig10_roc_curves.png`)  
> **Figura 8** — Importância das Features (ver `notebooks/data/fig08_feature_importance.png`)  
> **Figura 9** — Métricas por Classe (ver `notebooks/data/fig09_metricas_por_classe.png`)  
> **Figura 10** — Desempenho por Fold (ver `notebooks/data/fig11_desempenho_por_fold.png`)

### 4.8 Interpretação: Atributos Mais Relevantes e Implicações Clínicas

Apesar do desempenho de generalização modesto, a análise de importância de features (Figura 8) revela padrões clinicamente coerentes: HNR, PPE, DFA e RPDE consistentemente se destacam como os atributos vocais de maior ganho nas árvores de decisão.

Do ponto de vista clínico, esses resultados são interpretados como:

- **HNR (Harmonic-to-Noise Ratio):** redução progressiva na relação harmônicos/ruído é um marcador da deterioração do controle motor laríngeo — biologicamente plausível e amplamente reportado na literatura.
- **PPE (Pitch Period Entropy) e RPDE:** capturam a irregularidade e caos da vibração das pregas vocais em múltiplas escalas temporais; são mais informativos do que métricas locais como Jitter e Shimmer para caracterizar o estado neuromotor global.
- **DFA (Detrended Fluctuation Analysis):** mede a autocorrelação de longo alcance no sinal vocal; desvios do padrão saudável indicam perturbações no sistema de controle central.

Contudo, o fraco desempenho de generalização entre pacientes (AUC ≈ 0,535) sugere que esses biomarcadores, embora biologicamente relevantes, apresentam alta variabilidade interindividual: pacientes com o mesmo estágio UPDRS podem ter perfis vocais muito distintos. Esse achado é consistente com Tsanas *et al.* (2010), que demonstraram que modelos dentro do mesmo paciente (personalizados) superam substancialmente modelos entre pacientes (genéricos) neste dataset.

A alta multicolinearidade entre variantes de Jitter e Shimmer (correlação > 0,95 entre si) indica que muitas dessas medidas são redundantes — um subconjunto reduzido poderia ser utilizado sem perda significativa de informação.

---

## 5. CONCLUSÃO

### 5.1 Síntese dos Resultados

Este trabalho demonstrou que a classificação da progressão da Doença de Parkinson a partir de biomarcadores vocais é um problema significativamente mais difícil do que a literatura frequentemente reporta. Ao adotar avaliação metodologicamente correta via `GroupKFold` por paciente, o XGBoost atingiu acurácia de **38,5% ± 2,7%** e AUC-ROC de **0,535 ± 0,024** — modestos, porém honestos.

Uma contribuição central deste trabalho é a demonstração quantitativa do impacto do *data leakage* por paciente: o split aleatório de registros infla artificialmente a acurácia de 38% para 98%, distorcendo a percepção de viabilidade clínica do modelo. Este resultado alerta para um erro metodológico prevalente em trabalhos que utilizam datasets longitudinais de telemonitoramento.

Os atributos HNR, PPE, DFA e RPDE se destacaram como os mais informativos nas árvores do XGBoost, alinhando-se com evidências da neurologia clínica sobre as manifestações vocais do Parkinson. Entretanto, a alta variabilidade interindividual dos perfis vocais limita a generalização entre pacientes não vistos, indicando que modelos personalizados (dentro do mesmo paciente) são necessários para atingir desempenho clinicamente útil.

### 5.2 Vantagens e Desvantagens do Método

**Vantagens do XGBoost neste domínio:**

- **Robustez a outliers:** o boosting com árvores de decisão é menos sensível a valores extremos do que modelos lineares.
- **Interpretabilidade parcial:** a importância por gain permite identificar quais biomarcadores vocais mais contribuem para a classificação.
- **Eficiência computacional:** o treinamento completo com validação cruzada por paciente foi concluído em poucos minutos.
- **Sem suposições distribucionais:** ao contrário de modelos como LDA ou Naive Bayes, não exige normalidade dos atributos.

**Desvantagens e Limitações:**

- **Generalização cross-paciente limitada:** acurácia de 38,5% evidencia que os biomarcadores vocais têm alta variabilidade interindividual — o modelo generaliza mal para pacientes não vistos.
- **Dataset pequeno (31 pacientes):** poucos sujeitos distintos limitam a robustez da avaliação por `GroupKFold` e tornam os resultados sensíveis à composição dos folds.
- **Natureza longitudinal ignorada:** múltiplas gravações por paciente ao longo do tempo são tratadas como independentes; modelos temporais poderiam aproveitar a trajetória de progressão.
- **Ausência de grupo controle saudável:** o dataset contém apenas pacientes com Parkinson confirmado; os resultados não se aplicam ao diagnóstico diferencial Parkinson vs. saudável.
- **SMOTE pode gerar amostras não realistas:** especialmente para a classe Grave, com menor representação.

### 5.3 Trabalhos Futuros

- **Modelos personalizados (within-patient):** treinar modelos individuais por paciente com as primeiras semanas de gravação e aplicá-los às semanas seguintes — abordagem que Tsanas *et al.* (2010) demonstraram ser significativamente mais eficaz neste dataset.
- **Modelos de séries temporais (LSTM, GRU):** explorar a natureza longitudinal para modelar trajetórias temporais de progressão vocal, ao invés de tratar cada gravação de forma isolada.
- **SHAP Values para interpretabilidade local:** obter explicações por paciente, identificando quais biomarcadores vocais são mais informativos para cada indivíduo.
- **Aumento de amostra:** coletar dados de mais pacientes para tornar a avaliação `GroupKFold` mais robusta e reduzir a variância entre folds.
- **Dataset com grupo controle:** utilizar o UCI Parkinsons Dataset original (197 registros, com controles saudáveis) para explorar o diagnóstico diferencial Parkinson vs. saudável.
- **Aprendizado de transferência:** pré-treinar com outros datasets de doenças neurodegenerativas e fazer *fine-tuning* com dados de Parkinson.

---

## REFERÊNCIAS

BRAAK, H. *et al.* Staging of brain pathology related to sporadic Parkinson's disease. **Neurobiology of Aging**, v. 24, n. 2, p. 197-211, 2003.

DINESH, K. N.; BHAVANAM, G. R. Parkinson's Disease Prediction using XGBoost Ensemble Learning Method. **International Journal of Advanced Research in Engineering and Technology**, v. 13, n. 3, p. 211-220, 2022.

GOETZ, C. G. *et al.* Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS): Scale presentation and clinimetric testing results. **Movement Disorders**, v. 23, n. 15, p. 2129-2170, 2008.

GROVER, S. *et al.* Various Machine Learning Methods and Their Comparative Analysis for the Diagnosis of Parkinson's Disease. In: **International Conference on Advanced Informatics for Computing Research (ICAICR)**, 2018.

HUGHES, A. J. *et al.* Accuracy of clinical diagnosis of idiopathic Parkinson's disease: a clinico-pathological study of 100 cases. **Journal of Neurology, Neurosurgery and Psychiatry**, v. 55, n. 3, p. 181-184, 1992.

LITTLE, M. A. *et al.* Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. **IEEE Transactions on Biomedical Engineering**, v. 56, n. 4, p. 1015-1022, 2009.

ORGANIZAÇÃO MUNDIAL DA SAÚDE (OMS). **Parkinson disease**. Genebra: OMS, 2023. Disponível em: https://www.who.int/news-room/fact-sheets/detail/parkinson-disease. Acesso em: 28 maio 2026.

RAMIG, L. O. *et al.* Speech treatment for Parkinson's disease. **Expert Review of Neurotherapeutics**, v. 4, n. 2, p. 299-311, 2004.

TSANAS, A. *et al.* Accurate telemonitoring of Parkinson's disease progression by noninvasive speech tests. **IEEE Transactions on Biomedical Engineering**, v. 57, n. 4, p. 884-893, 2010.

UCI MACHINE LEARNING REPOSITORY. **Parkinsons Telemonitoring Data Set**. Irvine: University of California, 2009. Disponível em: https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring.
