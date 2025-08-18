# Credit Card Default Prediction – Case Data Masters

## Objetivo do Case

Desenvolver uma solução completa de Engenharia de Machine Learning e MLOps para prever inadimplência de clientes de cartão de crédito (credit card default).
A solução deverá abranger desde a ingestão de dados até o monitoramento em produção, atendendo aos requisitos de treinamento, CI/CD, orquestração, gerenciamento de artefatos, observabilidade e escalabilidade.

## 1. Descrição do Problema

Instituições financeiras precisam minimizar perdas com inadimplência identificando, de forma antecipada, quais clientes provavelmente não irão pagar a próxima fatura.

Neste case, o objetivo é criar um modelo capaz de prever a probabilidade de default para cada cliente no próximo ciclo de cobrança, permitindo que ações preventivas (ex.: renegociação, bloqueio temporário, cobrança antecipada) sejam tomadas.

## 2. Variável Alvo

Coluna: default

Valores:

1 → Cliente inadimplente no próximo ciclo

0 → Cliente que pagou normalmente

## 3. Métricas de Avaliação
- Métricas Offline (modelo)

    - ROC-AUC → medida global de separação entre bons e maus pagadores.

    - Precision@k → precisão dentro dos k% clientes mais arriscados.

    - Recall@k → cobertura de inadimplentes no top k%.

    - F1-score → equilíbrio entre precisão e recall.

- Métricas de Negócio

    - Custo evitado → valor estimado que deixaria de ser perdido caso ações fossem tomadas nos clientes previstos como default.

    - ROI das ações preventivas → retorno financeiro das medidas.


## 4. Metas

- ROC-AUC ≥ 0.80

- Precision@10% ≥ 40%

- Drift de dados monitorado continuamente

## 5. Fonte dos Dados

Dataset simulado com dados históricos de clientes, incluindo informações de limite de crédito, perfil demográfico, histórico de pagamentos e valores de fatura.

Colunas principais:

    LIMIT_BAL – limite de crédito

    SEX – gênero

    EDUCATION – nível de educação

    MARRIAGE – estado civil

    AGE – idade

    PAY_0 a PAY_6 – histórico de pagamentos

    BILL_AMT1 a BILL_AMT6 – valores de fatura

    PAY_AMT1 a PAY_AMT6 – valores pagos

    default – variável alvo

6. Estrutura do Repositório
```css
.
├── data
│   └── bronze
│       └── credit_card_default.parquet
├── notebooks
│   └── eda_day2.ipynb
├── src
│   ├── ingest_bronze.py
│   └── ...
├── README.md
└── requirements.txt
```
7. Instruções de Ingestão

Colocar o dataset original credit_card_default.csv na raiz do projeto.

Executar:

make dataset


O arquivo tratado será salvo em data/bronze/credit_card_default.parquet.

8. Próximos Passos

Criar pipeline de limpeza (camada Silver).

Implementar validações de dados com Great Expectations.

Definir features iniciais e persistir no Feature Store.

Iniciar treinamento baseline com Logistic Regression.

### Camada Silver (Dia 3)

A camada **Silver** contém os dados limpos e padronizados, prontos para análises e modelagem.  
Transformações aplicadas:

- Colunas renomeadas para `snake_case`.
- Valores inválidos em `EDUCATION` e `MARRIAGE` tratados.
- Tipos de dados ajustados:
  - Variáveis categóricas (`SEX`, `EDUCATION`, `MARRIAGE`, `DEFAULT`) como `int`.
  - Variáveis numéricas (`LIMIT_BAL`, `BILL_AMTx`, `PAY_AMTx`) como `float`.
- Dados salvos em `/data/silver/credit_card_default.parquet`.

---

### Camada Gold (Dia 5)

A camada **Gold** contém os dados prontos para treinamento de modelos, com **features derivadas** que capturam padrões relevantes de risco de crédito.

Features criadas:
- `age_bin` → Agrupa clientes em faixas etárias (18–25, 26–35, 36–50, 50+), permitindo capturar efeitos não lineares da idade no risco de crédito.
- `bill_trend` → Mede a diferença entre `BILL_AMT6` e `BILL_AMT1`, mostrando se a dívida está crescendo ou diminuindo ao longo do tempo.
- `pay_ratio` → Razão entre a soma de pagamentos (`PAY_AMT1–6`) e a soma das faturas (`BILL_AMT1–6`), indicando capacidade de pagamento.
- `utilization` → Proporção do limite de crédito utilizado (`BILL_AMT6 / LIMIT_BAL`), refletindo o nível de alavancagem do cliente.


Dados salvos em `/data/gold/credit_card_default_features.parquet`.