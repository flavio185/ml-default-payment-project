# ml-default-payment-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Cenario

### Contexto
Instituicao financeira quer reduzir perdas com inadimplencia prevendo quais clientes provavelmente nao pagarao a proxima fatura.

### Variavel alvo
`default_payment_next_month` (1 = inadimplente, 0 = pagador regular).

### Uso previsto
Priorizar acoes de cobranca, renegociacao e bloqueio preventivo.

## Metricas de avaliacao

| Metrica | Tipo | Descricao |
| :--- | :--- | :--- |
| ROC-AUC | Offline | Capacidade geral de discriminacao |
| Precision@k | Offline | Precisao nos k% mais arriscados |
| Recall@k | Offline | Cobertura de inadimplentes nos k% |
| F1-score | Offline | Equilibrio precisao/recall |
| Custo evitado | Negocio | Valor estimado que deixaria de ser perdido |
| ROI | Negocio | Retorno sobre investimento das acoes preventivas |

## Arquitetura

### Medallion Architecture (Bronze / Silver / Gold)

```
Azure Blob Storage    S3 (Bronze)         S3 (Silver)         S3 (Gold)           MLflow
  CSV bruto      -->  ingest_bronze.py --> clean_data.py   --> feature_pipeline --> training_pipeline
                                           validate_data.py    + Feast materialize  + MLflow logging
```

### Pipelines independentes

| Pipeline | Entrada | Saida | Descricao |
| :--- | :--- | :--- | :--- |
| **Feature Pipeline** | Silver parquet | Gold parquet + Feast online store | Engenharia de features, metadados versionados, materializacao Feast |
| **Training Pipeline** | Gold (via Feast ou parquet direto) | Modelo MLflow | Treina LogisticRegression, RandomForest, SVM; loga metricas e assinaturas |
| **Inference Pipeline** | Modelo MLflow + dados de entrada | Predicoes em batch | Aplica mesma engenharia de features para evitar train-serve skew |

### Feature Store (Feast)

O projeto usa **Feast** para gerenciar features de forma consistente entre treinamento e inferencia:

- **Registry**: SQL (PostgreSQL compartilhado com MLflow)
- **Offline Store**: `file` (PyArrow sobre S3 parquet) — escala para `spark` via Ray alterando uma linha no YAML
- **Online Store**: Redis standalone — escala para Redis Cluster
- **Fronteira**: Feast armazena features **engenheiradas** (age_bin, bill_trend, pay_ratio, utilization) mas **antes** do preprocessing sklearn. StandardScaler/OneHotEncoder ficam no artefato do modelo MLflow

### Serving

**FastAPI** com dois endpoints:

- `POST /predict` — features enviadas no request body (retrocompativel)
- `POST /predict/feast` — lookup de features pelo `customer_id` no Feast online store
- `GET /health` — health check

### CI/CD

| Etapa | Ferramenta | Descricao |
| :--- | :--- | :--- |
| **CI** | GitHub Actions | Lint (ruff) + testes (pytest) + build Docker + push GHCR |
| **GitOps** | GitHub Actions | Atualiza tag da imagem no repo `ml-platform-gitops` |
| **CD** | Argo CD | Sync automatico dos manifests Kubernetes |
| **Orquestracao** | Argo Workflows | DAG: bronze -> silver -> validate -> features -> training |
| **Agendamento** | CronWorkflow | Pipeline diario as 02:00 UTC |
| **Retraining** | Argo Events | Sensor de drift dispara retraining automatico |

## Estrutura do Projeto

```
ml-default-payment-project/
|-- .github/workflows/
|   └── ci.yml                    # CI: lint, test, Docker build, gitops update
|
|-- data_processing/
|   |-- bronze/
|   |   └── ingest_bronze.py      # Ingestao de CSV do Azure Blob
|   |-- silver/
|   |   |-- clean_data.py         # Limpeza, snake_case, customer_id
|   |   └── validate_data.py      # Validacao com Great Expectations
|   └── check_s3.py               # Utilitario para aguardar objetos no S3
|
|-- feast/
|   |-- feature_store.yaml        # Configuracao Feast (registry, offline, online)
|   └── features/
|       └── credit_card_features.py  # Entity, FileSource, FeatureView (27 features)
|
|-- gitops/
|   |-- kustomization.yaml        # Kustomize overlay
|   |-- namespace.yaml            # Namespace ml-credit-default + ResourceQuota + RBAC
|   |-- argo-workflow.yaml        # WorkflowTemplate DAG do pipeline completo
|   |-- cron-workflow.yaml        # CronWorkflow diario
|   |-- argo-events.yaml          # EventSource + Sensor para retraining por drift
|   └── kserve-inference.yaml     # InferenceService KServe
|
|-- ml_classification/
|   |-- config.py                 # Constantes (paths S3, Feast, etc.)
|   |-- features/
|   |   |-- engineering.py        # age_bin, bill_trend, pay_ratio, utilization
|   |   |-- preprocessing.py      # StandardScaler/OneHotEncoder com config salva
|   |   └── feast_utils.py        # get_training_features(), get_online_features()
|   |-- modeling/
|   |   |-- data_loader.py        # load_features() e load_features_from_feast()
|   |   |-- models.py             # Definicoes de modelos e hiperparametros
|   |   |-- eval.py               # Metricas (ROC-AUC, F1, precision, recall)
|   |   |-- trainer.py            # Treinamento de modelos
|   |   └── mlflow_logger.py      # Logging de metricas e artefatos no MLflow
|   |-- pipelines/
|   |   |-- feature_pipeline.py   # Silver -> Gold + Feast materialize
|   |   |-- training_pipeline_v2.py # Treino com logging MLflow
|   |   └── inference_pipeline.py # Inferencia batch
|   └── serving/
|       └── app.py                # FastAPI (/predict, /predict/feast, /health)
|
|-- tests/
|   |-- features/                 # Testes de engenharia e preprocessing
|   |-- ml_classification/modeling/ # Testes de modelos, eval, pipeline
|   └── pipelines/                # Testes de integracao
|
|-- Dockerfile                    # Imagem Docker (base ml-platform-base)
|-- Makefile                      # Comandos de conveniencia
|-- make.py                       # CLI typer com comandos de pipeline
|-- pyproject.toml                # Dependencias (uv + flit)
└── .pre-commit-config.yaml       # ruff + pytest pre-commit hooks
```

## Desenvolvimento

### Setup

```bash
make create_environment    # Cria venv Python 3.12 com uv
make requirements          # uv sync — instala dependencias
```

### Comandos

```bash
# Qualidade de codigo
make lint                  # ruff format --check && ruff check
make format                # ruff check --fix && ruff format
make test                  # uv run pytest tests/

# Pipelines individuais
python make.py bronze              # Ingerir dados do Azure Blob
python make.py silver              # Limpar e normalizar dados
python make.py validate            # Validacao Great Expectations
python make.py feature-pipeline    # Silver -> Gold + Feast
python make.py training-pipeline   # Treinar modelos + MLflow
python make.py full-pipeline       # Pipeline completo

# Docker
docker build -t ml-default-payment .
docker run ml-default-payment python make.py full-pipeline
```

### Em producao (Kubernetes)

```bash
# Submeter pipeline manualmente via Argo Workflows
argo submit -n ml-credit-default --from workflowtemplate/full-ml-pipeline

# Verificar CronWorkflow
argo cron list -n ml-credit-default

# Verificar sync Argo CD
argocd app get credit-default
```

## Caminhos de escala

| Componente | Atual (minimo) | Escala para |
| :--- | :--- | :--- |
| Feast offline store | `file` (PyArrow) | `spark` no Ray |
| Feast online store | Redis standalone | Redis Cluster (6+ nos) |
| Training step | `container-step` (pod unico) | `rayjob-step` (Ray distribuido) |
| Inferencia | KServe 0-5 replicas | HPA com mais replicas |

--------
