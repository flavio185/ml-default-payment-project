# ml-default-payment-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Definição do cenário
### Contexto: 
Instituição financeira quer reduzir perdas com inadimplência prevendo quais clientes provavelmente não pagarão a próxima fatura.

### Variável alvo: 
    default (1 = inadimplente, 0 = pagador regular).

### Uso previsto: 
Priorizar ações de cobrança, renegociação e bloqueio preventivo.

## Métricas de avaliação do modelo

Como é um problema de classificação desbalanceado (poucos inadimplentes comparado ao total), vamos usar:

### Métricas offline:

ROC-AUC (capacidade geral de discriminação).

Precision@k (precisão nos k% mais arriscados).

Recall@k (cobertura de inadimplentes nos k%).

F1-score (equilíbrio precisão/recall).

### Métrica de negócio:

Custo evitado: valor total estimado que deixaria de ser perdido se as ações fossem tomadas nos clientes previstos como default.

Retorno sobre investimento (ROI) das ações preventivas.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ml_classification and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ml_classification   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ml_classification a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

