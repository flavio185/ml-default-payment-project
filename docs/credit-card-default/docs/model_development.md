# Desenvolvimento e Treinamento de Modelos

Este documento detalha o fluxo de experimentação, treino e rastreamento de modelos do projeto.

## 1. Modelos Baseline
- Script: `ml_classification/modeling/train.py`
- Treina um modelo de regressão logística baseline, logando métricas (ROC-AUC, Precision@10%, Recall@10%) e o modelo no MLflow.


## 3. Rastreamento de Experimentos
- Todos os experimentos são registrados no MLflow, permitindo comparação e reprodutibilidade.

Consulte o [index.md](index.md) para visão geral do fluxo e comandos principais.
