# Pipeline de Processamento de Dados

Este documento detalha as etapas de processamento de dados do projeto, desde a ingestão até a validação e geração de features.

## 1. Ingestão (Bronze)
- Script: `data_processing/bronze/ingest_bronze.py`
- Baixa e salva o dataset original em formato Parquet.

## 2. Limpeza (Silver)
- Script: `data_processing/silver/clean_data.py`
- Renomeia colunas, trata valores inválidos e ajusta tipos.

## 3. Validação
- Script: `data_processing/silver/validate_data.py`
- Valida os dados com Great Expectations e salva logs.

## 4. Feature Engineering (Gold)
- Script: `data_processing/gold/build_features.py`
- Cria variáveis derivadas relevantes para modelagem.

Todos os scripts podem ser executados individualmente ou orquestrados em pipelines.
