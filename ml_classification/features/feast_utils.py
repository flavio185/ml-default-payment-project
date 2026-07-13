"""Feast feature store utilities for training and serving.

Provides wrappers around Feast's get_historical_features() and get_online_features()
for consistent feature retrieval across training and inference.
"""

import os

from feast import FeatureStore
from loguru import logger
import pandas as pd

from ml_classification.config import FEAST_REPO_PATH

# Lista de features para recuperar do Feast
FEATURE_REFS = [
    "credit_card_features:limit_bal",
    "credit_card_features:sex",
    "credit_card_features:education",
    "credit_card_features:marriage",
    "credit_card_features:age",
    "credit_card_features:pay_0",
    "credit_card_features:pay_2",
    "credit_card_features:pay_3",
    "credit_card_features:pay_4",
    "credit_card_features:pay_5",
    "credit_card_features:pay_6",
    "credit_card_features:bill_amt1",
    "credit_card_features:bill_amt2",
    "credit_card_features:bill_amt3",
    "credit_card_features:bill_amt4",
    "credit_card_features:bill_amt5",
    "credit_card_features:bill_amt6",
    "credit_card_features:pay_amt1",
    "credit_card_features:pay_amt2",
    "credit_card_features:pay_amt3",
    "credit_card_features:pay_amt4",
    "credit_card_features:pay_amt5",
    "credit_card_features:pay_amt6",
    "credit_card_features:age_bin",
    "credit_card_features:bill_trend",
    "credit_card_features:pay_ratio",
    "credit_card_features:utilization",
]


def _get_store() -> FeatureStore:
    """Cria uma instância do FeatureStore usando o repo path configurado."""
    repo_path = os.environ.get("FEAST_REPO_PATH", str(FEAST_REPO_PATH))
    return FeatureStore(repo_path=repo_path)


def get_training_features(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Recupera features históricas do Feast para treinamento.

    Args:
        entity_df: DataFrame com colunas 'customer_id' e 'event_timestamp'

    Returns:
        DataFrame com todas as features históricas (point-in-time join)
    """
    store = _get_store()
    logger.info(f"Recuperando features históricas para {len(entity_df)} entidades")

    training_df = store.get_historical_features(
        entity_rows=entity_df,
        features=FEATURE_REFS,
    ).to_df()

    logger.info(
        f"Features recuperadas: {len(training_df)} linhas, {len(training_df.columns)} colunas"
    )
    return training_df


def get_online_features(entity_rows: list[dict]) -> pd.DataFrame:
    """Recupera features online do Feast para inferência em tempo real.

    Args:
        entity_rows: Lista de dicts com 'customer_id' para cada entidade

    Returns:
        DataFrame com features atuais do online store
    """
    store = _get_store()
    logger.info(f"Recuperando features online para {len(entity_rows)} entidades")

    online_features = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=entity_rows,
    ).to_df()

    logger.info(f"Features online recuperadas: {len(online_features)} linhas")
    return online_features
