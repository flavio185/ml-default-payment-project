"""Feast feature definitions for credit card default prediction.

Stores raw + engineered features BEFORE sklearn preprocessing.
StandardScaler/OneHotEncoder stay in the MLflow model pipeline
because they are model-specific transformations.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

from ml_classification.config import S3_BUCKET

# Entidade: cada observação é um cliente de cartão de crédito
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    description="Identificador único do cliente (chave substituta)",
)

# Fonte de dados: Gold layer no S3
gold_source = FileSource(
    name="credit_card_gold",
    path=f"s3://{S3_BUCKET}/gold/credit_card_default_features.parquet",
    timestamp_field="ingestion_time",
)

# FeatureView com todas as features (raw + engenharia) da Gold layer
credit_card_features = FeatureView(
    name="credit_card_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        # Features brutas
        Field(name="limit_bal", dtype=Float64),
        Field(name="sex", dtype=Int64),
        Field(name="education", dtype=Int64),
        Field(name="marriage", dtype=Int64),
        Field(name="age", dtype=Int64),
        Field(name="pay_0", dtype=Int64),
        Field(name="pay_2", dtype=Int64),
        Field(name="pay_3", dtype=Int64),
        Field(name="pay_4", dtype=Int64),
        Field(name="pay_5", dtype=Int64),
        Field(name="pay_6", dtype=Int64),
        Field(name="bill_amt1", dtype=Float64),
        Field(name="bill_amt2", dtype=Float64),
        Field(name="bill_amt3", dtype=Float64),
        Field(name="bill_amt4", dtype=Float64),
        Field(name="bill_amt5", dtype=Float64),
        Field(name="bill_amt6", dtype=Float64),
        Field(name="pay_amt1", dtype=Float64),
        Field(name="pay_amt2", dtype=Float64),
        Field(name="pay_amt3", dtype=Float64),
        Field(name="pay_amt4", dtype=Float64),
        Field(name="pay_amt5", dtype=Float64),
        Field(name="pay_amt6", dtype=Float64),
        # Features de engenharia
        Field(name="age_bin", dtype=String),
        Field(name="bill_trend", dtype=Float64),
        Field(name="pay_ratio", dtype=Float64),
        Field(name="utilization", dtype=Float64),
    ],
    source=gold_source,
    online=True,
)
