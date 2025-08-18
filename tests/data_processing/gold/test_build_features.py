import pytest
from data_processing.gold import build_features
import pandas as pd

def test_main_runs(tmp_path):
    # Create a dummy silver parquet file
    df = pd.DataFrame({
        'age': [30, 40],
        'bill_amt1': [100, 200], 'bill_amt6': [150, 250],
        'pay_amt1': [10, 20], 'pay_amt2': [10, 20], 'pay_amt3': [10, 20],
        'pay_amt4': [10, 20], 'pay_amt5': [10, 20], 'pay_amt6': [10, 20],
        'bill_amt2': [100, 200], 'bill_amt3': [100, 200], 'bill_amt4': [100, 200], 'bill_amt5': [100, 200],
        'limit_bal': [1000, 2000]
    })
    input_path = tmp_path/"credit_card_default.parquet"
    output_path = tmp_path/"credit_card_default_features.parquet"
    df.to_parquet(input_path)
    build_features.main(input_path=input_path, output_path=output_path)
    assert output_path.exists()
    out_df = pd.read_parquet(output_path)
    assert "age_bin" in out_df.columns
    assert "bill_trend" in out_df.columns
    assert "pay_ratio" in out_df.columns
    assert "utilization" in out_df.columns
