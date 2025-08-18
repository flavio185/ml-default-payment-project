import pytest
from data_processing.silver import clean_data
import pandas as pd

def test_main_runs(tmp_path):
    # Create a dummy bronze parquet file
    df = pd.DataFrame({
        'SEX': [1, 2], 'EDUCATION': [1, 2], 'MARRIAGE': [1, 2], 'default_payment_next_month': [0, 1],
        'BILL_AMT1': [100, 200], 'BILL_AMT2': [100, 200], 'BILL_AMT3': [100, 200], 'BILL_AMT4': [100, 200], 'BILL_AMT5': [100, 200], 'BILL_AMT6': [100, 200],
        'PAY_AMT1': [10, 20], 'PAY_AMT2': [10, 20], 'PAY_AMT3': [10, 20], 'PAY_AMT4': [10, 20], 'PAY_AMT5': [10, 20], 'PAY_AMT6': [10, 20],
        'LIMIT_BAL': [1000, 2000]
    })
    input_path = tmp_path/"credit_card_default.parquet"
    output_path = tmp_path/"silver_credit_card_default.parquet"
    df.to_parquet(input_path)
    clean_data.main(input_path=input_path, output_path=output_path)
    assert output_path.exists()
    out_df = pd.read_parquet(output_path)
    assert "education" in out_df.columns
    assert "marriage" in out_df.columns
    assert out_df["education"].isin([1,2,4]).all()
    assert out_df["marriage"].isin([1,2,3]).all()
