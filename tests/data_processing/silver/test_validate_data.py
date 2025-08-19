import pytest
from data_processing.silver import validate_data
import pandas as pd

def test_main_runs(monkeypatch, tmp_path):
    # Create a dummy silver parquet file that passes all expectations
    df = pd.DataFrame({
        'age': [25, 30],
        'bill_amt1': [100, 200], 'bill_amt6': [150, 250],
        'pay_amt1': [10, 20], 'pay_amt2': [10, 20], 'pay_amt3': [10, 20],
        'pay_amt4': [10, 20], 'pay_amt5': [10, 20], 'pay_amt6': [10, 20],
        'bill_amt2': [100, 200], 'bill_amt3': [100, 200], 'bill_amt4': [100, 200], 'bill_amt5': [100, 200],
        'limit_bal': [1000, 2000],
        'sex': [1, 2],
        'education': [1, 2],
        'marriage': [1, 2],
        'default_payment_next_month': [0, 1]
    })
    input_path = tmp_path/"credit_card_default.parquet"
    log_output = tmp_path/"validation_results.json"
    df.to_parquet(input_path)
    # Patch LOGS_DIR to tmp_path to avoid writing to real logs
    monkeypatch.setattr(validate_data, "LOGS_DIR", tmp_path)
    validate_data.main(input_path=input_path, log_output=log_output)
    assert log_output.exists()
