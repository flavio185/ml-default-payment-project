import pytest
from ml_classification import plots
from pathlib import Path

def test_main_runs(tmp_path):
    # Create a dummy csv file
    csv_path = tmp_path/"dataset.csv"
    csv_path.write_text("col1,col2\n1,2\n3,4\n")
    output_path = tmp_path/"plot.png"
    plots.main(input_path=csv_path, output_path=output_path)
    # No assertion, just check code runs
