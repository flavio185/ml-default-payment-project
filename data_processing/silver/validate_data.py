from pathlib import Path
import pandas as pd
import great_expectations as gx
import typer
from loguru import logger
import sys
import json
from datetime import datetime

from ml_classification.config import SILVER_DATA_DIR, VALIDATION_REPORTS_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = SILVER_DATA_DIR / "credit_card_default.parquet",
    log_output: Path = VALIDATION_REPORTS_DIR / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
):
    """
    Day 4 – Data Validation
    Validate Silver dataset using Great Expectations before moving to Gold.
    """

    logger.info(f"Loading Silver dataset from: {input_path}")
    
    df = pd.read_parquet(input_path)
    if df.empty:
        logger.error("Input DataFrame is empty. Exiting validation.")
        sys.exit(1)


    data_source_name = "credit_card_default"
    data_asset_name = "credit_card_default_asset"
    batch_definition_name = "credit_card_default_batch"
    expectation_suite_name = "credit_card_default_suite"

    # Retrieve your Data Context
    context = gx.get_context()

    # Add the Data Source to the Data Context
    data_source = context.data_sources.add_pandas(data_source_name)
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)

    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
    assert batch_definition.name == batch_definition_name

    # Define the Batch Parameters
    batch_parameters = {"dataframe": df}
    # Retrieve the Batch
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    # Create an Expectation Suite
    suite = gx.ExpectationSuite(name=expectation_suite_name)
    # Add Expectations
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="default_payment_next_month"
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="age", min_value=18, max_value=100)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="limit_bal", min_value=1)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="sex", value_set=[1, 2])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="education", value_set=[1, 2, 3, 4])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="marriage", value_set=[1, 2, 3])
    )

    # Add the Expectation Suite to the Context
    context.suites.add_or_update(suite)

    # --- Validation Rules ---
    
    logger.info("Running Great Expectations checks...")

    # Validate the Data Against the Suite
    validation_results = batch.validate(suite)    
    json_result = validation_results.to_json_dict()
    logger.info(validation_results.statistics)
    logger.info("Great Expectations checks complete.")

    # --- Save Logs ---
    log_output.parent.mkdir(parents=True, exist_ok=True)

    with open(log_output, "w") as f:
        json.dump(json_result, f, indent=4)
    logger.info(f"Validation log saved at: {log_output}")

    # --- Fail Fast if Any Validation Fails ---
    if not validation_results.success:
        logger.error("❌ Data validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    app()
