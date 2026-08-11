"""Tests for drift_check.py's pure logic (PSI computation, S3 URI parsing).

resolve_champion_gold_uri / read_parquet_version / run_drift_check hit MLflow
and S3 directly and aren't covered here.
"""

import numpy as np
import pandas as pd
import pytest

from ml_classification.pipelines.drift_check import _parse_s3_uri, compute_psi


def test_psi_identical_distributions_is_near_zero():
    baseline = pd.Series(np.linspace(0, 100, 500))
    current = baseline.copy()

    assert compute_psi(baseline, current) == pytest.approx(0.0, abs=1e-6)


def test_psi_shifted_distribution_is_flagged_critical():
    rng = np.random.default_rng(42)
    baseline = pd.Series(rng.normal(loc=0, scale=1, size=1000))
    current = pd.Series(rng.normal(loc=5, scale=1, size=1000))

    assert compute_psi(baseline, current) >= 0.2


def test_psi_categorical_column():
    baseline = pd.Series(["a", "a", "a", "b", "b", "c"])
    current = pd.Series(["a", "b", "b", "b", "c", "c"])

    psi = compute_psi(baseline, current)

    assert psi > 0.0


def test_psi_categorical_new_category_in_current():
    baseline = pd.Series(["a", "a", "b", "b"])
    current = pd.Series(["a", "b", "c", "c"])

    # Must not raise even though "c" is absent from the baseline categories.
    psi = compute_psi(baseline, current)

    assert psi > 0.0


def test_psi_near_constant_baseline_falls_back_to_single_split():
    # A near-constant column degenerates quantile binning to <3 edges --
    # compute_psi must fall back rather than raising from np.unique.
    baseline = pd.Series([1] * 99 + [2])
    current = pd.Series([1] * 90 + [2] * 10)

    psi = compute_psi(baseline, current)

    assert psi >= 0.0


def test_psi_ignores_nan_values():
    baseline = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    current = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])

    assert compute_psi(baseline, current) == pytest.approx(0.0, abs=1e-6)


def test_parse_s3_uri():
    bucket, key = _parse_s3_uri("s3://my-bucket/gold/credit_card_default_features.parquet")

    assert bucket == "my-bucket"
    assert key == "gold/credit_card_default_features.parquet"


def test_parse_s3_uri_nested_key():
    bucket, key = _parse_s3_uri("s3://my-bucket/a/b/c/file.parquet")

    assert bucket == "my-bucket"
    assert key == "a/b/c/file.parquet"
