"""Smoke-test the deployed default-payment-predictor InferenceService.

Not a pytest unit test: this calls the *live* KServe endpoint over the
Istio ingress gateway, so it needs cluster access (kubectl) and a running
InferenceService. Use it after a promote/deploy to confirm the model
actually serves predictions, not just that the Kubernetes objects exist.

Usage:
    uv run scripts/test_inference_service.py
    uv run scripts/test_inference_service.py --ingress-ip 20.1.2.3
    uv run scripts/test_inference_service.py --payload my_customer.json

KServe's sklearn runtime quirk (see gitops/kserve-inference.yaml): the
request body must be a dict of column-name -> single-element-list, not the
more common list-of-row-dicts or row-array shapes. Sending those instead
fails with confusing 500s ("Specifying the columns using strings is only
supported for dataframes" / "If using all scalar values, you must pass an
index") rather than a clear "wrong format" error.
"""

import json
from pathlib import Path
import subprocess
import sys
import typing
import urllib.error
import urllib.request

from loguru import logger
import typer

app = typer.Typer(add_completion=False)

DEFAULT_MODEL_NAME = "default-payment-predictor"
DEFAULT_NAMESPACE = "ml-credit-default"
DEFAULT_HOST = f"{DEFAULT_MODEL_NAME}-predictor.{DEFAULT_NAMESPACE}.example.com"

# The 28 fields the model's ColumnTransformer expects, in training column
# order (see feature_metadata.json logged by mlflow_logger.py). Two named
# scenarios covering opposite ends of the risk spectrum, not exhaustive
# coverage -- this is a smoke test, not a model-quality check.
SCENARIOS: dict[str, dict[str, typing.Any]] = {
    "high_risk": {
        "limit_bal": 20000.0,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 24,
        "pay_0": 2,
        "pay_2": 2,
        "pay_3": -1,
        "pay_4": -1,
        "pay_5": -2,
        "pay_6": -2,
        "bill_amt1": 3913.0,
        "bill_amt2": 3102.0,
        "bill_amt3": 689.0,
        "bill_amt4": 0.0,
        "bill_amt5": 0.0,
        "bill_amt6": 0.0,
        "pay_amt1": 0.0,
        "pay_amt2": 689.0,
        "pay_amt3": 0.0,
        "pay_amt4": 0.0,
        "pay_amt5": 0.0,
        "pay_amt6": 0.0,
        "customer_id": 1,
        "age_bin": "18_25",
        "bill_trend": -3913.0,
        "pay_ratio": 0.4715,
        "utilization": 0.0,
    },
    "low_risk": {
        "limit_bal": 500000.0,
        "sex": 1,
        "education": 1,
        "marriage": 2,
        "age": 40,
        "pay_0": 0,
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "bill_amt1": 10000.0,
        "bill_amt2": 10000.0,
        "bill_amt3": 10000.0,
        "bill_amt4": 10000.0,
        "bill_amt5": 10000.0,
        "bill_amt6": 10000.0,
        "pay_amt1": 10000.0,
        "pay_amt2": 10000.0,
        "pay_amt3": 10000.0,
        "pay_amt4": 10000.0,
        "pay_amt5": 10000.0,
        "pay_amt6": 10000.0,
        "customer_id": 2,
        "age_bin": "36_50",
        "bill_trend": 0.0,
        "pay_ratio": 1.0,
        "utilization": 0.02,
    },
}


def get_ingress_ip() -> str:
    """Look up the Istio ingress gateway's external IP via kubectl.

    Not cached/hardcoded anywhere: this environment's AKS clusters get
    rebuilt often enough that the IP changes across sessions.
    """
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "svc",
            "-n",
            "istio-system",
            "istio-ingressgateway",
            "-o",
            "jsonpath={.status.loadBalancer.ingress[0].ip}",
        ],
        capture_output=True,
        text=True,
    )
    ip = result.stdout.strip()
    if result.returncode != 0 or not ip:
        logger.error(f"Could not resolve istio-ingressgateway IP: {result.stderr.strip()}")
        raise typer.Exit(code=1)
    return ip


def to_kserve_payload(row: dict[str, typing.Any]) -> dict:
    """Wrap each field as a single-element list -- the "dict of column
    lists" shape KServe's sklearn runtime actually accepts (see module
    docstring)."""
    return {"instances": [{k: [v] for k, v in row.items()}]}


def call(url: str, host: str, body: dict | None = None) -> tuple[int, dict | str]:
    """POST (or GET, if body is None) and return (status_code, parsed_json_or_text)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Host": host, "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        status = e.code
    except urllib.error.URLError as e:
        logger.error(f"Request to {url} failed: {e.reason}")
        raise typer.Exit(code=1) from e

    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


@app.command()
def main(
    ingress_ip: str = typer.Option(
        None, help="Istio ingress gateway IP. Auto-discovered via kubectl if omitted."
    ),
    host: str = typer.Option(DEFAULT_HOST, help="Host header (KServe routes on this, not DNS)."),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="InferenceService name."),
    payload: Path = typer.Option(
        None,
        help="Path to a JSON file with raw field:value pairs to test instead of the "
        "built-in high_risk/low_risk scenarios (values get auto-wrapped for KServe).",
    ),
):
    """Smoke-test the live default-payment-predictor InferenceService."""
    ip = ingress_ip or get_ingress_ip()
    base_url = f"http://{ip}"
    logger.info(f"Testing {model_name} via {base_url} (Host: {host})")

    logger.info("Checking readiness...")
    status, body = call(f"{base_url}/v1/models/{model_name}", host)
    if status != 200 or not isinstance(body, dict) or not body.get("ready"):
        logger.error(f"Model not ready: HTTP {status} {body}")
        raise typer.Exit(code=1)
    logger.success(f"Ready: {body}")

    scenarios = {"custom": json.loads(payload.read_text())} if payload else SCENARIOS

    failures = 0
    for name, row in scenarios.items():
        logger.info(f"--- {name} ---")
        status, body = call(
            f"{base_url}/v1/models/{model_name}:predict", host, to_kserve_payload(row)
        )
        if status == 200 and isinstance(body, dict) and "predictions" in body:
            logger.success(f"HTTP {status}: {body}")
        else:
            logger.error(f"HTTP {status}: {body}")
            failures += 1

    if failures:
        logger.error(f"{failures}/{len(scenarios)} scenario(s) failed")
        sys.exit(1)
    logger.success(f"All {len(scenarios)} scenario(s) passed")


if __name__ == "__main__":
    app()
