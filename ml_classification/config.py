from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

S3_BUCKET = "datamasters2025"
VALIDATION_REPORTS_DIR = PROJ_ROOT / "data" / "validation_reports"
DRIFT_REPORTS_DIR = PROJ_ROOT / "data" / "drift_reports"

# Feast
FEAST_REPO_PATH = PROJ_ROOT / "feast"
