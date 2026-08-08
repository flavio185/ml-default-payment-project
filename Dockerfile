# Per-project image extending the shared platform base.
# Heavy dependencies (feast, mlflow, ray, pandas, sklearn) are cached in the base image.
# gh CLI and shared platform scripts (e.g. promote_model.py, used by the
# promote-model ClusterWorkflowTemplate) come from the base image too.
FROM docker.io/flavio185/ml-platform-base:latest

WORKDIR /app

# Install project-specific dependencies (cached layer, no project source needed yet)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy project code and install the project itself
COPY . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run"]
