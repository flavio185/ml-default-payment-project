# Per-project image extending the shared platform base.
# Heavy dependencies (feast, mlflow, ray, pandas, sklearn) are cached in the base image.
FROM ghcr.io/datamaster2026/ml-platform-base:latest

WORKDIR /app

# Install project-specific dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy project code
COPY . .

ENTRYPOINT ["uv", "run"]
