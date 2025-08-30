#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ml-default-payment-project
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Clean artifacts (datasets, models, logs)
.PHONY: clean_artifacts
clean_artifacts:
	rm -rf data/bronze/* data/silver/* data/gold/* models/* logs/*

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format



## Run tests
.PHONY: test
test:
	python -m pytest --cov=data_processing --cov=ml_classification --cov-report=term-missing tests/

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"


#################################################################################
# PIPELINE RULES                                                                #
#################################################################################

## Ingest Bronze
.PHONY: bronze
bronze: 
	$(PYTHON_INTERPRETER) data_processing/bronze/ingest_bronze.py

## Clean Silver
.PHONY: silver
silver: 
	$(PYTHON_INTERPRETER) data_processing/silver/clean_data.py

## Validate Silver
.PHONY: validate
validate: 
	$(PYTHON_INTERPRETER) data_processing/silver/validate_data.py

## Create Gold Features
.PHONY: gold
gold: 
	$(PYTHON_INTERPRETER) data_processing/gold/build_features.py

## Run full pipeline: Bronze → Silver → Validate → Gold
.PHONY: pipeline
pipeline: requirements bronze silver gold
	@echo ">>> Full pipeline executed successfully!"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make train
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) ml_classification/modeling/train.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
