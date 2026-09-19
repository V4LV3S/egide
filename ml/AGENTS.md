# Python and Machine Learning Development Guide

## Tooling and Environment

- Use `uv` for all Python dependency management, virtual environments, and script execution.
- Run Python scripts with `uv run`, for example: `uv run python scripts/train.py`.
- Add dependencies with `uv add <package>` and development dependencies with `uv add --dev <package>`.
- Do not use `pip`, `poetry`, `conda`, or direct `python` execution outside `uv run`.
- Keep dependency definitions and lock files up to date when dependencies change.

## Code Standards

- Write all production code with clear comments explaining non-obvious logic, assumptions, and decisions.
- Add docstrings to modules, public functions, classes, and complex data-processing routines.
- Use type hints for function parameters, return values, and important variables.
- Prefer small, focused, testable functions and avoid hidden global state.
- Follow PEP 8 and use descriptive, domain-specific names.
- Never hard-code secrets, credentials, absolute local paths, or environment-specific configuration.

## Machine Learning Practices

- Make experiments reproducible: set and document random seeds, package versions, data versions, and configuration values.
- Separate data ingestion, preprocessing, training, evaluation, and inference into clear modules or scripts.
- Validate input data and explicitly handle missing values, schema changes, and data leakage risks.
- Split data correctly for training, validation, and testing; never use test data for model selection.
- Log metrics, parameters, artifacts, and model versions for every training run.
- Save model artifacts with metadata describing the model, feature schema, training data, and evaluation results.
- Document computational requirements and expected runtime for resource-intensive workflows.

## Testing and Quality

- Write or update tests for every behavior change, including data transformations and model-serving interfaces.
- Run tests through `uv`, for example: `uv run pytest`.
- Run formatting, linting, and type checks through `uv` before completing changes.
- Keep tests deterministic and avoid requiring network access, large datasets, or GPUs unless explicitly marked.

## Documentation

- Document setup, execution, configuration, inputs, outputs, and expected results for every script.
- Include runnable `uv run` commands in project documentation and examples.
- Comment assumptions, limitations, and potential failure modes in ML pipelines and models.
