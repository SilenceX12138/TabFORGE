# Repository Instructions

## Coding style

- Before editing or creating code, read and follow `.agent/style.md`.
- Treat `.agent/style.md` as the authoritative coding-style guide for this repository.
- Apply its conventions to all newly written or substantially modified code.
- When existing code conflicts with `.agent/style.md`, follow the style guide for the touched scope unless doing so would require unrelated refactoring.
- For style validation, follow `.github/workflows/style_check.yaml` and run `bash scripts/utils/style_check.sh` through the required Conda environment. Use this Pylint and Flake8 workflow as the repository style check.

## Runtime environment

Use the Conda environment `tfm` for all Python execution, tests, evaluation, and experiments.

Run commands through:

`conda run -n tfm --no-capture-output <command>`

Do not use another Python/Conda environment unless explicitly requested.

## Experiment execution

When running new experiments with Weights & Biases:

- Use online W&B logging by default.
- Store local W&B logging files under `./logs/wandb`.
- Store experiment handoff reports under `./logs/report`.

Do not use offline W&B mode unless explicitly requested or required by the environment.

## Research experiment logging

When you complete an important experiment that answers a research question,
tests a hypothesis, or provides meaningful evidence for a research decision,
invoke `$log-experiment-notion` before finishing the task.

Use `.agent/notion.toml` as the authoritative repository-specific Notion
configuration.

Do not log smoke tests, debugging runs, failed runs, or routine sanity checks.
