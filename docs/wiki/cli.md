# Command-line interface

Install `tabforge-ml` using the [installation guide](getting_started.md#install), then run `tabforge` to fit estimators from CSV files and use saved fitted checkpoints.

## Fit and save a model

Prepare `train.csv` with a `label` target and feature columns including `region` and `plan`. Prepare `test.csv` with the same feature columns and no target. Paths below are relative to your working directory. Fit a supervised estimator by naming the target column:

```bash
tabforge fit \
  --input train.csv \
  --checkpoint models/classifier \
  --estimator classifier \
  --target label \
  --categorical-features region,plan \
  --max-steps 10000 \
  --device auto \
  --random-state 7
```

The available estimator values are `generator`, `classifier`, `regressor`, `embedder`, and `imputer`. `generator` and `embedder` also require `--task` with `classification`, `regression`, or `unsupervision`. Classifiers and regressors require `--target`; imputer fitting does not accept one.

Use `--model-path mock` for CPU development runs without fetching the Structure-aware Feature Encoder. The CLI accepts `--max-steps`, `--batch-size`, `--embedding-dimension`, `--device`, `--strategy`, `--gradient-accumulation`, and `--deterministic` for common training/runtime choices. Install `tabforge-ml[wandb]`, run `wandb login`, and add `--log-wandb` to enable online W&B logging; `--wandb-dir` defaults to `./logs/wandb`.

The mock backend also skips the default pretrained TabFORGE backbone because its small embedding architecture differs from the official one. See the [Python quickstart](getting_started.md#generate-a-small-synthetic-table) for a complete offline installation check with explicit scratch initialization.

## Run a saved estimator

For feature-only examples, prepare `data.csv` with complete feature rows and `incomplete.csv` with the same columns and some missing cells. Fit each checkpoint before its inference command:

```bash
tabforge fit \
  --input data.csv \
  --checkpoint models/generator \
  --estimator generator \
  --task unsupervision

tabforge fit \
  --input data.csv \
  --checkpoint models/embedder \
  --estimator embedder \
  --task unsupervision

tabforge fit \
  --input data.csv \
  --checkpoint models/imputer \
  --estimator imputer
```

Generate rows from the fitted generator checkpoint:

```bash
tabforge generate \
  --checkpoint models/generator \
  --output synthetic.csv \
  --n-samples 1000 \
  --random-state 7
```

Predict with a fitted classifier or regressor:

```bash
tabforge predict \
  --checkpoint models/classifier \
  --input test.csv \
  --output predictions.csv
```

The remaining inference commands are:

```bash
tabforge embed \
  --checkpoint models/embedder \
  --input data.csv \
  --output embeddings.npy

tabforge impute \
  --checkpoint models/imputer \
  --input incomplete.csv \
  --output imputed.csv
```

`generate` writes feature columns to CSV and adds a `target` column for supervised generation. Classifier `predict` writes `proba_<class>` columns; regressor `predict` writes a `prediction` column. `embed` writes a NumPy token grid, and `impute` writes the completed feature table.

Checkpoint transfer and detokeniser policies are exposed by the Python API. Clustering and anomaly detection also use the Python estimators; see the [user guide](user_guide.md).

Use `tabforge --help` or `tabforge fit --help` to inspect current command options. The CLI's complete task interfaces are also available in Python; see the [user guide](user_guide.md) and [API reference](public_estimator_api.md).
