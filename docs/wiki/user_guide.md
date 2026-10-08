# User Guide

This guide maps a table-learning task to a TabFORGE estimator and summarizes the fitted-model workflow. Every estimator follows the familiar `fit` and task-specific inference pattern; the detailed signatures live in the [API reference](public_estimator_api.md).

<nav class="guide-index" aria-label="User guide topics">
<ul><li><a href="#generation">Generative modelling</a><ul><li><a href="#generation">Synthetic tables</a> · <a href="#imputation">Missing-value imputation</a></li></ul></li><li><a href="#prediction">Supervised learning</a><ul><li><a href="#prediction">Classification and regression</a> · <a href="#fit-infer-and-save">Fitting and checkpoints</a></li></ul></li><li><a href="#embedding">Embedding</a><ul><li><a href="#embedding">Feature embeddings</a> · <a href="#clustering">Clustering</a> · <a href="#anomaly-detection">Anomaly detection</a></li></ul></li><li><a href="#configuration">Configuration and runtime</a><ul><li><a href="configuration.md">Configuration reference</a> · <a href="cli.md">Command-line interface</a></li></ul></li></ul>
</nav>

## Choose an estimator

| Task | Estimator | Main operation |
| --- | --- | --- |
| Generate synthetic rows | `TabFORGEGenerator` | `fit(X, y)` then `generate(n_samples)` |
| Classification | `TabFORGEClassifier` | `fit(X, y)`, `predict(X)` or `predict_proba(X)` |
| Regression | `TabFORGERegressor` | `fit(X, y)`, `predict(X)` or `predict_distribution(X)` |
| Feature embeddings | `TabFORGEEmbedder` | `fit(X, y)` then `transform(X)` |
| Clustering | `TabFORGEClusterer` | `fit_predict(X)` or `predict(X)` |
| Anomaly detection | `TabFORGEAnomalyDetector` | `fit(X)`, `decision_function(X)` or `predict(X)` |
| Missing-value imputation | `TabFORGEImputer` | `fit(X)` then `transform(X)` |

All estimators accept DataFrames or NumPy arrays. DataFrames preserve their column names in generated and imputed outputs. Categorical columns can be inferred from DataFrame dtypes or specified by name or integer position with `categorical_features`. For array inputs, use integer positions.

## Fit, infer, and save

The snippets below assume aligned `X_train`, `X_test`, `X_valid`, and targets have been prepared. `y_train` and `y_class_train` contain class labels; `y_regression_train` contains numeric targets. `X_incomplete` has the same columns as `X_train` with missing cells. The [tutorials](tutorials.md) include data preparation and complete runnable workflows.

Fitting creates the feature processor, embedding context, and task-specific model state. In most workflows, use the official pretrained components with the default `checkpoint="pretrained"` argument. For component initialization from scratch, pass `checkpoint=None` explicitly. A validation split can be supplied as `validation_data`; classification and regression expect an `(X_valid, y_valid)` pair.

```python
from tabforge import TabFORGEClassifier

classifier = TabFORGEClassifier(categorical_features=["region"], random_state=11)
classifier.fit(X_train, y_train, validation_data=(X_valid, y_valid))
probabilities = classifier.predict_proba(X_test)
labels = classifier.predict(X_test)
classifier.save_checkpoint("models/classifier")
```

Restore a fitted estimator with `restore_from_checkpoint`. A fitted checkpoint contains the data-dependent state needed to use that trained estimator. `save_backbone_checkpoint` stores reusable Denoising-aligned Latent Decoder and diffusion weights without fitted dataset state; consult [checkpoint lifecycle](public_estimator_api_lifecycle.md) before choosing a format.

## Prediction

For classification, labels in `y` may be categorical; `classes_` gives the probability-column order. For regression, `predict_distribution(X)` returns sampled target values, while `predict` summarizes those samples. `predict_std` and `predict_interval` expose uncertainty summaries. Configure `n_prediction_samples` on either estimator to control the number of target trajectories.

```python
from tabforge import TabFORGEClassifier, TabFORGERegressor

classifier = TabFORGEClassifier(n_prediction_samples=32, random_state=7)
classifier.fit(X_train, y_class_train)
probabilities = classifier.predict_proba(X_test)
labels = classifier.predict(X_test)

regressor = TabFORGERegressor(n_prediction_samples=32, random_state=7)
regressor.fit(X_train, y_regression_train)
predictions = regressor.predict(X_test)
intervals = regressor.predict_interval(X_test, alpha=0.05)
```

## Generation

Use `task="unsupervision"` for a feature-only table. Use `task="classification"` or `"regression"` and provide aligned targets to model a joint feature-target table. Generation starts from latent embeddings with random noise, denoises them, and decodes synthetic table values.

```python
from tabforge import TabFORGEGenerator

generator = TabFORGEGenerator(task="unsupervision", random_state=7).fit(X_train)
synthetic = generator.generate(1_000, random_state=7)
generator.save_checkpoint("models/generator")
```

<span id="representations" aria-hidden="true"></span>

## Embedding

`TabFORGEEmbedder.transform` returns per-feature embeddings by default, shaped `(rows, tokens, embedding_dimension)`. Request `pooling="mean"`, `"max"`, or `"flatten"` for one embedding per row. `source="encoder"` selects the Structure-aware Feature Encoder; `source="decoder"` selects the Denoising-aligned Latent Decoder. `layer` selects an intermediate layer of the chosen component.

```python
from tabforge import TabFORGEEmbedder

embedder = TabFORGEEmbedder(task="unsupervision", random_state=7).fit(X_train)
token_grid = embedder.transform(X_test)
row_features = embedder.transform(
    X_test, source="encoder", layer="middle", pooling="mean"
)
```

## Imputation

`TabFORGEImputer.transform(X)` regenerates missing cells. Supply a boolean `mask` to choose cells explicitly; `True` marks cells to regenerate. Observed cells are retained. Set `n_imputation_samples` to aggregate multiple conditional trajectories.

```python
from tabforge import TabFORGEImputer

imputer = TabFORGEImputer(n_imputation_samples=8, random_state=7).fit(X_train)
completed = imputer.transform(X_incomplete)
```

## Clustering

`TabFORGEClusterer` applies KMeans to pooled TabFORGE embeddings. Use `fit_predict(X)` to fit the fitted embedding workflow and obtain training cluster labels, then `predict(X_new)` for new rows.

```python
from tabforge import TabFORGEClusterer

clusterer = TabFORGEClusterer(n_clusters=4, random_state=7).fit(X_train)
cluster_labels = clusterer.predict(X_test)
```

## Anomaly detection

`TabFORGEAnomalyDetector` applies Isolation Forest to pooled TabFORGE embeddings. `decision_function(X)` returns threshold-relative scores, and `predict(X)` returns `+1` for inliers and `-1` for anomalies.

```python
from tabforge import TabFORGEAnomalyDetector

detector = TabFORGEAnomalyDetector(random_state=7).fit(X_train)
scores = detector.score_samples(X_test)
outlier_labels = detector.predict(X_test)
```

## Configuration

Pass immutable config objects or mappings to an estimator. The main groups are:

- `TabFORGEEmbeddingConfig`: encoder backend, folds, layer, and extraction batch size.
- `TabFORGEArchitectureConfig`: Denoising-aligned Latent Decoder and Score-based Diffusion Transformer depth, attention heads, and width.
- `TabFORGEDiffusionConfig`: noise schedule and sampling steps.
- `TabFORGETrainingConfig`: optimization budget, batch size, phase shares, and validation settings.
- `TabFORGERuntimeConfig`: device, distributed strategy, deterministic execution, and optional W&B logging.

Configuration classes are exported from `tabforge`. Their defaults and validation rules are documented in [configuration](configuration.md). For online experiment logging, configure `runtime_config=TabFORGERuntimeConfig(log_wandb=True, wandb_dir="./logs/wandb")` after authenticating W&B.

## More detail

- [Estimator lifecycle and checkpoints](public_estimator_api_lifecycle.md)
- [Estimator classes](public_estimator_api_estimators.md)
- [Configuration reference](configuration.md)
- [Feature processing](tabular_feature_processing.md)
- [Tutorial gallery](tutorials.md)
