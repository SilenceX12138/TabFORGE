# Getting Started

TabFORGE provides sklearn-style estimators for synthetic table generation, prediction, imputation, feature embeddings, clustering, and anomaly detection. This page takes you from installation to a small CPU example.

## Install

TabFORGE requires Python 3.10 or later. The PyPI distribution is [tabforge-ml](https://pypi.org/project/tabforge-ml/); Python imports and the command-line entry point use `tabforge`. Choose the core package or include the optional notebook dependencies:

<div class="code-tabs">
<div role="tablist" aria-label="Installation options"><button type="button" role="tab" id="install-core-tab" aria-controls="install-core" aria-selected="true">Core package</button><button type="button" role="tab" id="install-tutorials-tab" aria-controls="install-tutorials" aria-selected="false" tabindex="-1">With tutorials</button></div>
<div role="tabpanel" id="install-core" aria-labelledby="install-core-tab">

```bash
pip install tabforge-ml
```

</div>
<div role="tabpanel" id="install-tutorials" aria-labelledby="install-tutorials-tab" hidden>

```bash
pip install 'tabforge-ml[tutorials]'
```

</div>
</div>

For an editable source installation:

```bash
git clone https://github.com/SilenceX12138/TabFORGE.git
cd TabFORGE
pip install -e '.[tutorials]'
```

Optional online logging is available with `pip install 'tabforge-ml[wandb]'`. See the [runtime settings](configuration.md#tabforgeruntimeconfig) for device selection and logging configuration.

<aside class="doc-note" aria-label="Model requirements"><strong>Model requirements</strong><p>The Structure-aware Feature Encoder uses pretrained weights by default. Its first run may download model files. The example below uses <code>model_path="mock"</code> for a small CPU demonstration; use pretrained embeddings for real modelling.</p></aside>

## Generate a small synthetic table

<aside class="doc-note" aria-label="Example settings"><strong>Example settings</strong><p>This example uses CPU execution, mock embeddings, and a short training budget. It demonstrates fitting and generation without downloading model weights. The pretrained workflow below shows the settings for real data.</p></aside>

```python
import pandas as pd

from tabforge import (
    TabFORGEEmbeddingConfig,
    TabFORGEGenerator,
    TabFORGERuntimeConfig,
    TabFORGETrainingConfig,
)

X = pd.DataFrame(
    {
        "age": [23, 31, 45, 52, 38, 29, 61, 47],
        "region": ["north", "south", "north", "west", "east", "south", "west", "east"],
        "balance": [120.0, 80.0, 300.0, 240.0, 160.0, 90.0, 410.0, 270.0],
    }
)

model = TabFORGEGenerator(
    task="unsupervision",
    categorical_features=["region"],
    embedding_config=TabFORGEEmbeddingConfig(model_path="mock", n_folds=0),
    training_config=TabFORGETrainingConfig(max_steps=20, batch_size=8),
    runtime_config=TabFORGERuntimeConfig(device="cpu"),
    random_state=7,
)
model.fit(X, checkpoint=None)
synthetic = model.generate(n_samples=5, random_state=7)
print(synthetic)
```

The result is a five-row DataFrame with the original `age`, `region`, and `balance` columns. These demonstration settings verify the API; use pretrained components and a suitable training budget for useful synthetic data.

`task="unsupervision"` fits the generator to feature rows alone. For supervised generation, set `task` to `"classification"` or `"regression"`, pass `y` to `fit`, and `generate` returns an `(X, y)` pair.

## Use the pretrained Structure-aware Feature Encoder

For real use, keep `model_path="auto"` (the default), provide enough data and training steps for your task, and select a device suitable for your workload. The default component checkpoint is loaded during fitting; pass `checkpoint=None` only when you intend to initialize components from scratch.

Prepare `customer-table.csv` with columns `age`, `region`, and `balance` and at least 20 rows. Use your actual training table for this workflow; the eight-row mock table above only demonstrates the API.

```python
X_train = pd.read_csv("customer-table.csv")

model = TabFORGEGenerator(
    task="unsupervision",
    categorical_features=["region"],
    embedding_config=TabFORGEEmbeddingConfig(n_folds=2),
    training_config=TabFORGETrainingConfig(max_steps=10_000, batch_size=512),
    runtime_config=TabFORGERuntimeConfig(device="auto"),
    random_state=7,
)
model.fit(X_train)  # Uses the default pretrained components and encoder.
model.save_checkpoint("runs/customer-table")
```

The saved fitted checkpoint can be restored for generation or inference. See [Checkpoint lifecycle](public_estimator_api_lifecycle.md) for the distinction between fitted checkpoints and reusable backbone checkpoints.

```python
from tabforge import restore_from_checkpoint

restored = restore_from_checkpoint("runs/customer-table", device="auto")
synthetic = restored.generate(100, random_state=7)
```

## Next steps

- [User guide](user_guide.md) for task selection and common workflows.
- [API reference](public_estimator_api.md) for estimator and configuration details.
- [Tutorial gallery](tutorials.md) for nine runnable notebooks.
- [Command-line interface](cli.md) for CSV-based fitting and inference.
