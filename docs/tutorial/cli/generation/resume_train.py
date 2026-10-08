import os
from pathlib import Path

import torch
from tabcamel.data.dataset import TabularDataset

from tabforge import TabFORGEGenerator, TabFORGERuntimeConfig, TabFORGETrainingConfig

SOURCE_CHECKPOINT_PATH = Path("./logs/checkpoints/generator")
RESUMED_CHECKPOINT_PATH = Path("./logs/checkpoints/generator-resumed")
SYNTHETIC_DATA_PATH = Path("./logs/synthetic/credit-g-resumed.csv")


def main() -> None:
    """Continue training from saved weights and export a new synthetic table."""
    dataset = TabularDataset("credit-g", task_type="classification")
    max_rows = os.environ.get("TABFORGE_TUTORIAL_MAX_ROWS")
    data = dataset.data_df if max_rows is None else dataset.data_df.iloc[:int(max_rows)]
    model = TabFORGEGenerator(
        task="unsupervision",
        training_config=TabFORGETrainingConfig(
            max_steps=int(os.environ.get("TABFORGE_TUTORIAL_MAX_STEPS", "1000")),
            batch_size=512,
        ),
        runtime_config=TabFORGERuntimeConfig(
            device="cuda",
            strategy="ddp",
            log_wandb=True,
            wandb_project="tabforge",
            # wandb_entity="YOUR_ENTITY",
            wandb_dir="./logs/wandb",
        ),
        random_state=7,
    )

    model.fit(data, checkpoint=SOURCE_CHECKPOINT_PATH)
    model.save_checkpoint(RESUMED_CHECKPOINT_PATH)
    synthetic = model.generate(len(data), random_state=8)
    if _is_primary_process():
        SYNTHETIC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(SYNTHETIC_DATA_PATH, index=False)


def _is_primary_process() -> bool:
    """Return whether this process owns user-facing output files."""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


if __name__ == "__main__":
    main()
