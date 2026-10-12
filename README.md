# Student Dropout Prediction with MLflow

## How to run

Requires [uv](https://docs.astral.sh/uv/): the dataset is downloaded automatically.

```bash
# 1. Install dependencies
uv sync

# 2. Run one training (default values)
uv run train.py

# 3. Run with custom hyperparameters
uv run train.py --C 0.1 --class-weight balanced --max-iter 500 --seed 7

# 4. Run all 8 experiments
bash runner.sh

# 5. Open the MLflow UI
uv run mlflow ui
```

In the UI, open the experiment `student-dropout-logreg-hparams`, select the runs and click **Compare**.

Arguments (all optional): `--C`, `--class-weight` (`none` or `balanced`), `--max-iter`, `--seed`, `--experiment-name`.
