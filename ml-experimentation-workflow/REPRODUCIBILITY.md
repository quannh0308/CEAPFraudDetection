# Reproducibility Checklist

Use this checklist before promoting an experiment to production or sharing results with the team. Every experiment should be fully reproducible from its tracked metadata.

## Data Version Tracking

- [ ] Dataset S3 path is recorded (e.g. `s3://fraud-detection-data/train/v1.2.3/`)
- [ ] Dataset version or snapshot ID is logged to ExperimentTracker via `dataset_version`
- [ ] Train/validation/test split ratios are documented
- [ ] Any data preprocessing steps are captured in the notebook or logged as parameters
- [ ] Data filtering criteria (date ranges, feature subsets) are recorded

```python
experiment_id = tracker.start_experiment(
    experiment_name="fraud-detection-v2",
    algorithm="xgboost",
    dataset_version="v1.2.3",  # Always set this
)
tracker.log_parameters(experiment_id, {
    "data_s3_path": "s3://fraud-detection-data/train/v1.2.3/",
    "train_split": 0.7,
    "validation_split": 0.15,
    "test_split": 0.15,
})
```

## Code Version Tracking

- [ ] Git commit hash is logged to ExperimentTracker via `code_version`
- [ ] Git branch name is recorded
- [ ] Working directory is clean (no uncommitted changes) — check `git_dirty` flag
- [ ] Notebook is saved and committed before running the experiment
- [ ] The `template.ipynb` auto-logging captures this automatically when used

```python
import subprocess

commit = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
).decode().strip()

branch = subprocess.check_output(
    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL
).decode().strip()

tracker.log_parameters(experiment_id, {
    "code_git_commit": commit,
    "code_git_branch": branch,
})
```

## Hyperparameter Tracking

- [ ] All hyperparameters are logged via `tracker.log_parameters()`
- [ ] Hyperparameter search ranges are documented (for tuning experiments)
- [ ] The search strategy is recorded (grid, random, Bayesian)
- [ ] Best hyperparameters from tuning are explicitly logged
- [ ] Production-format parameters (objective, num_round, max_depth, eta, subsample, colsample_bytree) are validated before promotion

```python
tracker.log_parameters(experiment_id, {
    "max_depth": 7,
    "eta": 0.15,
    "num_round": 150,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "search_strategy": "grid_search",
    "search_n_combinations": 48,
})
```

## Random Seed Management

- [ ] A fixed random seed is set for Python's `random` module
- [ ] A fixed random seed is set for NumPy
- [ ] A fixed random seed is set for the ML framework (XGBoost, LightGBM, scikit-learn)
- [ ] The seed value is logged as a hyperparameter
- [ ] Data shuffling uses a fixed seed via `train_test_split(random_state=...)`

```python
import random
import numpy as np

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Log the seed
tracker.log_parameters(experiment_id, {"random_seed": RANDOM_SEED})

# Use in train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# Use in model training
from xgboost import XGBClassifier
model = XGBClassifier(random_state=RANDOM_SEED, max_depth=7, learning_rate=0.15)
```

## Metrics and Artifacts

- [ ] All evaluation metrics are logged via `tracker.log_metrics()`
- [ ] Model artifacts are uploaded to S3 via `tracker.log_artifacts()`
- [ ] Visualization files (confusion matrix, ROC curve, PR curve) are saved and logged
- [ ] Baseline comparison results are recorded

## Environment

- [ ] Python version is logged (captured automatically by `template.ipynb`)
- [ ] Key package versions are logged (boto3, sagemaker, pandas, numpy, scikit-learn, xgboost, lightgbm)
- [ ] SageMaker instance type is documented for training jobs

## Quick Verification

To verify an experiment is reproducible:

1. Check out the recorded git commit: `git checkout <commit_hash>`
2. Install the recorded dependency versions
3. Load the recorded dataset version from S3
4. Set the recorded random seed
5. Run with the recorded hyperparameters
6. Compare metrics — they should match within floating-point tolerance
