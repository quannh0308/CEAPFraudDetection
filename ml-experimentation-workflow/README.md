# ML Experimentation Workflow

A Python toolkit for data scientists to experiment with and optimize fraud detection models in SageMaker Studio. It bridges exploratory data science and the [production fraud-detection-ml-pipeline](../README.md), providing structured experiment tracking, hyperparameter tuning, algorithm comparison, feature engineering, model evaluation, and a promotion path to production.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SageMaker Studio                             │
│                                                                     │
│  ┌────────────────────┐       ┌─────────────────────┐               │
│  │ experiment_tracking │       │ hyperparameter_tuning│              │
│  │ (SageMaker          │       │ (Grid / Random /     │              │
│  │  Experiments API)   │       │  Bayesian via AMT)   │              │
│  └────────┬───────────┘       └──────────┬──────────┘               │
│           │                              │                          │
│  ┌────────┴───────────┐       ┌──────────┴──────────┐               │
│  │ feature_engineering │       │ algorithm_comparison │              │
│  │ (Time/Amount/       │       │ (XGBoost, LightGBM,  │              │
│  │  Interaction feats, │       │  Random Forest,       │              │
│  │  Selection: RFE,    │       │  Neural Network)      │              │
│  │  Univariate, Imp.)  │       └──────────┬──────────┘               │
│  └────────────────────┘                  │                          │
│                                ┌─────────┴──────────┐               │
│                                │  model_evaluation   │               │
│                                │  (Metrics, ROC/PR,  │               │
│                                │   Confusion Matrix, │               │
│                                │   Threshold: ≥0.90) │               │
│                                └─────────┬──────────┘               │
│                                          │                          │
│                                ┌─────────┴──────────┐               │
│                                │production_integration│              │
│                                │  (Validate → Backup │               │
│                                │   → Write Params →  │               │
│                                │   S3 Config → Trigger│              │
│                                │   Pipeline)          │              │
│                                └─────────┬──────────┘               │
└──────────────────────────────────────────┼──────────────────────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     ▼                     ▼                     ▼
            ┌──────────────┐      ┌──────────────┐     ┌──────────────┐
            │  SSM Param   │      │  S3 Config   │     │Step Functions│
            │  Store       │      │  Bucket      │     │  (Pipeline)  │
            └──────────────┘      └──────────────┘     └──────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `experiment_tracking.py` | `ExperimentTracker` — wraps SageMaker Experiments for logging parameters, metrics, and artifacts |
| `hyperparameter_tuning.py` | `HyperparameterTuner` — grid search, random search, and SageMaker Automatic Model Tuning (Bayesian) |
| `algorithm_comparison.py` | `AlgorithmComparator` — compares XGBoost, LightGBM, Random Forest, and Neural Networks on the same dataset |
| `feature_engineering.py` | `FeatureEngineer` — creates time/amount/interaction features; selection via univariate, RFE, and importance analysis |
| `model_evaluation.py` | `ModelEvaluator` — accuracy, precision, recall, F1, AUC, confusion matrices, ROC/PR curves, production threshold check (≥0.90) |
| `production_integration.py` | `ProductionIntegrator` — promotes winning configs to production via Parameter Store, S3 config, and Step Functions trigger |

## Setup

```bash
cd ml-experimentation-workflow
pip install -r requirements.txt
```

Dependencies: boto3, sagemaker, pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, pyyaml, hypothesis, pytest, moto.

## Usage

### Experiment Tracking

```python
from src.experiment_tracking import ExperimentTracker

tracker = ExperimentTracker(experiment_name="fraud-detection-v2")

# Log a trial
tracker.log_parameters({"max_depth": 6, "eta": 0.1, "num_round": 200})
tracker.log_metrics({"accuracy": 0.94, "f1_score": 0.87, "auc": 0.96})
tracker.log_artifact("model", "s3://bucket/models/xgboost-v2.tar.gz")

# Query past experiments
results = tracker.query_experiments(metric="accuracy", ascending=False)
```

### Hyperparameter Tuning

```python
from src.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner()

# Random search
best_params = tuner.random_search(
    param_distributions={
        "max_depth": [4, 6, 8, 10],
        "eta": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.7, 0.8, 0.9],
    },
    n_iterations=20,
)

# SageMaker Automatic Model Tuning (Bayesian)
best_params = tuner.bayesian_search(
    objective_metric="validation:auc",
    max_jobs=50,
)
```

### Promoting to Production

```python
from src.production_integration import ProductionIntegrator

integrator = ProductionIntegrator(
    config_bucket="fraud-detection-config-bucket",
    pipeline_arn="arn:aws:states:us-east-1:123456789:stateMachine:FraudDetectionTrainingWorkflow",
)

# Promote winning hyperparameters
integrator.promote_to_production(
    hyperparameters={"max_depth": "6", "eta": "0.1", "num_round": "200"},
    trigger_pipeline=True,
)
```

This validates the parameters, backs up current Parameter Store values to S3, writes new values to `/fraud-detection/hyperparameters/*`, generates `production-model-config.yaml` in S3, triggers the production pipeline, and logs the promotion event.

## Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Property-based tests (Hypothesis)
pytest tests/property/

# Verbose with coverage
pytest tests/ -v --tb=short
```

## Project Structure

```
ml-experimentation-workflow/
├── src/
│   ├── __init__.py
│   ├── experiment_tracking.py
│   ├── hyperparameter_tuning.py
│   ├── algorithm_comparison.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   └── production_integration.py
├── tests/
│   ├── unit/                  # Unit tests for all modules
│   └── property/              # Property-based tests (Hypothesis)
├── notebooks/                 # Jupyter notebooks for exploration
├── infrastructure/            # CDK stacks (placeholder)
├── requirements.txt
└── README.md
```
