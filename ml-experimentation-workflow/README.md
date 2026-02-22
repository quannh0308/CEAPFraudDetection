# ML Experimentation Workflow

A Python toolkit for data scientists to experiment with and optimize fraud detection models in SageMaker Studio. It bridges exploratory data science and the [production fraud-detection-ml-pipeline](../README.md), providing structured experiment tracking, hyperparameter tuning, algorithm comparison, feature engineering, model evaluation, and a promotion path to production.

## Table of Contents

- [System Architecture](#system-architecture)
- [Modules](#modules)
- [SageMaker Studio Setup](#sagemaker-studio-setup)
- [Quick Start](#quick-start)
- [Example Workflows](#example-workflows)
- [Integration with Production Pipeline](#integration-with-production-pipeline)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Known Limitations and Future Work](#known-limitations-and-future-work)
- [Troubleshooting](#troubleshooting)

## System Architecture

The system is organized into three layers: the SageMaker Studio experimentation environment, an experiment tracking layer, and a production integration layer that connects to the live fraud-detection-ml-pipeline.

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
│                                          │                          │
│                                ┌─────────┴──────────┐               │
│                                │   ab_testing        │               │
│                                │  (Deploy Challenger │               │
│                                │   → Compare → Promote│              │
│                                │   to Champion)       │              │
│                                └─────────┬──────────┘               │
└──────────────────────────────────────────┼──────────────────────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     ▼                     ▼                     ▼
            ┌──────────────┐      ┌──────────────┐     ┌──────────────┐
            │  SSM Param   │      │  S3 Config   │     │Step Functions│
            │  Store       │      │  Bucket      │     │  (Pipeline)  │
            └──────┬───────┘      └──────┬───────┘     └──────┬───────┘
                   │                     │                    │
                   └─────────────────────┼────────────────────┘
                                         ▼
                              ┌──────────────────────┐
                              │  fraud-detection-ml-  │
                              │  pipeline (production)│
                              └──────────────────────┘
```

### Data Flow

1. Data scientists load fraud detection data from `s3://fraud-detection-data` (Parquet train/validation/test splits).
2. Experiments are run in SageMaker Studio notebooks using the Python modules in `src/`.
3. Every experiment is tracked via SageMaker Experiments (parameters, metrics, artifacts).
4. Winning configurations are promoted to production through:
   - **Parameter Store**: hyperparameters written to `/fraud-detection/hyperparameters/*`
   - **S3 Config**: `production-model-config.yaml` written to `s3://fraud-detection-config/`
   - **Step Functions**: optional trigger of the production retraining pipeline
5. A/B testing allows safe comparison of challenger models against the production champion.

### AWS Services Used

| Service | Purpose |
|---------|---------|
| SageMaker Studio | Notebook environment for experimentation |
| SageMaker Experiments | Experiment tracking and versioning |
| SageMaker Endpoints | Model deployment for A/B testing |
| SageMaker Automatic Model Tuning | Bayesian hyperparameter optimization |
| S3 | Data storage, config files, model artifacts, backups |
| Systems Manager Parameter Store | Production hyperparameter management |
| Step Functions | Production pipeline orchestration |
| CloudWatch Logs | Training job logs and monitoring |

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `experiment_tracking.py` | `ExperimentTracker` | Wraps SageMaker Experiments for logging parameters, metrics, and artifacts |
| `hyperparameter_tuning.py` | `HyperparameterTuner` | Grid search, random search, and SageMaker Automatic Model Tuning (Bayesian) |
| `algorithm_comparison.py` | `AlgorithmComparator` | Compares XGBoost, LightGBM, Random Forest, and Neural Networks on the same dataset |
| `feature_engineering.py` | `FeatureEngineer` | Creates time/amount/interaction features; selection via univariate, RFE, and importance analysis |
| `model_evaluation.py` | `ModelEvaluator` | Accuracy, precision, recall, F1, AUC, confusion matrices, ROC/PR curves, production threshold check (≥0.90) |
| `production_integration.py` | `ProductionIntegrator` | Promotes winning configs to production via Parameter Store, S3 config, and Step Functions trigger |
| `ab_testing.py` | `ABTestingManager` | Deploys challenger endpoints, configures traffic splits, compares endpoints, promotes challengers |


## SageMaker Studio Setup

The infrastructure for SageMaker Studio is defined as a CDK stack in `infrastructure/`.

### Prerequisites

- Python 3.8+
- Node.js 14+ (for AWS CDK CLI)
- AWS CLI configured with appropriate credentials
- AWS CDK CLI (`npm install -g aws-cdk`)

### Deploy SageMaker Studio

```bash
cd ml-experimentation-workflow/infrastructure

# Deploy the stack (bootstraps CDK, synthesizes, and deploys)
./deploy.sh

# Deploy to a specific region
./deploy.sh --region us-west-2

# Tear down the stack
./deploy.sh --destroy
```

The CDK stack (`sagemaker_studio_stack.py`) creates:

- **SageMaker Studio Domain** (`fraud-detection-experimentation`) with IAM authentication
- **IAM Execution Role** with permissions for:
  - S3 read/write on `fraud-detection-data`, `fraud-detection-config`, `fraud-detection-models`
  - SageMaker training jobs, endpoints, hyperparameter tuning, and experiments
  - Parameter Store read/write under `/fraud-detection/*`
  - Step Functions execution for `fraud-detection-*` state machines
  - CloudWatch Logs for `/aws/sagemaker/*`
- **S3 Bucket** (`fraud-detection-config`) with:
  - Versioning enabled
  - `archive/` prefix with Glacier transition after 30 days, expiration after 365 days
  - `parameter-store-backups/` prefix with expiration after 90 days
- **Kernel Gateway** configured with `ml.t3.medium` instance and the Data Science 1.0 image

After deployment, the SageMaker Studio URL is printed in the stack outputs.

### Install Python Dependencies

```bash
cd ml-experimentation-workflow
pip install -r requirements.txt
```

Dependencies: boto3, sagemaker, pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, pyyaml, hypothesis, pytest, moto.

### First-Time Setup in SageMaker Studio

1. Open SageMaker Studio from the AWS Console using the URL from the CDK output.
2. Create a new user profile (or use the default one).
3. Open a terminal in Studio and clone this repository.
4. Install dependencies: `pip install -r requirements.txt`
5. Open any notebook in `notebooks/` to get started.

> **Important: Data Dependency** — The experimentation notebooks load Parquet data splits from `s3://fraud-detection-data-{BUCKET_SUFFIX}/prepared/` (train.parquet, validation.parquet, test.parquet). These files are created by the training pipeline's DataPrep stage. You must deploy and run the training pipeline at least once before the notebooks will work. See the [main README's End-to-End Workflow](../README.md#end-to-end-workflow) for the full deployment sequence.

## Quick Start

> **Prerequisite:** The training pipeline must have run at least once to create the Parquet data splits in S3. See the [End-to-End Workflow](../README.md#end-to-end-workflow) in the main README.

```python
from src.experiment_tracking import ExperimentTracker
from src.hyperparameter_tuning import HyperparameterTuner
from src.model_evaluation import ModelEvaluator
from src.production_integration import ProductionIntegrator

# 1. Set up tracking
tracker = ExperimentTracker()

# 2. Tune hyperparameters
tuner = HyperparameterTuner(tracker=tracker)
results = tuner.grid_search(
    model_class=XGBClassifier,
    param_grid={"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1]},
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
)

# 3. Evaluate the best model
evaluator = ModelEvaluator()
eval_results = evaluator.evaluate_model(model, X_test, y_test, baseline_metrics)

# 4. Promote to production if threshold met
if eval_results["meets_production_threshold"]:
    integrator = ProductionIntegrator(experiment_tracker=tracker)
    integrator.promote_to_production(
        experiment_id="exp-001",
        hyperparameters=best_params,
        metrics=eval_results["metrics"],
        approver="data-science-team",
        trigger_pipeline=True,
    )
```

## Example Workflows

### Experiment Tracking

Track every experiment with full metadata for reproducibility.

```python
from src.experiment_tracking import ExperimentTracker

tracker = ExperimentTracker()

# Start an experiment
experiment_id = tracker.start_experiment(
    experiment_name="fraud-detection-v2",
    algorithm="xgboost",
    dataset_version="v1.2.3",
    code_version="abc1234",
)

# Log hyperparameters and metrics
tracker.log_parameters(experiment_id, {"max_depth": 6, "eta": 0.1, "num_round": 200})
tracker.log_metrics(experiment_id, {"accuracy": 0.94, "f1_score": 0.87, "auc_roc": 0.96})

# Upload artifacts
tracker.log_artifacts(
    experiment_id,
    ["model.pkl", "confusion_matrix.png"],
    s3_bucket="fraud-detection-models",
    s3_prefix="experiments",
)

# Close the experiment
tracker.close_experiment(experiment_id)

# Query past experiments
results = tracker.query_experiments(
    experiment_name="fraud-detection-v2",
    min_accuracy=0.95,
)
```

See `notebooks/01_data_exploration.ipynb` for a complete data exploration workflow.

### Hyperparameter Tuning

Three strategies are available: grid search, random search, and Bayesian optimization.

```python
from src.hyperparameter_tuning import HyperparameterTuner
from xgboost import XGBClassifier

tuner = HyperparameterTuner(tracker=tracker)

# Grid search — exhaustive over all combinations
grid_results = tuner.grid_search(
    model_class=XGBClassifier,
    param_grid={
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
    },
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    experiment_name="xgboost-grid-search",
)
print(f"Best params: {grid_results['best_params']}")

# Random search — sample from distributions
import random
random_results = tuner.random_search(
    model_class=XGBClassifier,
    param_distributions={
        "max_depth": [3, 5, 7, 10],
        "learning_rate": lambda: random.uniform(0.01, 0.3),
        "n_estimators": [50, 100, 150, 200],
    },
    n_iter=20,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    experiment_name="xgboost-random-search",
)

# Bayesian optimization via SageMaker Automatic Model Tuning
from sagemaker.tuner import IntegerParameter, ContinuousParameter
bayesian_results = tuner.bayesian_optimization(
    estimator=xgb_estimator,
    objective_metric_name="validation:auc",
    hyperparameter_ranges={
        "max_depth": IntegerParameter(3, 10),
        "eta": ContinuousParameter(0.01, 0.3),
        "subsample": ContinuousParameter(0.5, 1.0),
    },
    max_jobs=50,
    max_parallel_jobs=5,
    train_data_s3="s3://fraud-detection-data/train",
    validation_data_s3="s3://fraud-detection-data/validation",
)
```

See `notebooks/02_hyperparameter_tuning.ipynb` for the full workflow.

### Algorithm Comparison

Compare multiple algorithms side-by-side on the same dataset.

```python
from src.algorithm_comparison import AlgorithmComparator

comparator = AlgorithmComparator(tracker=tracker)

# Compare default algorithms (XGBoost, LightGBM, Random Forest, Neural Network)
results_df = comparator.compare_algorithms(
    X_train, y_train, X_test, y_test,
    experiment_name="fraud-detection-comparison",
)
print(results_df[["algorithm", "accuracy", "f1", "auc_roc", "training_time_seconds"]])

# Generate comparison visualizations
comparator.visualize_comparison(results_df, save_path="comparison.png")
```

See `notebooks/03_algorithm_comparison.ipynb` for the full workflow.

### Feature Engineering

Create derived features and select the most important ones.

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Create time-based features (hour, day_of_week, is_weekend, is_night)
df = engineer.create_time_features(df)

# Create amount-based features (log, squared, sqrt)
df = engineer.create_amount_features(df)

# Create interaction features
df = engineer.create_interaction_features(df, [("V1", "V2"), ("V3", "V4")])

# Select top features using univariate tests
selected_features, scores = engineer.select_features_univariate(X, y, k=20)

# Select features using Recursive Feature Elimination
selected_rfe, ranking = engineer.select_features_rfe(X, y, n_features=15)

# Analyze feature importance with Random Forest
importance_df = engineer.analyze_feature_importance(X, y)
```

See `notebooks/04_feature_engineering.ipynb` for the full workflow.

### Model Evaluation

Evaluate models with standardized metrics and production threshold checking.

```python
from src.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Calculate metrics
metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
# {'accuracy': 0.96, 'precision': 0.92, 'recall': 0.88, 'f1_score': 0.90, 'auc_roc': 0.95}

# Generate visualizations
evaluator.plot_confusion_matrix(y_true, y_pred, "confusion_matrix.png")
evaluator.plot_roc_curve(y_true, y_pred_proba, "roc_curve.png")
evaluator.plot_precision_recall_curve(y_true, y_pred_proba, "pr_curve.png")

# Compare to production baseline
baseline = {"accuracy": 0.952, "precision": 0.89, "recall": 0.85}
comparison = evaluator.compare_to_baseline(metrics, baseline)

# Full evaluation with threshold check (accuracy >= 0.90 required)
results = evaluator.evaluate_model(model, X_test, y_test, baseline_metrics=baseline)
print(f"Meets production threshold: {results['meets_production_threshold']}")
# Warning issued if accuracy < 0.80
```

See `notebooks/05_production_promotion.ipynb` for the full workflow.

### Production Promotion

Promote winning configurations to the production pipeline.

```python
from src.production_integration import ProductionIntegrator

integrator = ProductionIntegrator(experiment_tracker=tracker)

# Validate hyperparameters before promotion
hyperparameters = {
    "objective": "binary:logistic",
    "num_round": 150,
    "max_depth": 7,
    "eta": 0.15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
integrator.validate_hyperparameters(hyperparameters)

# Full promotion: Parameter Store + S3 config + optional pipeline trigger
result = integrator.promote_to_production(
    experiment_id="exp-20240115-001",
    hyperparameters=hyperparameters,
    metrics={"accuracy": 0.96, "precision": 0.92, "recall": 0.88},
    approver="data-science-team",
    trigger_pipeline=True,
)
print(f"Backup key: {result['promotion_event']['backup_key']}")
print(f"Pipeline ARN: {result['execution_arn']}")

# Check pipeline status
status = integrator.check_pipeline_status(result["execution_arn"])
print(f"Pipeline status: {status['status']}")

# Rollback if needed
integrator.rollback_parameter_store(result["promotion_event"]["backup_key"])
```

### A/B Testing

Deploy challenger models and compare against the production champion.

```python
from src.ab_testing import ABTestingManager

manager = ABTestingManager()

# Deploy a challenger endpoint
challenger = manager.deploy_challenger_endpoint(
    experiment_id="exp-20240115-001",
    model_artifact_s3_path="s3://fraud-detection-models/exp-001/model.tar.gz",
)

# Generate traffic split config (staged rollout: 1% → 10% → 50% → 100%)
config = manager.generate_traffic_split_config(
    champion_endpoint="fraud-detection-production",
    challenger_endpoint=challenger,
    challenger_traffic_pct=10,
)

# Compare endpoints on test data
comparison, champ_preds, chal_preds = manager.compare_endpoints(
    champion_endpoint="fraud-detection-production",
    challenger_endpoint=challenger,
    test_data=test_records,
)
print(f"Champion avg latency: {comparison['champion']['avg_latency_ms']:.1f}ms")
print(f"Challenger avg latency: {comparison['challenger']['avg_latency_ms']:.1f}ms")

# Promote challenger if it wins
manager.promote_challenger_to_champion(challenger_endpoint=challenger)
```

See `notebooks/05_production_promotion.ipynb` for the full A/B testing workflow.

## Integration with Production Pipeline

This experimentation workflow integrates with the production `fraud-detection-ml-pipeline` through three channels:

### Parameter Store Integration

Hyperparameters are written to the same Parameter Store paths that the production pipeline reads:

| Parameter | Path |
|-----------|------|
| objective | `/fraud-detection/hyperparameters/objective` |
| num_round | `/fraud-detection/hyperparameters/num_round` |
| max_depth | `/fraud-detection/hyperparameters/max_depth` |
| eta | `/fraud-detection/hyperparameters/eta` |
| subsample | `/fraud-detection/hyperparameters/subsample` |
| colsample_bytree | `/fraud-detection/hyperparameters/colsample_bytree` |

Before every write, current values are backed up to `s3://fraud-detection-config/parameter-store-backups/`.

### S3 Configuration File

A YAML configuration file is written to `s3://fraud-detection-config/production-model-config.yaml` containing:

```yaml
model:
  algorithm: xgboost
  version: <experiment_id>
  hyperparameters:
    max_depth: 7
    eta: 0.15
    # ...
  performance:
    accuracy: 0.96
    precision: 0.92
  tested_date: "2024-01-15"
  approved_by: data-science-team
```

Previous configs are archived to `s3://fraud-detection-config/archive/`.

### Step Functions Pipeline Trigger

The production pipeline can be triggered directly from the experimentation environment:

```python
integrator = ProductionIntegrator()
arn = integrator.trigger_production_pipeline("exp-20240115-001")
status = integrator.check_pipeline_status(arn)
```

The pipeline state machine ARN: `arn:aws:states:us-east-1:<account>:stateMachine:fraud-detection-training-pipeline`

### Production Thresholds

- **Production ready**: accuracy ≥ 0.90
- **Warning**: accuracy < 0.80 (below production quality)

## Testing

```bash
cd ml-experimentation-workflow

# Run all tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Property-based tests (Hypothesis)
pytest tests/property/

# Verbose with short tracebacks
pytest tests/ -v --tb=short
```

## Project Structure

```
ml-experimentation-workflow/
├── src/
│   ├── __init__.py
│   ├── experiment_tracking.py      # ExperimentTracker
│   ├── hyperparameter_tuning.py    # HyperparameterTuner
│   ├── algorithm_comparison.py     # AlgorithmComparator
│   ├── feature_engineering.py      # FeatureEngineer
│   ├── model_evaluation.py         # ModelEvaluator
│   ├── production_integration.py   # ProductionIntegrator
│   └── ab_testing.py               # ABTestingManager
├── tests/
│   ├── unit/                       # Unit tests for all modules
│   └── property/                   # Property-based tests (Hypothesis)
├── notebooks/
│   ├── template.ipynb              # Notebook template with experiment tracking
│   ├── 01_data_exploration.ipynb
│   ├── 02_hyperparameter_tuning.ipynb
│   ├── 03_algorithm_comparison.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_production_promotion.ipynb
├── infrastructure/
│   ├── sagemaker_studio_stack.py   # CDK stack for SageMaker Studio
│   ├── app.py                      # CDK app entry point
│   ├── deploy.sh                   # Deployment script
│   ├── cdk.json                    # CDK configuration
│   └── requirements.txt            # CDK dependencies
├── REPRODUCIBILITY.md              # Reproducibility checklist
├── requirements.txt                # Python dependencies
└── README.md
```

## Known Limitations and Future Work

### What the experimentation flow produces vs what production uses

The experimentation notebooks generate three categories of insights, but only one is currently consumed by the production pipeline:

| Insight | Produced by | Used by production? | Notes |
|---------|-------------|---------------------|-------|
| Optimized hyperparameters | Notebook 02 (grid/random search) | ✅ Yes | Written to Parameter Store, read by TrainHandler at runtime |
| Best algorithm | Notebook 03 (algorithm comparison) | ❌ No | Production pipeline is hardcoded to XGBoost. RandomForest outperformed XGBoost in testing (accuracy 0.9994 vs 0.9991) but can't be used without pipeline changes |
| Best feature subset | Notebook 04 (feature selection) | ❌ No | 20 features identified as most important, but the pipeline trains on all features. Feature selection would require changes to the DataPrep stage |

### Gaps to address

- **Algorithm flexibility**: The TrainHandler and CDK Step Functions definition both hardcode the XGBoost container image. Supporting alternative algorithms (RandomForest, LightGBM) would require making the algorithm configurable via Parameter Store or S3 config, and updating the SageMaker container image selection.

- **Feature selection integration**: The experimentation flow identifies optimal feature subsets, but the training pipeline uses all features from the dataset. Integrating feature selection would require the DataPrep stage to filter columns based on a feature list stored in S3 or Parameter Store.

- **Narrow hyperparameter grid**: The default grid search in notebook 02 tests a limited range. For more meaningful optimization, expand the parameter grid (e.g., `eta`: [0.01, 0.05, 0.1, 0.15, 0.2, 0.3], `max_depth`: [3, 5, 7, 9, 12], `num_round`: [50, 100, 150, 200, 300]).

- **SageMaker-specific features**: Bayesian optimization (SageMaker Automatic Model Tuning) and experiment querying (SageMaker Experiments) only work inside SageMaker Studio, not locally. These sections are skipped when running notebooks on a local machine.

- **A/B testing**: The A/B testing module (`ABTestingManager`) requires a deployed SageMaker endpoint and model artifacts. It's not functional until the training pipeline has completed and deployed a production endpoint.

## Troubleshooting

### SageMaker Studio won't start

- Verify the CDK stack deployed successfully: check CloudFormation in the AWS Console.
- Ensure your IAM user has `sagemaker:CreatePresignedDomainUrl` permission.
- Check that the VPC has internet access (NAT gateway or public subnets).

### S3 access denied errors

- Verify the SageMaker execution role has the required S3 permissions. The CDK stack grants access to `fraud-detection-data`, `fraud-detection-config`, and `fraud-detection-models` buckets.
- Check that the bucket names match exactly (they are case-sensitive).
- If using a custom role, ensure it includes `s3:GetObject`, `s3:PutObject`, and `s3:ListBucket`.

### Parameter Store write failures

- Ensure the execution role has `ssm:PutParameter` permission for `/fraud-detection/*`.
- Check that parameter values are valid strings. Numeric values are automatically converted.
- If a write fails mid-way, use `integrator.rollback_parameter_store(backup_key)` to restore from the backup.

### Training job failures

- Use `handle_sagemaker_training_error(sagemaker_client, job_name)` to get the failure reason and CloudWatch log reference.
- Common causes: insufficient instance quota, invalid hyperparameters, data format issues.
- Check CloudWatch Logs at `/aws/sagemaker/TrainingJobs/<job-name>`.

### Hyperparameter validation errors

- All six required parameters must be provided: `objective`, `num_round`, `max_depth`, `eta`, `subsample`, `colsample_bytree`.
- Valid ranges: `max_depth` [1, 20], `eta` (0, 1], `num_round` [1, 1000], `subsample` (0, 1], `colsample_bytree` (0, 1].

### A/B testing endpoint issues

- Ensure the model artifact exists at the specified S3 path.
- Check SageMaker endpoint quotas in your account.
- Endpoint deployment can take 5-10 minutes. The `deploy_challenger_endpoint` method waits automatically.

### Import errors

- Run `pip install -r requirements.txt` to install all dependencies.
- XGBoost and LightGBM are optional for algorithm comparison; the module gracefully handles missing imports.
- The `sagemaker` package is required for Bayesian optimization and experiment tracking with SageMaker Experiments.

### CDK deployment failures

- Ensure CDK is bootstrapped: `cdk bootstrap` (the deploy script does this automatically).
- Check that your AWS credentials have CloudFormation, IAM, S3, and SageMaker permissions.
- If the bucket name `fraud-detection-config` is taken, modify `_create_config_bucket` in the CDK stack.
