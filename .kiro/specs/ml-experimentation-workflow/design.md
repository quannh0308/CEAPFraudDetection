# Design Document: ML Experimentation Workflow

## Overview

The ML Experimentation Workflow provides a SageMaker Studio-based environment for data scientists to explore, experiment with, and optimize fraud detection models. This system bridges the gap between exploratory data science and production ML pipelines by providing:

1. **Experimentation Tools**: Notebooks for hyperparameter tuning, algorithm comparison, and feature engineering
2. **Experiment Tracking**: MLflow or SageMaker Experiments for versioning and reproducibility
3. **Production Integration**: Seamless promotion of winning configurations to the fraud-detection-ml-pipeline via Parameter Store and configuration files
4. **A/B Testing Support**: Tools for deploying challenger models and comparing against production champions

The design emphasizes developer productivity, reproducibility, and safe production integration. Data scientists can iterate quickly in notebooks while maintaining full traceability and the ability to promote improvements to production with a single command.

### Key Design Principles

- **Notebook-First Development**: All experimentation happens in Jupyter notebooks with rich visualizations
- **Automated Tracking**: Every experiment is automatically logged with full context (hyperparameters, metrics, code versions)
- **Safe Production Promotion**: Configuration changes are validated, backed up, and versioned before deployment
- **Reproducibility**: All experiments can be reproduced from tracked metadata
- **Integration Over Replacement**: Works alongside existing production pipeline, not replacing it

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  SageMaker Studio Notebook Environment                          │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ Data Exploration │  │ Hyperparameter   │  │ Algorithm    │ │
│  │ Notebooks        │  │ Tuning Notebooks │  │ Comparison   │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ Feature          │  │ Model Evaluation │  │ Production   │ │
│  │ Engineering      │  │ Framework        │  │ Integration  │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Experiment Tracking (MLflow or SageMaker Experiments)          │
│  ─────────────────────────────────────────────────────────────  │
│  • Experiment metadata (ID, timestamp, user)                    │
│  • Hyperparameters and algorithm configuration                  │
│  • Performance metrics (accuracy, precision, recall, AUC)       │
│  • Model artifacts and dataset versions                         │
│  • Code versions and dependencies                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Production Integration Layer                                    │
│  ─────────────────────────────────────────────────────────────  │
│  • AWS Systems Manager Parameter Store (hyperparameters)        │
│  • S3 Configuration Files (production-model-config.yaml)        │
│  • Step Functions Trigger (retrain production pipeline)         │
│  • SageMaker Endpoints (A/B testing challenger deployment)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Production Pipeline (fraud-detection-ml-pipeline)               │
│  ─────────────────────────────────────────────────────────────  │
│  • Reads hyperparameters from Parameter Store                   │
│  • Reads configuration from S3                                  │
│  • Trains models with experiment-optimized settings             │
│  • Deploys to production endpoints                              │
└─────────────────────────────────────────────────────────────────┘
```


### Component Interaction Flow

#### Experimentation Flow
```
1. Data Scientist opens SageMaker Studio notebook
2. Load data from S3 (fraud-detection-data bucket)
3. Run experiments (hyperparameter tuning, algorithm comparison, feature engineering)
4. MLflow/SageMaker Experiments automatically logs all experiment metadata
5. Evaluate results and select winning configuration
6. Promote winning configuration to production via helper functions
```

#### Production Promotion Flow
```
1. Data scientist calls promote_to_production(experiment_id)
2. System retrieves experiment metadata from MLflow/SageMaker Experiments
3. System validates hyperparameters and configuration
4. System backs up current Parameter Store values with timestamp
5. System writes new hyperparameters to Parameter Store
6. System generates and writes production-model-config.yaml to S3
7. System logs promotion event to experiment tracker
8. (Optional) System triggers production pipeline retraining
```

#### A/B Testing Flow
```
1. Data scientist calls deploy_challenger(experiment_id)
2. System deploys new model to SageMaker endpoint with unique name
3. System provides traffic splitting configuration template
4. Data scientist monitors champion vs challenger metrics
5. If challenger wins, call promote_challenger_to_champion()
6. System updates production endpoint configuration
```

## Components and Interfaces

### 1. SageMaker Studio Environment

**Purpose**: Provide pre-configured notebook environment with all necessary libraries and permissions.

**Implementation**:
- SageMaker Studio Domain with data science user profile
- Pre-installed libraries: XGBoost, LightGBM, scikit-learn, pandas, numpy, matplotlib, seaborn, mlflow, boto3
- IAM role with permissions for:
  - S3 read/write (fraud-detection-data, fraud-detection-config, fraud-detection-models)
  - SageMaker training jobs and endpoints
  - Parameter Store read/write
  - Step Functions execution
  - CloudWatch Logs

**Configuration**:
```yaml
SageMakerStudioDomain:
  DomainName: fraud-detection-experimentation
  AuthMode: IAM
  DefaultUserSettings:
    ExecutionRole: !GetAtt SageMakerExecutionRole.Arn
    JupyterServerAppSettings:
      DefaultResourceSpec:
        InstanceType: ml.t3.medium
        SageMakerImageArn: arn:aws:sagemaker:us-east-1:081325390199:image/datascience-1.0
```


### 2. Experiment Tracking System

**Purpose**: Log and version all experiments for reproducibility and audit trail.

**Technology Choice**: SageMaker Experiments (native AWS integration, no additional infrastructure)

**Alternative**: MLflow (more features, requires hosting)

**Recommendation**: Start with SageMaker Experiments for simplicity, migrate to MLflow if advanced features needed.

**SageMaker Experiments API**:
```python
import sagemaker
from sagemaker.experiments import Run

# Create experiment run
with Run(
    experiment_name="fraud-detection-optimization",
    run_name=f"xgboost-tuning-{timestamp}",
    sagemaker_session=sagemaker_session
) as run:
    # Log hyperparameters
    run.log_parameters({
        "max_depth": 7,
        "eta": 0.15,
        "num_round": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })
    
    # Train model
    model = train_model(hyperparameters)
    
    # Log metrics
    run.log_metric("accuracy", 0.961)
    run.log_metric("precision", 0.92)
    run.log_metric("recall", 0.88)
    run.log_metric("f1_score", 0.90)
    run.log_metric("auc_roc", 0.95)
    
    # Log artifacts
    run.log_file("model.pkl", is_output=True)
    run.log_file("confusion_matrix.png", is_output=True)
```

**Query API**:
```python
from sagemaker.experiments import search_expression

# Find best experiments by accuracy
search_exp = search_expression.SearchExpression(
    filters=[
        search_expression.Filter(
            name="Metrics.accuracy.Max",
            operator=search_expression.Operator.GREATER_THAN_OR_EQUAL,
            value="0.95"
        )
    ]
)

results = sagemaker.experiments.search(
    search_expression=search_exp,
    sort_by="Metrics.accuracy.Max",
    sort_order="Descending",
    max_results=10
)
```


### 3. Hyperparameter Tuning Module

**Purpose**: Provide utilities for grid search, random search, and Bayesian optimization.

**Implementation Options**:

**Option A: SageMaker Automatic Model Tuning (Recommended)**
```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

# Define hyperparameter ranges
hyperparameter_ranges = {
    'max_depth': IntegerParameter(3, 10),
    'eta': ContinuousParameter(0.01, 0.3),
    'subsample': ContinuousParameter(0.5, 1.0),
    'colsample_bytree': ContinuousParameter(0.5, 1.0),
    'num_round': IntegerParameter(50, 200)
}

# Create tuner
tuner = HyperparameterTuner(
    estimator=xgboost_estimator,
    objective_metric_name='validation:auc',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=20,
    max_parallel_jobs=5,
    strategy='Bayesian'
)

# Run tuning
tuner.fit({'train': train_data_s3, 'validation': val_data_s3})

# Get best hyperparameters
best_training_job = tuner.best_training_job()
best_hyperparameters = tuner.best_estimator().hyperparameters()
```

**Option B: Scikit-learn Grid/Random Search (For local experiments)**
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

# Grid search
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

**Performance Considerations**:
- SageMaker tuning: Parallel jobs on separate instances (faster, more expensive)
- Local tuning: Sequential on notebook instance (slower, cheaper)
- Recommendation: Use SageMaker for final tuning, local for quick iterations


### 4. Algorithm Comparison Module

**Purpose**: Provide utilities to train and compare multiple algorithms on the same dataset.

**Supported Algorithms**:
1. XGBoost (current production algorithm)
2. LightGBM (faster training, similar performance)
3. Random Forest (baseline, interpretable)
4. Neural Networks (MLP for tabular data)

**Implementation**:
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

def compare_algorithms(X_train, y_train, X_test, y_test):
    """Compare multiple algorithms and return performance metrics."""
    
    algorithms = {
        'XGBoost': XGBClassifier(
            max_depth=5,
            learning_rate=0.2,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        'LightGBM': LGBMClassifier(
            max_depth=5,
            learning_rate=0.2,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        'RandomForest': RandomForestClassifier(
            max_depth=10,
            n_estimators=100,
            max_features='sqrt'
        ),
        'NeuralNetwork': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=200
        )
    }
    
    results = []
    
    for name, model in algorithms.items():
        # Train and time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'algorithm': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'training_time_seconds': training_time
        }
        
        results.append(metrics)
        
        # Log to experiment tracker
        with Run(experiment_name="algorithm-comparison", run_name=f"{name}-{timestamp}") as run:
            run.log_parameters(model.get_params())
            run.log_metrics(metrics)
    
    return pd.DataFrame(results)
```

**Visualization**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_algorithm_comparison(results_df):
    """Create comparison visualizations."""
    
    # Metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'training_time_seconds']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        sns.barplot(data=results_df, x='algorithm', y=metric, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.show()
```


### 5. Feature Engineering Module

**Purpose**: Provide utilities for creating, transforming, and selecting features.

**Implementation**:
```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer:
    """Utilities for feature engineering experiments."""
    
    def create_time_features(self, df):
        """Create time-based features from transaction timestamp."""
        df['hour'] = pd.to_datetime(df['Time'], unit='s').dt.hour
        df['day_of_week'] = pd.to_datetime(df['Time'], unit='s').dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].between(22, 6).astype(int)
        return df
    
    def create_amount_features(self, df):
        """Create amount-based features."""
        df['amount_log'] = np.log1p(df['Amount'])
        df['amount_squared'] = df['Amount'] ** 2
        df['amount_sqrt'] = np.sqrt(df['Amount'])
        return df
    
    def create_interaction_features(self, df, feature_pairs):
        """Create interaction features between specified pairs."""
        for feat1, feat2 in feature_pairs:
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        return df
    
    def select_features_univariate(self, X, y, k=20):
        """Select top k features using univariate statistical tests."""
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        
        # Get feature scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        selected_features = scores.head(k)['feature'].tolist()
        return selected_features, scores
    
    def select_features_rfe(self, X, y, n_features=20):
        """Select features using recursive feature elimination."""
        estimator = RandomForestClassifier(n_estimators=50)
        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.support_].tolist()
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')
        
        return selected_features, feature_ranking
    
    def analyze_feature_importance(self, X, y):
        """Analyze feature importance using Random Forest."""
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
```

**Feature Engineering Workflow**:
```python
# Example usage in notebook
engineer = FeatureEngineer()

# Create new features
df = engineer.create_time_features(df)
df = engineer.create_amount_features(df)
df = engineer.create_interaction_features(df, [('V1', 'V2'), ('V3', 'V4')])

# Analyze feature importance
importance = engineer.analyze_feature_importance(X_train, y_train)
print(importance.head(20))

# Select top features
selected_features, scores = engineer.select_features_univariate(X_train, y_train, k=25)

# Train model with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

model = XGBClassifier()
model.fit(X_train_selected, y_train)
accuracy = model.score(X_test_selected, y_test)

# Log experiment
with Run(experiment_name="feature-engineering", run_name=f"top-25-features-{timestamp}") as run:
    run.log_parameters({'n_features': 25, 'selection_method': 'univariate'})
    run.log_metric('accuracy', accuracy)
    run.log_artifact('feature_scores.csv', scores.to_csv(index=False))
```


### 6. Model Evaluation Framework

**Purpose**: Provide standardized evaluation metrics and visualizations.

**Implementation**:
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Standardized model evaluation framework."""
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all standard metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Generate confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path='roc_curve.png'):
        """Generate ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        return fpr, tpr, auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path='pr_curve.png'):
        """Generate precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        return precision, recall
    
    def compare_to_baseline(self, current_metrics, baseline_metrics):
        """Compare current experiment to production baseline."""
        comparison = {}
        for metric in current_metrics:
            if metric in baseline_metrics:
                current = current_metrics[metric]
                baseline = baseline_metrics[metric]
                diff = current - baseline
                pct_change = (diff / baseline) * 100 if baseline != 0 else 0
                
                comparison[metric] = {
                    'current': current,
                    'baseline': baseline,
                    'difference': diff,
                    'percent_change': pct_change,
                    'improved': diff > 0
                }
        
        return comparison
    
    def evaluate_model(self, model, X_test, y_test, baseline_metrics=None):
        """Complete model evaluation with all metrics and visualizations."""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate visualizations
        cm = self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba)
        self.plot_precision_recall_curve(y_test, y_pred_proba)
        
        # Compare to baseline if provided
        comparison = None
        if baseline_metrics:
            comparison = self.compare_to_baseline(metrics, baseline_metrics)
        
        # Check production threshold
        meets_threshold = metrics['accuracy'] >= 0.90
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'comparison': comparison,
            'meets_production_threshold': meets_threshold
        }
```

**Usage in Notebook**:
```python
evaluator = ModelEvaluator()

# Load production baseline
baseline_metrics = {
    'accuracy': 0.952,
    'precision': 0.89,
    'recall': 0.85,
    'f1_score': 0.87,
    'auc_roc': 0.94
}

# Evaluate model
results = evaluator.evaluate_model(model, X_test, y_test, baseline_metrics)

print("Metrics:", results['metrics'])
print("Meets production threshold:", results['meets_production_threshold'])

if results['comparison']:
    print("\nComparison to baseline:")
    for metric, comp in results['comparison'].items():
        print(f"  {metric}: {comp['current']:.4f} (baseline: {comp['baseline']:.4f}, "
              f"change: {comp['percent_change']:+.2f}%)")
```


### 7. Production Integration Module

**Purpose**: Provide utilities to promote winning configurations to production pipeline.

**Implementation**:
```python
import boto3
import yaml
from datetime import datetime

class ProductionIntegrator:
    """Utilities for promoting experiments to production."""
    
    def __init__(self):
        self.ssm_client = boto3.client('ssm')
        self.s3_client = boto3.client('s3')
        self.sfn_client = boto3.client('stepfunctions')
        self.config_bucket = 'fraud-detection-config'
    
    def backup_current_parameters(self):
        """Backup current Parameter Store values before updating."""
        param_paths = [
            '/fraud-detection/hyperparameters/objective',
            '/fraud-detection/hyperparameters/num_round',
            '/fraud-detection/hyperparameters/max_depth',
            '/fraud-detection/hyperparameters/eta',
            '/fraud-detection/hyperparameters/subsample',
            '/fraud-detection/hyperparameters/colsample_bytree'
        ]
        
        backup = {}
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        for path in param_paths:
            try:
                response = self.ssm_client.get_parameter(Name=path)
                backup[path] = response['Parameter']['Value']
            except self.ssm_client.exceptions.ParameterNotFound:
                backup[path] = None
        
        # Save backup to S3
        backup_key = f'parameter-store-backups/backup-{timestamp}.yaml'
        self.s3_client.put_object(
            Bucket=self.config_bucket,
            Key=backup_key,
            Body=yaml.dump(backup)
        )
        
        return backup, backup_key
    
    def validate_hyperparameters(self, hyperparameters):
        """Validate hyperparameter names and value formats."""
        required_params = [
            'objective', 'num_round', 'max_depth', 
            'eta', 'subsample', 'colsample_bytree'
        ]
        
        # Check all required parameters present
        for param in required_params:
            if param not in hyperparameters:
                raise ValueError(f"Missing required hyperparameter: {param}")
        
        # Validate value ranges
        validations = {
            'max_depth': lambda x: 1 <= int(x) <= 20,
            'eta': lambda x: 0.0 < float(x) <= 1.0,
            'num_round': lambda x: 1 <= int(x) <= 1000,
            'subsample': lambda x: 0.0 < float(x) <= 1.0,
            'colsample_bytree': lambda x: 0.0 < float(x) <= 1.0
        }
        
        for param, validator in validations.items():
            if param in hyperparameters:
                try:
                    if not validator(hyperparameters[param]):
                        raise ValueError(f"Hyperparameter {param} value out of valid range")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for hyperparameter {param}: {e}")
        
        return True
    
    def write_hyperparameters_to_parameter_store(self, hyperparameters):
        """Write hyperparameters to Parameter Store."""
        # Validate first
        self.validate_hyperparameters(hyperparameters)
        
        # Backup current values
        backup, backup_key = self.backup_current_parameters()
        print(f"Backed up current parameters to: {backup_key}")
        
        # Write new values
        param_mapping = {
            'objective': '/fraud-detection/hyperparameters/objective',
            'num_round': '/fraud-detection/hyperparameters/num_round',
            'max_depth': '/fraud-detection/hyperparameters/max_depth',
            'eta': '/fraud-detection/hyperparameters/eta',
            'subsample': '/fraud-detection/hyperparameters/subsample',
            'colsample_bytree': '/fraud-detection/hyperparameters/colsample_bytree'
        }
        
        for param_name, param_path in param_mapping.items():
            value = str(hyperparameters[param_name])
            self.ssm_client.put_parameter(
                Name=param_path,
                Value=value,
                Type='String',
                Overwrite=True
            )
            print(f"Updated {param_path} = {value}")
        
        return backup_key
    
    def generate_production_config(self, experiment_id, hyperparameters, metrics, approver):
        """Generate production configuration file."""
        config = {
            'model': {
                'algorithm': 'xgboost',
                'version': experiment_id,
                'hyperparameters': hyperparameters,
                'performance': metrics,
                'tested_date': datetime.now().strftime('%Y-%m-%d'),
                'approved_by': approver
            }
        }
        
        return config
    
    def write_config_to_s3(self, config):
        """Write production configuration to S3."""
        # Validate schema
        required_keys = ['model']
        if not all(key in config for key in required_keys):
            raise ValueError("Invalid configuration schema")
        
        # Create versioned backup of current config
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        try:
            current_config = self.s3_client.get_object(
                Bucket=self.config_bucket,
                Key='production-model-config.yaml'
            )
            self.s3_client.put_object(
                Bucket=self.config_bucket,
                Key=f'archive/production-model-config-{timestamp}.yaml',
                Body=current_config['Body'].read()
            )
            print(f"Archived current config to: archive/production-model-config-{timestamp}.yaml")
        except self.s3_client.exceptions.NoSuchKey:
            print("No existing config to archive")
        
        # Write new config
        self.s3_client.put_object(
            Bucket=self.config_bucket,
            Key='production-model-config.yaml',
            Body=yaml.dump(config)
        )
        print("Wrote new config to: production-model-config.yaml")
    
    def trigger_production_pipeline(self, experiment_id):
        """Trigger production pipeline retraining."""
        state_machine_arn = 'arn:aws:states:us-east-1:123456789012:stateMachine:fraud-detection-training-pipeline'
        
        execution_name = f"experiment-{experiment_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        response = self.sfn_client.start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name,
            input=json.dumps({
                'experimentId': experiment_id,
                'triggeredBy': 'experimentation-workflow'
            })
        )
        
        execution_arn = response['executionArn']
        print(f"Started production pipeline execution: {execution_arn}")
        
        return execution_arn
    
    def check_pipeline_status(self, execution_arn):
        """Check production pipeline execution status."""
        response = self.sfn_client.describe_execution(executionArn=execution_arn)
        
        return {
            'status': response['status'],
            'startDate': response['startDate'],
            'stopDate': response.get('stopDate'),
            'output': response.get('output')
        }
    
    def promote_to_production(self, experiment_id, hyperparameters, metrics, approver, trigger_pipeline=False):
        """Complete promotion workflow."""
        print(f"Promoting experiment {experiment_id} to production...")
        
        # Write to Parameter Store
        backup_key = self.write_hyperparameters_to_parameter_store(hyperparameters)
        
        # Generate and write config file
        config = self.generate_production_config(experiment_id, hyperparameters, metrics, approver)
        self.write_config_to_s3(config)
        
        # Log promotion event
        promotion_event = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'approver': approver,
            'backup_key': backup_key,
            'metrics': metrics
        }
        
        # Optionally trigger pipeline
        execution_arn = None
        if trigger_pipeline:
            execution_arn = self.trigger_production_pipeline(experiment_id)
        
        print(f"✓ Promotion complete!")
        print(f"  - Parameter Store updated (backup: {backup_key})")
        print(f"  - Config file written to S3")
        if execution_arn:
            print(f"  - Production pipeline triggered: {execution_arn}")
        
        return {
            'promotion_event': promotion_event,
            'execution_arn': execution_arn
        }
```


### 8. A/B Testing Module

**Purpose**: Deploy challenger models and compare against production champion.

**Implementation**:
```python
class ABTestingManager:
    """Utilities for A/B testing challenger models."""
    
    def __init__(self):
        self.sagemaker_client = boto3.client('sagemaker')
        self.s3_client = boto3.client('s3')
    
    def deploy_challenger_endpoint(self, experiment_id, model_artifact_s3_path, instance_type='ml.m5.large'):
        """Deploy challenger model to separate endpoint."""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        endpoint_name = f"fraud-detection-challenger-{experiment_id}-{timestamp}"
        model_name = f"fraud-detection-model-{experiment_id}"
        endpoint_config_name = f"fraud-detection-config-{experiment_id}-{timestamp}"
        
        # Create model
        self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
                'ModelDataUrl': model_artifact_s3_path
            },
            ExecutionRoleArn='arn:aws:iam::123456789012:role/SageMakerExecutionRole'
        )
        
        # Create endpoint configuration
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': instance_type
            }]
        )
        
        # Create endpoint
        self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        print(f"Deploying challenger endpoint: {endpoint_name}")
        print("Waiting for endpoint to be in service...")
        
        waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        print(f"✓ Challenger endpoint deployed: {endpoint_name}")
        
        return endpoint_name
    
    def generate_traffic_split_config(self, champion_endpoint, challenger_endpoint, challenger_traffic_pct=10):
        """Generate traffic splitting configuration template."""
        config = {
            'endpoints': {
                'champion': {
                    'name': champion_endpoint,
                    'traffic_percentage': 100 - challenger_traffic_pct
                },
                'challenger': {
                    'name': challenger_endpoint,
                    'traffic_percentage': challenger_traffic_pct
                }
            },
            'rollout_plan': [
                {'stage': 1, 'challenger_traffic': 1, 'duration_hours': 24},
                {'stage': 2, 'challenger_traffic': 10, 'duration_hours': 48},
                {'stage': 3, 'challenger_traffic': 50, 'duration_hours': 72},
                {'stage': 4, 'challenger_traffic': 100, 'duration_hours': 0}
            ],
            'success_criteria': {
                'min_accuracy': 0.90,
                'max_latency_ms': 100,
                'min_improvement_pct': 1.0
            }
        }
        
        return config
    
    def compare_endpoints(self, champion_endpoint, challenger_endpoint, test_data):
        """Compare champion and challenger performance on test data."""
        runtime_client = boto3.client('sagemaker-runtime')
        
        champion_predictions = []
        challenger_predictions = []
        champion_latencies = []
        challenger_latencies = []
        
        for record in test_data:
            # Champion prediction
            start = time.time()
            champion_response = runtime_client.invoke_endpoint(
                EndpointName=champion_endpoint,
                Body=json.dumps(record),
                ContentType='application/json'
            )
            champion_latencies.append((time.time() - start) * 1000)
            champion_predictions.append(float(champion_response['Body'].read()))
            
            # Challenger prediction
            start = time.time()
            challenger_response = runtime_client.invoke_endpoint(
                EndpointName=challenger_endpoint,
                Body=json.dumps(record),
                ContentType='application/json'
            )
            challenger_latencies.append((time.time() - start) * 1000)
            challenger_predictions.append(float(challenger_response['Body'].read()))
        
        # Calculate metrics
        comparison = {
            'champion': {
                'avg_latency_ms': np.mean(champion_latencies),
                'p95_latency_ms': np.percentile(champion_latencies, 95),
                'p99_latency_ms': np.percentile(champion_latencies, 99)
            },
            'challenger': {
                'avg_latency_ms': np.mean(challenger_latencies),
                'p95_latency_ms': np.percentile(challenger_latencies, 95),
                'p99_latency_ms': np.percentile(challenger_latencies, 99)
            }
        }
        
        return comparison, champion_predictions, challenger_predictions
    
    def promote_challenger_to_champion(self, challenger_endpoint):
        """Promote challenger to production champion endpoint."""
        production_endpoint = 'fraud-detection-production'
        
        # Get challenger endpoint config
        challenger_desc = self.sagemaker_client.describe_endpoint(
            EndpointName=challenger_endpoint
        )
        challenger_config = challenger_desc['EndpointConfigName']
        
        # Update production endpoint to use challenger config
        self.sagemaker_client.update_endpoint(
            EndpointName=production_endpoint,
            EndpointConfigName=challenger_config
        )
        
        print(f"Promoting challenger {challenger_endpoint} to production...")
        print("Waiting for endpoint update...")
        
        waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=production_endpoint)
        
        print(f"✓ Challenger promoted to production!")
        
        # Clean up old challenger endpoint
        self.sagemaker_client.delete_endpoint(EndpointName=challenger_endpoint)
        print(f"Cleaned up challenger endpoint: {challenger_endpoint}")
```

**Usage in Notebook**:
```python
ab_manager = ABTestingManager()

# Deploy challenger
challenger_endpoint = ab_manager.deploy_challenger_endpoint(
    experiment_id='exp-20240115-001',
    model_artifact_s3_path='s3://fraud-detection-models/exp-20240115-001/model.tar.gz'
)

# Generate traffic split config
traffic_config = ab_manager.generate_traffic_split_config(
    champion_endpoint='fraud-detection-production',
    challenger_endpoint=challenger_endpoint,
    challenger_traffic_pct=10
)

print("Traffic split configuration:")
print(yaml.dump(traffic_config))

# Compare performance
comparison, champ_preds, chal_preds = ab_manager.compare_endpoints(
    champion_endpoint='fraud-detection-production',
    challenger_endpoint=challenger_endpoint,
    test_data=test_transactions
)

print("\nPerformance comparison:")
print(f"Champion avg latency: {comparison['champion']['avg_latency_ms']:.2f}ms")
print(f"Challenger avg latency: {comparison['challenger']['avg_latency_ms']:.2f}ms")

# If challenger wins, promote
if challenger_better_than_champion(comparison):
    ab_manager.promote_challenger_to_champion(challenger_endpoint)
```


## Data Models

### ExperimentMetadata
```python
@dataclass
class ExperimentMetadata:
    """Metadata for a single experiment run."""
    experiment_id: str
    experiment_name: str
    run_name: str
    start_timestamp: datetime
    end_timestamp: Optional[datetime]
    status: str  # 'running', 'completed', 'failed'
    user: str
    
    # Model configuration
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_set: List[str]
    
    # Performance metrics
    metrics: Dict[str, float]  # accuracy, precision, recall, f1, auc
    
    # Artifacts
    model_artifact_path: Optional[str]
    confusion_matrix_path: Optional[str]
    
    # Versioning
    dataset_version: str
    code_version: str
    dependencies: Dict[str, str]
```

### ProductionConfig
```python
@dataclass
class ProductionConfig:
    """Production model configuration."""
    model: ModelConfig
    
@dataclass
class ModelConfig:
    """Model configuration details."""
    algorithm: str
    version: str  # experiment_id
    hyperparameters: Dict[str, Any]
    performance: Dict[str, float]
    tested_date: str
    approved_by: str
```

### HyperparameterTuningJob
```python
@dataclass
class HyperparameterTuningJob:
    """Configuration for hyperparameter tuning."""
    tuning_job_name: str
    algorithm: str
    objective_metric: str  # 'validation:auc', 'validation:accuracy'
    objective_type: str  # 'Maximize', 'Minimize'
    
    hyperparameter_ranges: Dict[str, ParameterRange]
    max_jobs: int
    max_parallel_jobs: int
    strategy: str  # 'Bayesian', 'Random', 'Grid'
    
    training_data_path: str
    validation_data_path: str
    
    status: str
    best_training_job: Optional[str]
    best_hyperparameters: Optional[Dict[str, Any]]
    best_metric_value: Optional[float]

@dataclass
class ParameterRange:
    """Range specification for a hyperparameter."""
    type: str  # 'Integer', 'Continuous', 'Categorical'
    min_value: Optional[float]
    max_value: Optional[float]
    values: Optional[List[Any]]  # For categorical
```

### ABTestConfig
```python
@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    test_id: str
    champion_endpoint: str
    challenger_endpoint: str
    challenger_experiment_id: str
    
    traffic_split: TrafficSplit
    rollout_plan: List[RolloutStage]
    success_criteria: SuccessCriteria
    
    start_date: datetime
    current_stage: int
    status: str  # 'running', 'completed', 'rolled_back'

@dataclass
class TrafficSplit:
    """Traffic distribution between endpoints."""
    champion_percentage: int
    challenger_percentage: int

@dataclass
class RolloutStage:
    """Single stage in gradual rollout."""
    stage: int
    challenger_traffic: int
    duration_hours: int

@dataclass
class SuccessCriteria:
    """Criteria for promoting challenger."""
    min_accuracy: float
    max_latency_ms: float
    min_improvement_pct: float
```

### PromotionEvent
```python
@dataclass
class PromotionEvent:
    """Record of experiment promotion to production."""
    promotion_id: str
    experiment_id: str
    timestamp: datetime
    approver: str
    
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    
    parameter_store_backup_key: str
    config_file_backup_key: str
    
    pipeline_execution_arn: Optional[str]
    status: str  # 'completed', 'failed', 'rolled_back'
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, I identified the following redundancies:
- Requirements 8.1 and 8.4 both test Parameter Store path correctness - combined into Property 1
- Requirements 8.1 and 15.2 both test Parameter Store path consistency - covered by Property 1
- Requirements 9.2 and 9.4 both test config file schema - combined into Property 6
- Multiple requirements test experiment logging completeness - combined into Property 11

The following properties provide unique validation value without redundancy:

### Property 1: Parameter Store Path Correctness
*For any* set of hyperparameters written to Parameter Store, they should appear at the expected paths: `/fraud-detection/hyperparameters/objective`, `/fraud-detection/hyperparameters/num_round`, `/fraud-detection/hyperparameters/max_depth`, `/fraud-detection/hyperparameters/eta`, `/fraud-detection/hyperparameters/subsample`, `/fraud-detection/hyperparameters/colsample_bytree`

**Validates: Requirements 8.1, 8.4, 15.2**

### Property 2: Hyperparameter Validation
*For any* hyperparameters with invalid names or out-of-range values, the validation function should raise a ValueError with descriptive message. *For any* valid hyperparameters, validation should succeed.

**Validates: Requirements 8.2, 14.1**

### Property 3: Parameter Store Backup Creation
*For any* Parameter Store write operation, a backup of previous values should be created in S3 with a timestamp in the key name.

**Validates: Requirements 8.3**

### Property 4: Promotion Event Logging
*For any* successful hyperparameter write to Parameter Store, a corresponding promotion event should be logged to the Experiment Tracker with experiment ID and timestamp.

**Validates: Requirements 8.5**

### Property 5: Configuration File YAML Format
*For any* generated production configuration file, it should be parseable as valid YAML and contain a 'model' key.

**Validates: Requirements 9.1**

### Property 6: Configuration File Completeness
*For any* generated production configuration file, it should contain all required fields: model.algorithm, model.hyperparameters, model.performance, model.tested_date, model.approved_by.

**Validates: Requirements 9.2, 9.4**

### Property 7: Configuration File S3 Location
*For any* configuration file write operation, the file should be retrievable from S3 at `s3://fraud-detection-config/production-model-config.yaml`.

**Validates: Requirements 9.3**

### Property 8: Configuration File Backup Creation
*For any* configuration file write to S3, a versioned backup should be created in `s3://fraud-detection-config/archive/` with a timestamp in the filename.

**Validates: Requirements 9.5**

### Property 9: Pipeline Trigger Metadata
*For any* production pipeline trigger, the Step Functions execution input should contain the experiment ID.

**Validates: Requirements 10.2**

### Property 10: Pipeline Trigger Response
*For any* successful production pipeline trigger, the function should return an execution ARN string starting with "arn:aws:states:".

**Validates: Requirements 10.3**

### Property 11: Experiment Metadata Completeness
*For any* completed experiment, the Experiment Tracker should contain all required fields: experiment_id, start_timestamp, hyperparameters, metrics (accuracy, precision, recall, f1_score, auc_roc), dataset_version, code_version.

**Validates: Requirements 7.2, 3.4, 5.3**

### Property 12: Experiment ID Uniqueness
*For any* two experiments started at different times, their experiment IDs should be different.

**Validates: Requirements 7.1**

### Property 13: Experiment Query Correctness
*For any* experiment query with metric threshold filter (e.g., accuracy >= 0.95), all returned experiments should satisfy the filter condition.

**Validates: Requirements 7.3**

### Property 14: Model Evaluation Metrics Completeness
*For any* model evaluation, the results should contain all required metrics: accuracy, precision, recall, f1_score, auc_roc.

**Validates: Requirements 6.1**

### Property 15: Production Threshold Detection
*For any* model with accuracy >= 0.90, the evaluation results should have meets_production_threshold = True. *For any* model with accuracy < 0.90, it should be False.

**Validates: Requirements 6.5**

### Property 16: Baseline Comparison Completeness
*For any* model evaluation with baseline metrics provided, the comparison results should include difference and percent_change for each metric present in both current and baseline.

**Validates: Requirements 6.4**

### Property 17: Low Accuracy Warning
*For any* experiment with accuracy < 0.80, a warning should be generated indicating results are below production threshold.

**Validates: Requirements 14.5**

### Property 18: Feature Selection Subset
*For any* feature selection operation with k features requested, the returned feature list should have length <= k and all returned features should be from the original feature set.

**Validates: Requirements 5.2**

### Property 19: Grid Search Completeness
*For any* grid search with parameter ranges specified, the number of experiments logged should equal the product of the number of values for each parameter (all combinations tested).

**Validates: Requirements 3.1**

### Property 20: Random Search Sampling
*For any* random search with n_iter trials, exactly n_iter experiments should be logged to the Experiment Tracker.

**Validates: Requirements 3.2**

### Property 21: Challenger Endpoint Naming
*For any* challenger endpoint deployment, the endpoint name should contain both the experiment ID and a timestamp.

**Validates: Requirements 11.2**

### Property 22: Notebook Execution Logging
*For any* notebook execution that trains a model, the Experiment Tracker should contain code_version and dependencies fields for that experiment.

**Validates: Requirements 12.4**

### Property 23: Hyperparameter Round-Trip Consistency
*For any* valid hyperparameters written to Parameter Store by the Experimentation Environment, when read back by the Production Pipeline, the values should be identical.

**Validates: Requirements 15.5**


## Error Handling

### Validation Errors

**Hyperparameter Validation**:
```python
def validate_hyperparameters(hyperparameters):
    """Validate hyperparameter names and value ranges."""
    required_params = ['objective', 'num_round', 'max_depth', 'eta', 'subsample', 'colsample_bytree']
    
    # Check required parameters
    missing = [p for p in required_params if p not in hyperparameters]
    if missing:
        raise ValueError(f"Missing required hyperparameters: {', '.join(missing)}")
    
    # Validate ranges
    validations = {
        'max_depth': (1, 20, int),
        'eta': (0.0, 1.0, float),
        'num_round': (1, 1000, int),
        'subsample': (0.0, 1.0, float),
        'colsample_bytree': (0.0, 1.0, float)
    }
    
    for param, (min_val, max_val, type_func) in validations.items():
        if param in hyperparameters:
            try:
                value = type_func(hyperparameters[param])
                if not (min_val < value <= max_val):
                    raise ValueError(
                        f"Hyperparameter '{param}' value {value} out of valid range "
                        f"({min_val}, {max_val}]"
                    )
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for hyperparameter '{param}': {e}")
```

**Configuration Schema Validation**:
```python
def validate_config_schema(config):
    """Validate production configuration schema."""
    required_structure = {
        'model': {
            'algorithm': str,
            'version': str,
            'hyperparameters': dict,
            'performance': dict,
            'tested_date': str,
            'approved_by': str
        }
    }
    
    if 'model' not in config:
        raise ValueError("Configuration missing required 'model' key")
    
    model = config['model']
    for field, expected_type in required_structure['model'].items():
        if field not in model:
            raise ValueError(f"Configuration missing required field: model.{field}")
        if not isinstance(model[field], expected_type):
            raise ValueError(
                f"Configuration field model.{field} has wrong type. "
                f"Expected {expected_type.__name__}, got {type(model[field]).__name__}"
            )
```

### AWS Service Errors

**S3 Access Errors**:
```python
def load_data_from_s3(bucket, key):
    """Load data from S3 with error handling."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"Data file not found: s3://{bucket}/{key}. "
            f"Ensure the data preparation pipeline has run successfully."
        )
    except s3_client.exceptions.NoSuchBucket:
        raise ValueError(
            f"S3 bucket does not exist: {bucket}. "
            f"Check your AWS configuration and bucket name."
        )
    except ClientError as e:
        if e.response['Error']['Code'] == '403':
            raise PermissionError(
                f"Access denied to s3://{bucket}/{key}. "
                f"Ensure your IAM role has s3:GetObject permission for this bucket."
            )
        else:
            raise RuntimeError(f"S3 error: {e}")
```

**Parameter Store Errors**:
```python
def write_to_parameter_store(param_name, value):
    """Write to Parameter Store with error handling."""
    try:
        ssm_client.put_parameter(
            Name=param_name,
            Value=value,
            Type='String',
            Overwrite=True
        )
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            raise PermissionError(
                f"Access denied writing to Parameter Store: {param_name}. "
                f"Ensure your IAM role has ssm:PutParameter permission. "
                f"Rollback: Previous values backed up at {backup_key}"
            )
        elif error_code == 'ParameterLimitExceeded':
            raise RuntimeError(
                f"Parameter Store limit exceeded. "
                f"Consider cleaning up old parameters or requesting limit increase."
            )
        else:
            raise RuntimeError(
                f"Parameter Store error: {e}. "
                f"Rollback: Restore from backup at {backup_key}"
            )
```

**SageMaker Training Errors**:
```python
def handle_training_failure(training_job_name):
    """Handle SageMaker training job failures."""
    try:
        response = sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        
        status = response['TrainingJobStatus']
        if status == 'Failed':
            failure_reason = response.get('FailureReason', 'Unknown failure')
            
            # Provide actionable guidance based on failure reason
            if 'ResourceLimitExceeded' in failure_reason:
                raise RuntimeError(
                    f"Training failed: Resource limit exceeded. "
                    f"Try reducing instance count or requesting limit increase. "
                    f"Failure reason: {failure_reason}"
                )
            elif 'AlgorithmError' in failure_reason:
                raise RuntimeError(
                    f"Training failed: Algorithm error. "
                    f"Check hyperparameters and data format. "
                    f"Failure reason: {failure_reason}"
                )
            else:
                raise RuntimeError(
                    f"Training failed: {failure_reason}. "
                    f"Check CloudWatch logs for details: /aws/sagemaker/TrainingJobs/{training_job_name}"
                )
    except ClientError as e:
        raise RuntimeError(f"Error describing training job: {e}")
```

### Rollback Procedures

**Parameter Store Rollback**:
```python
def rollback_parameter_store(backup_key):
    """Rollback Parameter Store to previous values from backup."""
    # Load backup from S3
    response = s3_client.get_object(
        Bucket='fraud-detection-config',
        Key=backup_key
    )
    backup = yaml.safe_load(response['Body'].read())
    
    # Restore each parameter
    for param_name, value in backup.items():
        if value is not None:
            ssm_client.put_parameter(
                Name=param_name,
                Value=value,
                Type='String',
                Overwrite=True
            )
            print(f"Restored {param_name} = {value}")
    
    print(f"✓ Rollback complete from backup: {backup_key}")
```

**Configuration File Rollback**:
```python
def rollback_config_file(backup_key):
    """Rollback production config to previous version."""
    # Copy backup to production location
    s3_client.copy_object(
        Bucket='fraud-detection-config',
        CopySource={'Bucket': 'fraud-detection-config', 'Key': backup_key},
        Key='production-model-config.yaml'
    )
    
    print(f"✓ Rolled back config from: {backup_key}")
```


## Testing Strategy

### Dual Testing Approach

This system requires both unit tests and property-based tests for comprehensive coverage:

**Unit Tests**: Verify specific examples, edge cases, and error conditions
- Test specific hyperparameter validation scenarios
- Test S3 access error handling
- Test Parameter Store write failures
- Test configuration file generation with sample data
- Test experiment tracking with mock SageMaker Experiments API

**Property-Based Tests**: Verify universal properties across all inputs
- Test hyperparameter validation across random valid and invalid inputs
- Test Parameter Store path correctness across random hyperparameter sets
- Test configuration file schema across random experiment metadata
- Test experiment ID uniqueness across multiple runs
- Test query filtering across random experiment datasets

Both approaches are complementary and necessary. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across the input space.

### Property-Based Testing Configuration

**Library Selection**: Use `hypothesis` for Python (industry standard for property-based testing)

**Test Configuration**:
```python
from hypothesis import given, settings, strategies as st
import pytest

# Configure for minimum 100 iterations per property test
@settings(max_examples=100)
```

**Example Property Test**:
```python
@given(
    max_depth=st.integers(min_value=1, max_value=20),
    eta=st.floats(min_value=0.01, max_value=1.0),
    num_round=st.integers(min_value=1, max_value=1000),
    subsample=st.floats(min_value=0.1, max_value=1.0),
    colsample_bytree=st.floats(min_value=0.1, max_value=1.0)
)
@settings(max_examples=100)
def test_property_1_parameter_store_path_correctness(
    max_depth, eta, num_round, subsample, colsample_bytree
):
    """
    Feature: ml-experimentation-workflow, Property 1: Parameter Store Path Correctness
    
    For any set of hyperparameters written to Parameter Store, they should appear 
    at the expected paths.
    
    Validates: Requirements 8.1, 8.4, 15.2
    """
    hyperparameters = {
        'objective': 'binary:logistic',
        'max_depth': max_depth,
        'eta': eta,
        'num_round': num_round,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    
    integrator = ProductionIntegrator()
    integrator.write_hyperparameters_to_parameter_store(hyperparameters)
    
    # Verify all parameters at expected paths
    expected_paths = {
        'objective': '/fraud-detection/hyperparameters/objective',
        'max_depth': '/fraud-detection/hyperparameters/max_depth',
        'eta': '/fraud-detection/hyperparameters/eta',
        'num_round': '/fraud-detection/hyperparameters/num_round',
        'subsample': '/fraud-detection/hyperparameters/subsample',
        'colsample_bytree': '/fraud-detection/hyperparameters/colsample_bytree'
    }
    
    for param_name, param_path in expected_paths.items():
        response = ssm_client.get_parameter(Name=param_path)
        actual_value = response['Parameter']['Value']
        expected_value = str(hyperparameters[param_name])
        assert actual_value == expected_value, \
            f"Parameter {param_name} at {param_path}: expected {expected_value}, got {actual_value}"
```

### Test Organization

**Directory Structure**:
```
ml-experimentation-workflow/
├── src/
│   ├── experiment_tracking.py
│   ├── hyperparameter_tuning.py
│   ├── production_integration.py
│   ├── model_evaluation.py
│   └── ab_testing.py
├── tests/
│   ├── unit/
│   │   ├── test_experiment_tracking.py
│   │   ├── test_hyperparameter_tuning.py
│   │   ├── test_production_integration.py
│   │   ├── test_model_evaluation.py
│   │   └── test_ab_testing.py
│   └── property/
│       ├── test_properties_parameter_store.py
│       ├── test_properties_config_files.py
│       ├── test_properties_experiment_tracking.py
│       └── test_properties_validation.py
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_hyperparameter_tuning.ipynb
    ├── 03_algorithm_comparison.ipynb
    ├── 04_feature_engineering.ipynb
    └── 05_production_promotion.ipynb
```

### Property Test Mapping

Each correctness property must be implemented by a single property-based test:

| Property | Test File | Test Function |
|----------|-----------|---------------|
| Property 1: Parameter Store Path Correctness | test_properties_parameter_store.py | test_property_1_parameter_store_path_correctness |
| Property 2: Hyperparameter Validation | test_properties_validation.py | test_property_2_hyperparameter_validation |
| Property 3: Parameter Store Backup Creation | test_properties_parameter_store.py | test_property_3_parameter_store_backup_creation |
| Property 4: Promotion Event Logging | test_properties_experiment_tracking.py | test_property_4_promotion_event_logging |
| Property 5: Configuration File YAML Format | test_properties_config_files.py | test_property_5_configuration_file_yaml_format |
| Property 6: Configuration File Completeness | test_properties_config_files.py | test_property_6_configuration_file_completeness |
| Property 7: Configuration File S3 Location | test_properties_config_files.py | test_property_7_configuration_file_s3_location |
| Property 8: Configuration File Backup Creation | test_properties_config_files.py | test_property_8_configuration_file_backup_creation |
| Property 9: Pipeline Trigger Metadata | test_properties_parameter_store.py | test_property_9_pipeline_trigger_metadata |
| Property 10: Pipeline Trigger Response | test_properties_parameter_store.py | test_property_10_pipeline_trigger_response |
| Property 11: Experiment Metadata Completeness | test_properties_experiment_tracking.py | test_property_11_experiment_metadata_completeness |
| Property 12: Experiment ID Uniqueness | test_properties_experiment_tracking.py | test_property_12_experiment_id_uniqueness |
| Property 13: Experiment Query Correctness | test_properties_experiment_tracking.py | test_property_13_experiment_query_correctness |
| Property 14: Model Evaluation Metrics Completeness | test_properties_validation.py | test_property_14_model_evaluation_metrics_completeness |
| Property 15: Production Threshold Detection | test_properties_validation.py | test_property_15_production_threshold_detection |
| Property 16: Baseline Comparison Completeness | test_properties_validation.py | test_property_16_baseline_comparison_completeness |
| Property 17: Low Accuracy Warning | test_properties_validation.py | test_property_17_low_accuracy_warning |
| Property 18: Feature Selection Subset | test_properties_validation.py | test_property_18_feature_selection_subset |
| Property 19: Grid Search Completeness | test_properties_experiment_tracking.py | test_property_19_grid_search_completeness |
| Property 20: Random Search Sampling | test_properties_experiment_tracking.py | test_property_20_random_search_sampling |
| Property 21: Challenger Endpoint Naming | test_properties_validation.py | test_property_21_challenger_endpoint_naming |
| Property 22: Notebook Execution Logging | test_properties_experiment_tracking.py | test_property_22_notebook_execution_logging |
| Property 23: Hyperparameter Round-Trip Consistency | test_properties_parameter_store.py | test_property_23_hyperparameter_round_trip_consistency |

### Unit Test Coverage

Unit tests should focus on:

1. **Specific Examples**:
   - Test data loading with sample Parquet file
   - Test XGBoost training with known hyperparameters
   - Test confusion matrix generation with sample predictions
   - Test traffic split configuration generation

2. **Edge Cases**:
   - Empty dataset handling
   - Missing S3 files
   - Invalid YAML in config files
   - SageMaker training job failures
   - Parameter Store write failures

3. **Integration Points**:
   - SageMaker Experiments API integration
   - Parameter Store read/write
   - S3 read/write operations
   - Step Functions trigger

### Mocking Strategy

For unit tests, mock AWS services:
```python
import boto3
from moto import mock_s3, mock_ssm, mock_sagemaker, mock_stepfunctions

@mock_s3
@mock_ssm
def test_promote_to_production():
    """Test production promotion with mocked AWS services."""
    # Setup mocks
    s3_client = boto3.client('s3', region_name='us-east-1')
    s3_client.create_bucket(Bucket='fraud-detection-config')
    
    ssm_client = boto3.client('ssm', region_name='us-east-1')
    
    # Test promotion
    integrator = ProductionIntegrator()
    result = integrator.promote_to_production(
        experiment_id='exp-001',
        hyperparameters={'max_depth': 7, 'eta': 0.15},
        metrics={'accuracy': 0.96},
        approver='data-scientist@example.com'
    )
    
    assert result['promotion_event']['experiment_id'] == 'exp-001'
    assert 'backup_key' in result['promotion_event']
```

### Continuous Integration

Run tests on every commit:
```yaml
# .github/workflows/test.yml
name: Test ML Experimentation Workflow

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest hypothesis moto
      - name: Run unit tests
        run: pytest tests/unit/ -v
      - name: Run property tests
        run: pytest tests/property/ -v --hypothesis-show-statistics
```

### Test Execution Time

- Unit tests: ~5 minutes (fast, mocked AWS services)
- Property tests: ~30 minutes (100 iterations per property, 23 properties)
- Total CI time: ~35 minutes

This is acceptable for comprehensive coverage. Property tests can be run less frequently (e.g., nightly) if CI time becomes a concern.
