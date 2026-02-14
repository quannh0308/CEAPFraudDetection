# Example Configuration Files

This directory contains example configuration files for the Fraud Detection ML Pipeline. These files demonstrate the input/output formats and configuration options for various components of the system.

## Workflow Configurations

### training-workflow-input.json
Example input for starting a training pipeline execution via Step Functions.

**Usage:**
```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:{account-id}:stateMachine:FraudDetectionTrainingWorkflow \
  --input file://examples/training-workflow-input.json
```

**Key Fields:**
- `datasetS3Path`: Location of the Kaggle Credit Card Fraud dataset
- `outputPrefix`: Where prepared datasets will be written
- `trainSplit`, `validationSplit`, `testSplit`: Data split ratios (must sum to 1.0)

### inference-workflow-input.json
Example input for starting an inference pipeline execution via Step Functions.

**Usage:**
```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:{account-id}:stateMachine:FraudDetectionInferenceWorkflow \
  --input file://examples/inference-workflow-input.json
```

**Key Fields:**
- `transactionBatchPath`: S3 path to daily transaction batch
- `batchDate`: Date of the batch (used for DynamoDB indexing)

## Data Formats

### transaction-batch.json
Example format for daily transaction batches that are scored by the inference pipeline.

**Structure:**
- JSON array of Transaction objects
- Each transaction has: `id`, `timestamp`, `amount`, `merchantCategory`, `features`
- Features include: `Time`, `V1`-`V28` (PCA components), `Amount`

**Upload to S3:**
```bash
aws s3 cp examples/transaction-batch.json \
  s3://fraud-detection-data/daily-batches/2024-01-15.json
```

## Lambda Handler Configurations

### lambda-environment-variables.json
Environment variables required by each Lambda handler.

**Handlers:**
- **TrainHandler**: Requires `SAGEMAKER_EXECUTION_ROLE_ARN`
- **EvaluateHandler**: Requires `SAGEMAKER_EXECUTION_ROLE_ARN`
- **DeployHandler**: Requires `SAGEMAKER_EXECUTION_ROLE_ARN`, `CONFIG_BUCKET`
- **ScoreHandler**: Requires `CONFIG_BUCKET`
- **StoreHandler**: Requires `DYNAMODB_TABLE_NAME`
- **AlertHandler**: Requires `FRAUD_ALERT_TOPIC_ARN`
- **MonitorHandler**: Requires `MONITORING_ALERT_TOPIC_ARN`, `METRICS_BUCKET`

**Note:** These are automatically configured by CDK stacks during deployment.

## Glue Job Configuration

### glue-job-parameters.json
Parameters for the Glue data preparation job.

**Key Parameters:**
- `--dataset_s3_path`: Input dataset location
- `--output_prefix`: Output location for prepared datasets
- `--train_split`, `--validation_split`, `--test_split`: Data split ratios

**Glue Configuration:**
- GlueVersion: 4.0 (Python 3.10, Spark 3.3)
- WorkerType: G.1X (1 DPU per worker)
- NumberOfWorkers: 5 (5 DPUs total)
- Timeout: 30 minutes

## Runtime Metadata

### endpoint-metadata.json
Metadata written by DeployHandler and read by ScoreHandler.

**Location:** `s3://fraud-detection-config/current-endpoint.json`

**Purpose:** Tells the inference pipeline which SageMaker endpoint to use for scoring.

**Updated:** Each time DeployHandler successfully deploys a new model (weekly).

### daily-metrics.json
Performance metrics written by MonitorHandler for historical tracking.

**Location:** `s3://fraud-detection-metrics/metrics/{batchDate}.json`

**Purpose:** Track fraud score distribution over time and detect model drift.

**Metrics:**
- `avgFraudScore`: Average fraud score for the batch
- `highRiskPct`, `mediumRiskPct`, `lowRiskPct`: Risk distribution
- `avgScoreDrift`, `highRiskDrift`: Deviation from 30-day baseline
- `driftDetected`: Boolean indicating if drift threshold exceeded

**Drift Detection:**
- Drift is detected if `avgScoreDrift > 0.1` OR `highRiskDrift > 0.05`
- When drift is detected, an SNS alert is sent to the monitoring topic

## Testing with Example Files

### 1. Test Training Pipeline

```bash
# Upload dataset
aws s3 cp kaggle-credit-card-fraud.csv s3://fraud-detection-data/

# Start training workflow
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:{account-id}:stateMachine:FraudDetectionTrainingWorkflow \
  --input file://examples/training-workflow-input.json

# Monitor execution
aws stepfunctions describe-execution --execution-arn <execution-arn>
```

### 2. Test Inference Pipeline

```bash
# Upload transaction batch
aws s3 cp examples/transaction-batch.json \
  s3://fraud-detection-data/daily-batches/2024-01-15.json

# Start inference workflow
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:{account-id}:stateMachine:FraudDetectionInferenceWorkflow \
  --input file://examples/inference-workflow-input.json

# Monitor execution
aws stepfunctions describe-execution --execution-arn <execution-arn>
```

### 3. Query Results

```bash
# Check DynamoDB for scored transactions
aws dynamodb query \
  --table-name FraudScores \
  --index-name BatchDateIndex \
  --key-condition-expression "batchDate = :date" \
  --expression-attribute-values '{":date":{"S":"2024-01-15"}}'

# Check metrics
aws s3 cp s3://fraud-detection-metrics/metrics/2024-01-15.json ./
cat 2024-01-15.json | jq '.'
```

## Customization

### Adjusting Data Split Ratios

Edit `training-workflow-input.json`:
```json
{
  "trainSplit": 0.80,
  "validationSplit": 0.10,
  "testSplit": 0.10
}
```

### Changing Drift Detection Thresholds

Modify `MonitorHandler.kt`:
```kotlin
val driftDetected = avgScoreDrift > 0.15 || highRiskDrift > 0.08
```

### Adjusting High-Risk Threshold

Modify `AlertHandler.kt`:
```kotlin
val highRiskTransactions = scoredTransactions.filter { 
    it.get("fraudScore").asDouble() >= 0.75  // Changed from 0.8
}
```

## Additional Resources

- [Main README](../README.md)
- [Design Document](../.kiro/specs/fraud-detection-ml-pipeline/design.md)
- [Requirements Document](../.kiro/specs/fraud-detection-ml-pipeline/requirements.md)
- [AWS Step Functions Documentation](https://docs.aws.amazon.com/step-functions/)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
