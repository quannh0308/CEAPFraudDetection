# Fraud Detection ML Pipeline

**Version:** 1.1.0-lambda (Lambda-based Data Preparation)  
**Previous Stable:** 1.0.0-glue (Glue-based - see tag `v1.0.0-glue`)  
**Status:** ✅ Deployed and Tested  
**Date:** February 15, 2026

A production-ready fraud detection system demonstrating the CEAP (Customer Engagement & Action Platform) workflow orchestration framework. The system implements two ML pipelines orchestrated via AWS Step Functions: a weekly training pipeline that builds and deploys fraud detection models to SageMaker, and a daily inference pipeline that scores transactions and alerts on high-risk cases.

> **📌 Version Note:** This version uses **AWS Lambda** for data preparation (10GB memory, 15-min timeout). The original Glue-based version (v1.0.0-glue) is available as a stable milestone for accounts with Glue service quotas enabled. See `MILESTONE-GLUE-VERSION.md` for the Glue implementation.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Components](#system-components)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Deployment Guide](#deployment-guide)
- [Monitoring and Operations](#monitoring-and-operations)
- [Troubleshooting](#troubleshooting)
- [Cost Estimation](#cost-estimation)

## Architecture Overview

The system consists of two independent ML pipelines:

### Training Pipeline (Standard Workflow - Weekly)
Trains fraud detection models on historical transaction data and deploys them to SageMaker endpoints.

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Data Prep│→ │  Train   │→ │ Evaluate │→ │  Deploy  │
│ (Lambda) │  │(SageMaker)│  │ (Lambda) │  │ (Lambda) │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
     ↓              ↓              ↓              ↓
┌────────────────────────────────────────────────────┐
│              S3 Workflow Bucket                     │
│  executions/{id}/DataPrepStage/output.json         │
│  executions/{id}/TrainStage/output.json            │
│  executions/{id}/EvaluateStage/output.json         │
│  executions/{id}/DeployStage/output.json           │
└────────────────────────────────────────────────────┘
```

**Stages:**
1. **DataPrepStage** (Lambda): Loads Kaggle Credit Card Fraud dataset, splits into train/validation/test (70/15/15), writes CSV files
2. **TrainStage** (Lambda): Configures and launches SageMaker XGBoost training job
3. **EvaluateStage** (Lambda): Creates temporary endpoint, evaluates model on test data, validates accuracy >= 0.90
4. **DeployStage** (Lambda): Deploys model to production endpoint, performs health check

### Inference Pipeline (Express Workflow - Daily)
Scores transactions for fraud risk, stores results in DynamoDB, and alerts on high-risk cases.

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Score   │→ │  Store   │→ │  Alert   │→ │ Monitor  │
│ (Lambda) │  │ (Lambda) │  │ (Lambda) │  │ (Lambda) │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
     ↓              ↓              ↓              ↓
┌────────────────────────────────────────────────────┐
│              S3 Workflow Bucket                     │
│  executions/{id}/ScoreStage/output.json            │
│  executions/{id}/StoreStage/output.json            │
│  executions/{id}/AlertStage/output.json            │
│  executions/{id}/MonitorStage/output.json          │
└────────────────────────────────────────────────────┘
```

**Stages:**
1. **ScoreStage** (Lambda): Loads daily transaction batch, invokes SageMaker endpoint for fraud scoring
2. **StoreStage** (Lambda): Writes scored transactions to DynamoDB with batch writes
3. **AlertStage** (Lambda): Filters high-risk transactions (score >= 0.8), publishes alerts to SNS
4. **MonitorStage** (Lambda): Detects distribution drift, writes metrics to S3

### Key Design Principles

- **CEAP Integration**: All Lambda handlers extend `WorkflowLambdaHandler` for S3 orchestration
- **Loose Coupling**: Stages communicate via S3, enabling independent testing and reordering
- **Convention-Based Paths**: S3 paths follow `executions/{executionId}/{stageName}/output.json`
- **Type Safety**: Kotlin implementation with strong typing for data models
- **Observability**: Comprehensive logging and CloudWatch metrics

## System Components

### Lambda Handlers (Kotlin)

All handlers extend the CEAP `WorkflowLambdaHandler` base class:

- **TrainHandler**: Orchestrates SageMaker training jobs
- **EvaluateHandler**: Evaluates model performance on test data
- **DeployHandler**: Deploys models to production endpoints
- **ScoreHandler**: Scores transactions using SageMaker endpoints
- **StoreHandler**: Persists scored transactions to DynamoDB
- **AlertHandler**: Sends SNS alerts for high-risk transactions
- **MonitorHandler**: Detects model drift and performance degradation

### Glue Jobs (Python)

- **data-prep.py**: PySpark script for data preparation, splitting, and Parquet conversion

### AWS Services

- **Step Functions**: Workflow orchestration (Standard for training, Express for inference)
- **SageMaker**: Model training and real-time inference endpoints
- **Lambda**: Serverless compute for workflow stages
- **Glue**: Distributed data processing with PySpark
- **S3**: Stage-to-stage data orchestration and storage
- **DynamoDB**: Persistent storage for scored transactions
- **SNS**: Alerting and notifications
- **EventBridge**: Scheduled workflow triggers
- **CloudWatch**: Logging, metrics, and alarms

## Prerequisites

### Required Software

- **Java 11+**: For Kotlin/Gradle builds
- **Gradle 7.4+**: Build automation
- **Python 3.9+**: For Glue scripts and testing
- **AWS CLI v2**: AWS service interaction
- **AWS CDK 2.x**: Infrastructure deployment

### AWS Account Setup

1. **IAM Permissions**: Ensure your AWS credentials have permissions for:
   - Step Functions (create/update workflows)
   - Lambda (create/update functions)
   - SageMaker (training jobs, endpoints)
   - Glue (jobs, crawlers)
   - S3 (bucket creation, object operations)
   - DynamoDB (table creation, read/write)
   - SNS (topic creation, publish)
   - EventBridge (rule creation)
   - IAM (role creation for services)
   - CloudWatch (logs, metrics, alarms)

2. **AWS Region**: Configure your default region (e.g., `us-east-1`)
   ```bash
   aws configure set region us-east-1
   ```

3. **Account ID**: Note your AWS account ID for bucket naming
   ```bash
   aws sts get-caller-identity --query Account --output text
   ```

### Dataset

Download the Kaggle Credit Card Fraud Detection dataset:
- **Source**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size**: 284,807 transactions (492 fraudulent)
- **Format**: CSV with 30 features (Time, V1-V28, Amount, Class)

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repository-url>
cd CEAPFraudDetection
```

### 2. Initialize CEAP Submodule

```bash
git submodule update --init --recursive
```

### 3. Build Project

```bash
./gradlew clean build
```

This compiles all Kotlin modules and runs tests:
- `fraud-detection-common`: Shared data models and utilities
- `fraud-training-pipeline`: Training pipeline handlers
- `fraud-inference-pipeline`: Inference pipeline handlers
- `infrastructure`: CDK infrastructure stacks

### 4. Run Tests

```bash
# Run all tests
./gradlew test

# Run property-based tests only
./gradlew test --tests "*PropertyTest"

# Run integration tests
./gradlew test --tests "*IntegrationTest"
```

### 5. Upload Dataset to S3

```bash
# Create data bucket with your unique identifier
aws s3 mb s3://fraud-detection-data-quannh0308-20260214

# Upload Kaggle dataset
aws s3 cp kaggle-credit-card-fraud.csv s3://fraud-detection-data-quannh0308-20260214/
```

## Deployment Guide

### Training Pipeline Deployment

The training pipeline uses a Standard workflow with Glue and Lambda stages.

```bash
# Deploy training pipeline infrastructure
./deploy-training-pipeline.sh

# This script performs:
# 1. Builds Gradle project
# 2. Packages Lambda functions as fat JARs
# 3. Uploads Glue script to S3
# 4. Deploys CDK stack (TrainingPipelineStack)
```

**Created Resources:**
- Step Functions workflow: `FraudDetectionTrainingWorkflow`
- Lambda functions: `fraud-detection-dataprep-handler`, `fraud-detection-train-handler`, `fraud-detection-evaluate-handler`, `fraud-detection-deploy-handler`
- S3 buckets: `fraud-detection-workflow-quannh0308-20260214`, `fraud-detection-data-quannh0308-20260214`, `fraud-detection-models-quannh0308-20260214`, `fraud-detection-config-quannh0308-20260214`
- EventBridge rule: Weekly trigger (Sunday 2 AM)
- SNS topic: `fraud-detection-failures`

> **Note:** The Glue job (`fraud-detection-data-prep`) is still created but deprecated. The workflow now uses the Lambda-based DataPrepHandler.

### Inference Pipeline Deployment

The inference pipeline uses an Express workflow with all Lambda stages.

```bash
# Deploy inference pipeline infrastructure
./deploy-inference-pipeline.sh

# This script performs:
# 1. Builds Gradle project
# 2. Packages Lambda functions as fat JARs
# 3. Deploys CDK stack (InferencePipelineStack)
```

**Created Resources:**
- Step Functions workflow: `FraudDetectionInferenceWorkflow`
- Lambda functions: `fraud-detection-score-handler`, `fraud-detection-store-handler`, `fraud-detection-alert-handler`, `fraud-detection-monitor-handler`
- DynamoDB table: `FraudScores` (with `BatchDateIndex` GSI)
- SNS topics: `fraud-detection-alerts`, `fraud-detection-monitoring`
- S3 bucket: `fraud-detection-metrics`
- EventBridge rule: Daily trigger (1 AM)

### Manual Workflow Execution

#### Training Pipeline

```bash
# Start training workflow
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:{account-id}:stateMachine:FraudDetectionTrainingWorkflow \
  --input '{
    "datasetS3Path": "s3://fraud-detection-data-quannh0308-20260214/kaggle-credit-card-fraud.csv",
    "outputPrefix": "s3://fraud-detection-data-quannh0308-20260214/prepared/",
    "trainSplit": 0.70,
    "validationSplit": 0.15,
    "testSplit": 0.15
  }'

# Check execution status
aws stepfunctions describe-execution \
  --execution-arn <execution-arn>
```

#### Inference Pipeline

```bash
# Prepare daily transaction batch
aws s3 cp daily-transactions.json s3://fraud-detection-data-quannh0308-20260214/daily-batches/2024-01-15.json

# Start inference workflow
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:{account-id}:stateMachine:FraudDetectionInferenceWorkflow \
  --input '{
    "transactionBatchPath": "s3://fraud-detection-data-quannh0308-20260214/daily-batches/2024-01-15.json",
    "batchDate": "2024-01-15"
  }'
```

## Monitoring and Operations

### CloudWatch Dashboards

Access pre-configured dashboards:
1. Navigate to CloudWatch → Dashboards
2. Select `FraudDetectionTrainingPipeline` or `FraudDetectionInferencePipeline`

**Key Metrics:**
- Workflow execution count and duration
- Lambda invocation count, duration, errors
- SageMaker training job duration, endpoint latency
- DynamoDB read/write capacity, throttles
- SNS publish success/failure

### CloudWatch Logs

**Log Groups:**
- `/aws/lambda/fraud-detection-train-handler`
- `/aws/lambda/fraud-detection-evaluate-handler`
- `/aws/lambda/fraud-detection-deploy-handler`
- `/aws/lambda/fraud-detection-score-handler`
- `/aws/lambda/fraud-detection-store-handler`
- `/aws/lambda/fraud-detection-alert-handler`
- `/aws/lambda/fraud-detection-monitor-handler`
- `/aws-glue/jobs/fraud-detection-data-prep`

**Log Insights Queries:**

Find failed workflow executions:
```
fields @timestamp, @message
| filter @message like /FAILED/
| sort @timestamp desc
| limit 20
```

Find high-risk transactions:
```
fields @timestamp, transactionId, fraudScore
| filter fraudScore >= 0.8
| sort fraudScore desc
| limit 50
```

### SNS Alert Subscriptions

Subscribe to alert topics:

```bash
# High-risk transaction alerts
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:{account-id}:fraud-detection-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com

# Model drift alerts
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:{account-id}:fraud-detection-monitoring \
  --protocol email \
  --notification-endpoint your-email@example.com

# Pipeline failure alerts
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:{account-id}:fraud-detection-failures \
  --protocol email \
  --notification-endpoint your-email@example.com
```

### DynamoDB Queries

Query scored transactions:

```bash
# Get all transactions for a specific date
aws dynamodb query \
  --table-name FraudScores \
  --index-name BatchDateIndex \
  --key-condition-expression "batchDate = :date" \
  --expression-attribute-values '{":date":{"S":"2024-01-15"}}'

# Get high-risk transactions for a date
aws dynamodb query \
  --table-name FraudScores \
  --index-name BatchDateIndex \
  --key-condition-expression "batchDate = :date AND fraudScore >= :threshold" \
  --expression-attribute-values '{":date":{"S":"2024-01-15"},":threshold":{"N":"0.8"}}'
```

### Model Performance Monitoring

Check daily metrics:

```bash
# Download metrics for a specific date
aws s3 cp s3://fraud-detection-metrics/metrics/2024-01-15.json ./

# View metrics
cat 2024-01-15.json | jq '.'
```

**Metrics Include:**
- `avgFraudScore`: Average fraud score for the batch
- `highRiskPct`: Percentage of high-risk transactions
- `avgScoreDrift`: Deviation from historical baseline
- `driftDetected`: Boolean indicating if drift was detected

## Troubleshooting

### Training Pipeline Issues

#### Issue: Training job fails with "Model accuracy below threshold"

**Cause**: Model accuracy < 0.90 on test dataset

**Resolution:**
1. Check training data quality in S3
2. Review hyperparameters in `TrainHandler.kt`
3. Increase training data size or adjust split ratios
4. Adjust XGBoost hyperparameters (max_depth, eta, num_round)

#### Issue: Glue job fails with "Dataset record count mismatch"

**Cause**: Uploaded dataset doesn't match expected 284,807 records

**Resolution:**
1. Verify dataset integrity: `wc -l kaggle-credit-card-fraud.csv`
2. Re-download dataset from Kaggle
3. Re-upload to S3: `aws s3 cp kaggle-credit-card-fraud.csv s3://fraud-detection-data-quannh0308-20260214/`

#### Issue: Endpoint deployment timeout

**Cause**: SageMaker endpoint creation exceeds 30-minute timeout

**Resolution:**
1. Check SageMaker service limits in AWS Console
2. Verify IAM role has SageMaker permissions
3. Try smaller instance type (ML_T2_MEDIUM instead of ML_M5_LARGE)
4. Check CloudWatch logs for SageMaker errors

### Inference Pipeline Issues

#### Issue: ScoreHandler fails with "Endpoint not found"

**Cause**: Training pipeline hasn't deployed a production endpoint yet

**Resolution:**
1. Run training pipeline first to deploy endpoint
2. Verify endpoint exists: `aws sagemaker list-endpoints`
3. Check `s3://fraud-detection-config/current-endpoint.json` exists

#### Issue: DynamoDB throttling errors

**Cause**: Write capacity exceeded for batch size

**Resolution:**
1. DynamoDB is configured for on-demand mode (auto-scaling)
2. Check CloudWatch metrics for throttling
3. Reduce batch size in transaction input
4. Add exponential backoff retry logic (already implemented in StoreHandler)

#### Issue: No alerts received for high-risk transactions

**Cause**: SNS subscription not confirmed or no high-risk transactions

**Resolution:**
1. Check SNS subscription status: `aws sns list-subscriptions`
2. Confirm email subscription via link in email
3. Verify high-risk transactions exist: Query DynamoDB with `fraudScore >= 0.8`
4. Check CloudWatch logs for AlertHandler errors

### General Issues

#### Issue: S3 access denied (403)

**Cause**: Lambda execution role lacks S3 permissions

**Resolution:**
1. Check IAM role attached to Lambda functions
2. Verify S3 bucket policies allow Lambda access
3. Add S3 permissions to Lambda execution role:
   ```json
   {
     "Effect": "Allow",
     "Action": ["s3:GetObject", "s3:PutObject"],
     "Resource": "arn:aws:s3:::fraud-detection-*/*"
   }
   ```

#### Issue: Previous stage output missing (404)

**Cause**: Previous workflow stage failed or S3 path incorrect

**Resolution:**
1. Check Step Functions execution history for failed stages
2. Verify S3 path convention: `executions/{executionId}/{stageName}/output.json`
3. Check CloudWatch logs for previous stage errors
4. Re-run workflow from beginning

#### Issue: Lambda timeout

**Cause**: Handler processing exceeds configured timeout

**Resolution:**
1. Check Lambda timeout configuration (default: 5 minutes)
2. Increase timeout in CDK stack (max: 15 minutes)
3. Optimize handler logic (batch processing, parallel execution)
4. For long-running tasks, consider Step Functions wait states

## Cost Estimation

### Training Pipeline (Weekly)

| Service | Usage | Cost |
|---------|-------|------|
| Step Functions | 4 stages × weekly | $0.004/week |
| Glue | 0.5 hours × 5 DPUs | $0.22/week |
| SageMaker Training | 1 hour × ML_M5_XLARGE | $0.27/week |
| SageMaker Endpoint | 24/7 × ML_M5_LARGE | $163.68/month |
| Lambda | Negligible | < $0.01/week |
| S3 | 1 GB storage | $0.023/month |
| **Total** | | **~$164/month** |

### Inference Pipeline (Daily)

| Service | Usage | Cost |
|---------|-------|------|
| Step Functions | 4 stages × daily | $0.12/month |
| Lambda | 30 executions × 4 handlers | $6/month |
| DynamoDB | 300K writes/month | $0.38/month |
| SNS | 1K messages/month | < $0.01/month |
| S3 | 0.5 GB storage | $0.012/month |
| **Total** | | **~$6.50/month** |

### Grand Total: ~$170/month

**Cost Optimization Tips:**
- Use SageMaker Serverless Inference for lower traffic (pay per invocation)
- Schedule endpoint shutdown during non-business hours
- Use S3 Intelligent-Tiering for infrequently accessed data
- Enable DynamoDB auto-scaling to match actual load
- Use Lambda reserved concurrency to avoid over-provisioning

---

## Additional Resources

- [CEAP Platform Documentation](./ceap-platform/README.md)
- [Design Document](./.kiro/specs/fraud-detection-ml-pipeline/design.md)
- [Requirements Document](./.kiro/specs/fraud-detection-ml-pipeline/requirements.md)
- [Implementation Tasks](./.kiro/specs/fraud-detection-ml-pipeline/tasks.md)
- [AWS Step Functions Documentation](https://docs.aws.amazon.com/step-functions/)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## License

This project is a demonstration of the CEAP workflow orchestration framework for educational and proof-of-concept purposes.
