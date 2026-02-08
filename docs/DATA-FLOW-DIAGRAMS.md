# Data Flow Diagrams: Fraud Detection System

## Overview

This document provides detailed data flow diagrams showing how data moves through the fraud detection system, including data transformations, format changes, and component interactions.

---

## Training Pipeline Data Flow

### End-to-End Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE (Weekly Execution)                                       │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: EventBridge Schedule Trigger
  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step Functions: Start Execution                                             │
│ ExecutionContext:                                                            │
│   executionId: "arn:aws:states:...:execution:training:abc123"              │
│   currentStage: "DataPrepStage"                                             │
│   previousStage: null                                                       │
│   workflowBucket: "fraud-detection-workflow-123456789012"                  │
│   initialData: {                                                            │
│     "datasetS3Path": "s3://fraud-detection-data/kaggle-credit-card-fraud.csv",
│     "outputPrefix": "s3://fraud-detection-data/prepared/",                 │
│     "trainSplit": 0.70, "validationSplit": 0.15, "testSplit": 0.15        │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DataPrepStage (Glue Job)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ INPUT (from initialData):                                                   │
│   • Raw CSV: 284,807 transactions                                          │
│   • Format: CSV with 31 columns (Time, V1-V28, Amount, Class)             │
│   • Size: ~150 MB                                                           │
│                                                                              │
│ PROCESSING:                                                                  │
│   1. Load CSV → PySpark DataFrame                                          │
│   2. Validate: 284,807 rows ✓                                              │
│   3. Random split: 70% train, 15% val, 15% test                           │
│   4. Convert to Parquet format                                              │
│   5. Write to S3                                                            │
│                                                                              │
│ OUTPUT (to S3):                                                             │
│   • s3://.../prepared/train.parquet (199,363 rows, ~140 MB)               │
│   • s3://.../prepared/validation.parquet (42,721 rows, ~30 MB)            │
│   • s3://.../prepared/test.parquet (42,723 rows, ~30 MB)                  │
│                                                                              │
│ METADATA (to S3 orchestration):                                            │
│   {                                                                         │
│     "trainDataPath": "s3://.../train.parquet",                            │
│     "validationDataPath": "s3://.../validation.parquet",                  │
│     "testDataPath": "s3://.../test.parquet",                              │
│     "recordCounts": {"train": 199363, "validation": 42721, "test": 42723},│
│     "features": ["Time", "V1", ..., "V28", "Amount"],                     │
│     "targetColumn": "Class"                                                 │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓ S3: executions/{id}/DataPrepStage/output.json
  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: TrainStage (Lambda → SageMaker)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ INPUT (from S3):                                                            │
│   • trainDataPath: "s3://.../train.parquet"                               │
│   • validationDataPath: "s3://.../validation.parquet"                     │
│                                                                              │
│ PROCESSING:                                                                  │
│   1. Lambda reads input from S3                                            │
│   2. Configure SageMaker training job:                                     │
│      - Algorithm: XGBoost (382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest)
│      - Instance: ml.m5.xlarge × 1                                          │
│      - Hyperparameters: {objective: binary:logistic, num_round: 100, ...} │
│      - Input channels: train, validation                                   │
│      - Output: s3://fraud-detection-models/                                │
│   3. Start SageMaker training job                                          │
│   4. Wait for completion (30-60 minutes)                                   │
│   5. Validate status = COMPLETED                                           │
│                                                                              │
│ SIDE EFFECT (SageMaker):                                                    │
│   • Training job executes on ml.m5.xlarge                                  │
│   • Reads Parquet data from S3                                             │
│   • Trains XGBoost model                                                   │
│   • Writes model.tar.gz to S3                                              │
│                                                                              │
│ OUTPUT (to S3):                                                             │
│   • Model artifact: s3://.../fraud-detection-{timestamp}/output/model.tar.gz
│   • Size: ~50 MB                                                            │
│                                                                              │
│ METADATA (to S3 orchestration):                                            │
│   {                                                                         │
│     "trainingJobName": "fraud-detection-1234567890",                       │
│     "modelArtifactPath": "s3://.../model.tar.gz",                         │
│     "trainingJobStatus": "Completed"                                        │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓ S3: executions/{id}/TrainStage/output.json
  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: EvaluateStage (Lambda → SageMaker)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ INPUT (from S3):                                                            │
│   • modelArtifactPath: "s3://.../model.tar.gz"                            │
│   • testDataPath: "s3://.../test.parquet" (from DataPrepStage)           │
│                                                                              │
│ PROCESSING:                                                                  │
│   1. Create temporary SageMaker endpoint for evaluation                   │
│   2. Deploy model to endpoint                                              │
│   3. Wait for endpoint InService                                           │
│   4. Load test data (42,723 transactions)                                 │
│   5. Invoke endpoint for each transaction                                  │
│   6. Calculate metrics: accuracy, precision, recall, F1, AUC              │
│   7. Validate: accuracy >= 0.90 ✓                                         │
│   8. Clean up evaluation endpoint                                          │
│                                                                              │
│ SIDE EFFECT (SageMaker):                                                    │
│   • Temporary endpoint created and deleted                                 │
│   • 42,723 endpoint invocations                                            │
│                                                                              │
│ OUTPUT (to S3):                                                             │
│   {                                                                         │
│     "modelArtifactPath": "s3://.../model.tar.gz",                         │
│     "accuracy": 0.9523,                                                    │
│     "precision": 0.8912,                                                   │
│     "recall": 0.8456,                                                      │
│     "f1Score": 0.8678,                                                     │
│     "auc": 0.9234,                                                         │
│     "testRecordCount": 42723                                               │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓ S3: executions/{id}/EvaluateStage/output.json
  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: DeployStage (Lambda → SageMaker)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ INPUT (from S3):                                                            │
│   • modelArtifactPath: "s3://.../model.tar.gz"                            │
│   • accuracy: 0.9523                                                       │
│   • Other metrics...                                                        │
│                                                                              │
│ PROCESSING:                                                                  │
│   1. Create SageMaker model from artifact                                  │
│   2. Create endpoint configuration                                         │
│   3. Create/update production endpoint: "fraud-detection-prod"            │
│   4. Wait for endpoint InService (5-10 minutes)                           │
│   5. Health check: invoke with test transaction                           │
│   6. Write endpoint metadata to config bucket                             │
│                                                                              │
│ SIDE EFFECT (SageMaker):                                                    │
│   • Production endpoint created/updated                                    │
│   • Endpoint: fraud-detection-prod (ml.m5.large)                          │
│   • Cost: ~$0.23/hour = ~$165/month                                        │
│                                                                              │
│ SIDE EFFECT (S3):                                                           │
│   • s3://fraud-detection-config/current-endpoint.json                     │
│   • Contains: endpoint name, model name, deployment timestamp             │
│                                                                              │
│ OUTPUT (to S3):                                                             │
│   {                                                                         │
│     "endpointName": "fraud-detection-prod",                                │
│     "modelName": "fraud-detection-prod-1234567890",                        │
│     "endpointConfigName": "fraud-detection-prod-1234567890-config",       │
│     "deploymentTimestamp": 1234567890,                                     │
│     "modelAccuracy": 0.9523,                                               │
│     "healthCheckPrediction": 0.0234                                        │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓ S3: executions/{id}/DeployStage/output.json
  ↓
OUTPUT: Training pipeline complete, production endpoint updated
```



---

## Inference Pipeline Data Flow

### End-to-End Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INFERENCE PIPELINE (Daily Execution)                                       │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: EventBridge Schedule Trigger (Daily at 1 AM)
  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step Functions: Start Execution                                             │
│ ExecutionContext:                                                            │
│   executionId: "arn:aws:states:...:execution:inference:xyz789"             │
│   currentStage: "ScoreStage"                                                │
│   previousStage: null                                                       │
│   workflowBucket: "fraud-detection-workflow-123456789012"                  │
│   initialData: {                                                            │
│     "transactionBatchPath": "s3://.../daily-batches/2024-01-15.json",     │
│     "batchDate": "2024-01-15"                                              │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: ScoreStage (Lambda → SageMaker Runtime)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ INPUT (from initialData):                                                   │
│   • transactionBatchPath: "s3://.../2024-01-15.json"                      │
│   • batchDate: "2024-01-15"                                                │
│                                                                              │
│ PROCESSING:                                                                  │
│   1. Read current endpoint name from S3 config                            │
│      → "fraud-detection-prod"                                              │
│   2. Load transaction batch from S3                                        │
│      → 10,000 transactions (example)                                       │
│   3. For each transaction:                                                 │
│      a. Extract features (V1-V28, Time, Amount)                           │
│      b. Invoke SageMaker endpoint                                          │
│      c. Receive fraud score (0.0-1.0)                                     │
│      d. Create ScoredTransaction object                                    │
│   4. Collect all scored transactions                                       │
│                                                                              │
│ TRANSACTION FORMAT:                                                         │
│   Input: {                                                                  │
│     "id": "txn-001",                                                       │
│     "timestamp": 1705334400000,                                            │
│     "amount": 149.62,                                                      │
│     "merchantCategory": "retail",                                          │
│     "features": {"Time": 0.0, "V1": -1.36, ..., "Amount": 149.62}        │
│   }                                                                         │
│   Output: {                                                                 │
│     "transactionId": "txn-001",                                            │
│     "timestamp": 1705334400000,                                            │
│     "amount": 149.62,                                                      │
│     "merchantCategory": "retail",                                          │
│     "features": {...},                                                     │
│     "fraudScore": 0.0234,  ← NEW                                          │
│     "scoringTimestamp": 1705334401000  ← NEW                              │
│   }                                                                         │
│                                                                              │
│ SIDE EFFECT (SageMaker Runtime):                                           │
│   • 10,000 endpoint invocations                                            │
│   • Latency: ~50ms per invocation                                         │
│   • Total time: ~8 minutes                                                 │
│                                                                              │
│ OUTPUT (to S3):                                                             │
│   {                                                                         │
│     "scoredTransactions": [                                                │
│       {"transactionId": "txn-001", "fraudScore": 0.0234, ...},           │
│       {"transactionId": "txn-002", "fraudScore": 0.8912, ...},           │
│       ...                                                                  │
│     ],                                                                     │
│     "batchDate": "2024-01-15",                                            │
│     "transactionCount": 10000,                                            │
│     "endpointName": "fraud-detection-prod"                                │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓ S3: executions/{id}/ScoreStage/output.json
  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: StoreStage (Lambda → DynamoDB)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ INPUT (from S3):                                                            │
│   • scoredTransactions: Array of 10,000 scored transactions               │
│   • batchDate: "2024-01-15"                                                │
│                                                                              │
│ PROCESSING:                                                                  │
│   1. Read scored transactions from S3                                      │
│   2. Batch write to DynamoDB (max 25 items per batch):                    │
│      - 400 batches for 10,000 transactions                                │
│      - Track success/error counts                                          │
│   3. Handle unprocessed items (retry logic)                                │
│   4. Calculate summary statistics:                                         │
│      - High risk: fraudScore >= 0.8                                       │
│      - Medium risk: 0.5 <= fraudScore < 0.8                               │
│      - Low risk: fraudScore < 0.5                                         │
│      - Average fraud score                                                 │
│                                                                              │
│ DYNAMODB ITEM FORMAT:                                                       │
│   {                                                                         │
│     "transactionId": "txn-001",  ← Partition Key                          │
│     "timestamp": 1705334400000,  ← Sort Key                               │
│     "batchDate": "2024-01-15",   ← GSI Partition Key                      │
│     "amount": 149.62,                                                      │
│     "merchantCategory": "retail",                                          │
│     "fraudScore": 0.0234,        ← GSI Sort Key                           │
│     "scoringTimestamp": 1705334401000,                                    │
│     "features": "{\"Time\":0.0,\"V1\":-1.36,...}"  ← JSON string          │
│   }                                                                         │
│                                                                              │
│ SIDE EFFECT (DynamoDB):                                                     │
│   • 10,000 items written to FraudScores table                             │
│   • Table: On-demand billing mode                                          │
│   • GSI: BatchDateIndex (batchDate, fraudScore)                           │
│                                                                              │
│ OUTPUT (to S3):                                                             │
│   {                                                                         │
│     "batchDate": "2024-01-15",                                            │
│     "totalTransactions": 10000,                                            │
│     "successCount": 10000,                                                 │
│     "errorCount": 0,                                                       │
│     "riskDistribution": {                                                  │
│       "highRisk": 520,    ← 5.2% of transactions                          │
│       "mediumRisk": 1500, ← 15% of transactions                           │
│       "lowRisk": 7980     ← 79.8% of transactions                         │
│     },                                                                     │
│     "avgFraudScore": 0.4573                                                │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓ S3: executions/{id}/StoreStage/output.json
  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: AlertStage (Lambda → SNS)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ INPUT (from S3):                                                            │
│   • scoredTransactions: Array of 10,000 scored transactions               │
│     (reads from ScoreStage output, not StoreStage)                        │
│   • batchDate: "2024-01-15"                                                │
│                                                                              │
│ PROCESSING:                                                                  │
│   1. Filter high-risk transactions (fraudScore >= 0.8)                    │
│      → 520 high-risk transactions found                                    │
│   2. Batch alerts (max 100 per SNS message):                              │
│      → 6 batches (5 × 100 + 1 × 20)                                       │
│   3. For each batch:                                                       │
│      a. Build alert message with transaction details                      │
│      b. Publish to SNS topic                                               │
│      c. Include metadata (batchDate, highRiskCount)                       │
│                                                                              │
│ ALERT MESSAGE FORMAT:                                                       │
│   Subject: "Fraud Alert: 100 High-Risk Transactions Detected"             │
│   Body:                                                                     │
│     High-Risk Fraud Transactions Detected                                  │
│     Batch Date: 2024-01-15                                                 │
│     Count: 100                                                             │
│                                                                              │
│     Transactions:                                                           │
│       - ID: txn-002                                                        │
│         Amount: $2500.00                                                   │
│         Fraud Score: 0.8912                                                │
│         Merchant: online                                                   │
│       ...                                                                  │
│                                                                              │
│ SIDE EFFECT (SNS):                                                          │
│   • 6 messages published to fraud-detection-alerts topic                  │
│   • Subscribers: Email, SMS, Lambda (for downstream processing)           │
│                                                                              │
│ OUTPUT (to S3):                                                             │
│   {                                                                         │
│     "batchDate": "2024-01-15",                                            │
│     "highRiskCount": 520,                                                  │
│     "alertsSent": 520,                                                     │
│     "alertBatches": 6                                                      │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓ S3: executions/{id}/AlertStage/output.json
  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: MonitorStage (Lambda → SNS + S3)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ INPUT (from S3):                                                            │
│   • batchDate: "2024-01-15"                                                │
│   • totalTransactions: 10000                                               │
│   • riskDistribution: {highRisk: 520, mediumRisk: 1500, lowRisk: 7980}   │
│   • avgFraudScore: 0.4573                                                  │
│                                                                              │
│ PROCESSING:                                                                  │
│   1. Load historical baseline (last 30 days average):                     │
│      → avgFraudScore: 0.4339                                              │
│      → highRiskPct: 0.041 (4.1%)                                          │
│   2. Calculate current metrics:                                            │
│      → highRiskPct: 520/10000 = 0.052 (5.2%)                             │
│      → mediumRiskPct: 1500/10000 = 0.15 (15%)                            │
│      → lowRiskPct: 7980/10000 = 0.798 (79.8%)                            │
│   3. Detect drift:                                                         │
│      → avgScoreDrift: |0.4573 - 0.4339| = 0.0234 < 0.1 ✓                │
│      → highRiskDrift: |0.052 - 0.041| = 0.011 < 0.05 ✓                  │
│      → driftDetected: false                                                │
│   4. Write metrics to S3 for historical tracking                          │
│                                                                              │
│ DRIFT DETECTION LOGIC:                                                      │
│   IF avgScoreDrift > 0.1 OR highRiskDrift > 0.05:                        │
│     driftDetected = true                                                   │
│     Send monitoring alert via SNS                                          │
│                                                                              │
│ SIDE EFFECT (S3):                                                           │
│   • s3://fraud-detection-metrics/metrics/2024-01-15.json                  │
│   • Retained for 90 days for trend analysis                               │
│                                                                              │
│ SIDE EFFECT (SNS - if drift detected):                                     │
│   • Message to fraud-detection-monitoring topic                           │
│   • Subject: "Model Drift Detected: 2024-01-15"                           │
│                                                                              │
│ OUTPUT (to S3):                                                             │
│   {                                                                         │
│     "batchDate": "2024-01-15",                                            │
│     "avgFraudScore": 0.4573,                                               │
│     "avgScoreDrift": 0.0234,                                               │
│     "highRiskDrift": 0.011,                                                │
│     "driftDetected": false,                                                │
│     "metricsPath": "metrics/2024-01-15.json"                              │
│   }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
  ↓ S3: executions/{id}/MonitorStage/output.json
  ↓
OUTPUT: Inference pipeline complete, 10,000 transactions scored and stored
```

