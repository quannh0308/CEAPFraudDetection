# Component Catalog: Fraud Detection System

## Overview

This document provides a comprehensive catalog of all components in the fraud detection system, including their inputs, outputs, processing logic, and side effects.

## Table of Contents

1. [Training Pipeline Components](#training-pipeline-components)
2. [Inference Pipeline Components](#inference-pipeline-components)
3. [Experimentation Components](#experimentation-components)
4. [Shared Components](#shared-components)

---

## Training Pipeline Components

### Component 1: DataPrepStage (Glue Job)

**Type**: AWS Glue PySpark Job  
**Execution Time**: 10-30 minutes  
**Compute**: 5 DPUs

#### Input

**Source**: ExecutionContext.initialData (first stage)

**Format**: JSON
```json
{
  "datasetS3Path": "s3://fraud-detection-data-{BUCKET_SUFFIX}/kaggle-credit-card-fraud.csv",
  "outputPrefix": "s3://fraud-detection-data-{BUCKET_SUFFIX}/prepared/",
  "trainSplit": 0.70,
  "validationSplit": 0.15,
  "testSplit": 0.15
}
```

**Schema**:
- `datasetS3Path` (string, required): S3 path to raw CSV dataset
- `outputPrefix` (string, required): S3 prefix for prepared data output
- `trainSplit` (float, required): Training set proportion (0.0-1.0)
- `validationSplit` (float, required): Validation set proportion (0.0-1.0)
- `testSplit` (float, required): Test set proportion (0.0-1.0)

#### Processing

1. **Load Dataset**: Read CSV from S3 using PySpark
2. **Validate Record Count**: Ensure 284,807 records (Kaggle dataset)
3. **Split Data**: Randomly split into train/validation/test sets
4. **Convert Format**: Write to Parquet format for SageMaker
5. **Calculate Statistics**: Record counts per split

**Dependencies**:
- pandas
- scikit-learn
- pyarrow (for Parquet)


#### Output

**Destination**: S3 at `executions/{executionId}/DataPrepStage/output.json`

**Format**: JSON
```json
{
  "trainDataPath": "s3://fraud-detection-data-{BUCKET_SUFFIX}/prepared/train.parquet",
  "validationDataPath": "s3://fraud-detection-data-{BUCKET_SUFFIX}/prepared/validation.parquet",
  "testDataPath": "s3://fraud-detection-data-{BUCKET_SUFFIX}/prepared/test.parquet",
  "recordCounts": {
    "train": 199363,
    "validation": 42721,
    "test": 42723
  },
  "features": ["Time", "V1", "V2", ..., "V28", "Amount"],
  "targetColumn": "Class"
}
```

**Schema**:
- `trainDataPath` (string): S3 path to training data Parquet file
- `validationDataPath` (string): S3 path to validation data Parquet file
- `testDataPath` (string): S3 path to test data Parquet file
- `recordCounts` (object): Record counts for each split
- `features` (array): List of feature column names
- `targetColumn` (string): Name of target/label column

#### Side Effects

- **S3 Writes**: 3 Parquet files written to S3
  - `s3://fraud-detection-data-{BUCKET_SUFFIX}/prepared/train.parquet` (~140 MB)
  - `s3://fraud-detection-data-{BUCKET_SUFFIX}/prepared/validation.parquet` (~30 MB)
  - `s3://fraud-detection-data-{BUCKET_SUFFIX}/prepared/test.parquet` (~30 MB)

- **CloudWatch Logs**: Glue job execution logs

#### Error Conditions

- **Dataset Not Found**: S3 path doesn't exist → Fail with 404 error
- **Invalid Record Count**: Dataset has != 284,807 records → Fail with validation error
- **Split Sum != 1.0**: Train + validation + test splits don't sum to 1.0 → Fail with validation error
- **S3 Write Failure**: Permission denied or throttling → Fail with S3 error

---

### Component 2: TrainHandler (Lambda)

**Type**: AWS Lambda (Kotlin)  
**Execution Time**: 1-2 hours (waiting for SageMaker)  
**Memory**: 512 MB  
**Timeout**: 2 hours

#### Input

**Source**: S3 at `executions/{executionId}/DataPrepStage/output.json`

**Format**: JSON
```json
{
  "trainDataPath": "s3://fraud-detection-data/prepared/train.parquet",
  "validationDataPath": "s3://fraud-detection-data/prepared/validation.parquet",
  "testDataPath": "s3://fraud-detection-data/prepared/test.parquet",
  "recordCounts": { "train": 199363, "validation": 42721, "test": 42723 },
  "features": ["Time", "V1", ..., "V28", "Amount"],
  "targetColumn": "Class"
}
```

**Schema**:
- `trainDataPath` (string, required): S3 path to training data
- `validationDataPath` (string, required): S3 path to validation data
- `testDataPath` (string, optional): S3 path to test data (used by EvaluateStage)
- `recordCounts` (object, optional): Record counts metadata
- `features` (array, optional): Feature names metadata
- `targetColumn` (string, optional): Target column metadata


#### Processing

1. **Extract Data Paths**: Parse trainDataPath and validationDataPath from input
2. **Generate Training Job Name**: Create unique name with timestamp
3. **Configure Training Job**: Build SageMaker CreateTrainingJobRequest with:
   - Algorithm: XGBoost container image
   - Instance: ml.m5.xlarge (1 instance)
   - Input channels: train and validation
   - Output: s3://fraud-detection-models/
   - Hyperparameters: objective, num_round, max_depth, eta, subsample, colsample_bytree
   - Timeout: 1 hour max runtime
4. **Start Training Job**: Call SageMaker CreateTrainingJob API
5. **Wait for Completion**: Poll training job status until COMPLETED or FAILED
6. **Validate Status**: Ensure status is COMPLETED, fail otherwise
7. **Extract Model Artifact**: Get S3 path to model.tar.gz

**Dependencies**:
- AWS SDK for SageMaker
- WorkflowLambdaHandler base class

**Hyperparameters** (currently hardcoded, will be externalized):
```kotlin
mapOf(
    "objective" to "binary:logistic",
    "num_round" to "100",
    "max_depth" to "5",
    "eta" to "0.2",
    "subsample" to "0.8",
    "colsample_bytree" to "0.8"
)
```

#### Output

**Destination**: S3 at `executions/{executionId}/TrainStage/output.json`

**Format**: JSON
```json
{
  "trainingJobName": "fraud-detection-1234567890",
  "modelArtifactPath": "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz",
  "trainingJobStatus": "Completed"
}
```

**Schema**:
- `trainingJobName` (string): Unique SageMaker training job name
- `modelArtifactPath` (string): S3 path to trained model artifact
- `trainingJobStatus` (string): Final training job status ("Completed")

#### Side Effects

- **SageMaker Training Job**: Creates and executes training job
  - Duration: 30-60 minutes
  - Cost: ~$0.27 per job (ml.m5.xlarge × 1 hour)
  
- **S3 Writes**: Model artifact written to S3
  - `s3://fraud-detection-models/{jobName}/output/model.tar.gz` (~50 MB)
  
- **CloudWatch Logs**: Training job logs in `/aws/sagemaker/TrainingJobs`

#### Error Conditions

- **Missing Input Paths**: trainDataPath or validationDataPath not in input → Fail with IllegalArgumentException
- **Training Job Creation Failed**: SageMaker API error → Fail with SageMaker exception
- **Training Job Failed**: Status is FAILED or STOPPED → Fail with IllegalStateException including failure reason
- **Timeout**: Training exceeds 1 hour → Fail with timeout error

---

