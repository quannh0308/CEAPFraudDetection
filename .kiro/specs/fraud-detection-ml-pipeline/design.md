# Design Document: Fraud Detection ML Pipeline

## Overview

This document specifies the design for a fraud detection system that demonstrates the CEAP (Customer Engagement & Action Platform) workflow orchestration framework. The system implements two ML pipelines orchestrated via AWS Step Functions:

1. **Training Pipeline** (Standard Workflow): Weekly execution that trains fraud detection models on historical transaction data and deploys them to SageMaker endpoints
2. **Inference Pipeline** (Express Workflow): Daily execution that scores transactions for fraud risk, stores results in DynamoDB, and alerts on high-risk cases

The implementation serves as a proof-of-work for clients building ML workflows using CEAP infrastructure, demonstrating:
- S3-based stage orchestration with WorkflowLambdaHandler pattern
- Integration of SageMaker for ML training and inference
- Mixed Lambda/Glue job workflows for data processing
- DynamoDB for persistent storage
- SNS for alerting and notifications

### Key Design Principles

1. **CEAP Integration**: All Lambda handlers extend WorkflowLambdaHandler for S3 orchestration
2. **Loose Coupling**: Stages communicate via S3, enabling independent testing and reordering
3. **Convention-Based Paths**: S3 paths follow `executions/{executionId}/{stageName}/output.json`
4. **Type Safety**: Kotlin implementation with strong typing for data models
5. **Observability**: Comprehensive logging and CloudWatch metrics
6. **Error Handling**: Graceful degradation with retry logic and failure notifications


## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Fraud Detection ML System                         │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Training Pipeline (Weekly)                  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  │
│  │  │ Data Prep│→ │  Train   │→ │ Evaluate │→ │  Deploy  │     │  │
│  │  │  (Glue)  │  │(SageMaker)│  │ (Lambda) │  │ (Lambda) │     │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │  │
│  │       ↓              ↓              ↓              ↓          │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │              S3 Workflow Bucket                         │  │  │
│  │  │  executions/{id}/DataPrepStage/output.json            │  │  │
│  │  │  executions/{id}/TrainStage/output.json               │  │  │
│  │  │  executions/{id}/EvaluateStage/output.json            │  │  │
│  │  │  executions/{id}/DeployStage/output.json              │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Inference Pipeline (Daily)                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  │
│  │  │  Score   │→ │  Store   │→ │  Alert   │→ │ Monitor  │     │  │
│  │  │ (Lambda) │  │ (Lambda) │  │ (Lambda) │  │ (Lambda) │     │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │  │
│  │       ↓              ↓              ↓              ↓          │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │              S3 Workflow Bucket                         │  │  │
│  │  │  executions/{id}/ScoreStage/output.json               │  │  │
│  │  │  executions/{id}/StoreStage/output.json               │  │  │
│  │  │  executions/{id}/AlertStage/output.json               │  │  │
│  │  │  executions/{id}/MonitorStage/output.json             │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  External Services:                                                  │
│  - SageMaker: Model training and inference endpoints                │
│  - DynamoDB: Fraud score storage                                    │
│  - SNS: Alerts and failure notifications                            │
│  - EventBridge: Scheduled triggers (weekly/daily)                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Workflow Types

**Training Pipeline**: Standard Workflow
- Execution time: 2-4 hours (model training is long-running)
- Includes Glue job for data preparation (PySpark processing)
- Asynchronous execution with EventBridge monitoring
- Cost: ~$25 per million transitions (4 stages × weekly = ~$0.004/week)

**Inference Pipeline**: Express Workflow
- Execution time: 5-30 minutes (fast batch scoring)
- All Lambda stages (no Glue jobs)
- Synchronous execution with immediate feedback
- Cost: ~$1 per million transitions (4 stages × daily = ~$0.12/month)


## Components and Interfaces

### Training Pipeline Components

#### 1. DataPrepStage (Glue Job)

**Purpose**: Prepare historical transaction data for model training

**Input** (from initialData):
```json
{
  "datasetS3Path": "s3://fraud-detection-data/kaggle-credit-card-fraud.csv",
  "outputPrefix": "s3://fraud-detection-data/prepared/",
  "trainSplit": 0.70,
  "validationSplit": 0.15,
  "testSplit": 0.15
}
```

**Processing**:
- Load Kaggle Credit Card Fraud Detection dataset (284,807 transactions)
- Validate record count matches expected 284,807
- Split data into train (70%), validation (15%), test (15%) sets
- Write prepared datasets to S3 in Parquet format for SageMaker

**Output** (to S3):
```json
{
  "trainDataPath": "s3://fraud-detection-data/prepared/train.parquet",
  "validationDataPath": "s3://fraud-detection-data/prepared/validation.parquet",
  "testDataPath": "s3://fraud-detection-data/prepared/test.parquet",
  "recordCounts": {
    "train": 199363,
    "validation": 42721,
    "test": 42723
  },
  "features": ["Time", "V1", "V2", ..., "V28", "Amount"],
  "targetColumn": "Class"
}
```

**Implementation**: PySpark script using AWS Glue
- DPU allocation: 5 DPUs (sufficient for ~300K records)
- Timeout: 30 minutes
- Libraries: pandas, scikit-learn for data splitting


#### 2. TrainStage (Lambda Handler)

**Purpose**: Configure and launch SageMaker training job

**Handler Class**: `TrainHandler : WorkflowLambdaHandler()`

**Input** (from DataPrepStage S3 output):
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

**Processing**:
```kotlin
override fun processData(input: JsonNode): JsonNode {
    // 1. Extract data paths from input
    val trainDataPath = input.get("trainDataPath").asText()
    val validationDataPath = input.get("validationDataPath").asText()
    
    // 2. Configure SageMaker training job
    val trainingJobName = "fraud-detection-${System.currentTimeMillis()}"
    val trainingJobRequest = CreateTrainingJobRequest.builder()
        .trainingJobName(trainingJobName)
        .algorithmSpecification(AlgorithmSpecification.builder()
            .trainingImage("382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest")
            .trainingInputMode(TrainingInputMode.FILE)
            .build())
        .roleArn(sageMakerExecutionRoleArn)
        .inputDataConfig(listOf(
            Channel.builder()
                .channelName("train")
                .dataSource(DataSource.builder()
                    .s3DataSource(S3DataSource.builder()
                        .s3Uri(trainDataPath)
                        .s3DataType(S3DataType.S3_PREFIX)
                        .build())
                    .build())
                .contentType("application/x-parquet")
                .build(),
            Channel.builder()
                .channelName("validation")
                .dataSource(DataSource.builder()
                    .s3DataSource(S3DataSource.builder()
                        .s3Uri(validationDataPath)
                        .s3DataType(S3DataType.S3_PREFIX)
                        .build())
                    .build())
                .contentType("application/x-parquet")
                .build()
        ))
        .outputDataConfig(OutputDataConfig.builder()
            .s3OutputPath("s3://fraud-detection-models/")
            .build())
        .resourceConfig(ResourceConfig.builder()
            .instanceType(TrainingInstanceType.ML_M5_XLARGE)
            .instanceCount(1)
            .volumeSizeInGB(30)
            .build())
        .stoppingCondition(StoppingCondition.builder()
            .maxRuntimeInSeconds(3600) // 1 hour max
            .build())
        .hyperParameters(mapOf(
            "objective" to "binary:logistic",
            "num_round" to "100",
            "max_depth" to "5",
            "eta" to "0.2",
            "subsample" to "0.8",
            "colsample_bytree" to "0.8"
        ))
        .build()
    
    // 3. Start training job
    sageMakerClient.createTrainingJob(trainingJobRequest)
    
    // 4. Wait for training completion (with timeout)
    val waiter = sageMakerClient.waiter()
    val waiterResponse = waiter.waitUntilTrainingJobCompletedOrStopped(
        DescribeTrainingJobRequest.builder()
            .trainingJobName(trainingJobName)
            .build(),
        WaiterOverrideConfiguration.builder()
            .maxAttempts(120) // 120 attempts × 30s = 1 hour
            .build()
    )
    
    // 5. Check training status
    val trainingJobStatus = sageMakerClient.describeTrainingJob(
        DescribeTrainingJobRequest.builder()
            .trainingJobName(trainingJobName)
            .build()
    ).trainingJobStatus()
    
    if (trainingJobStatus != TrainingJobStatus.COMPLETED) {
        throw IllegalStateException("Training job failed: $trainingJobStatus")
    }
    
    // 6. Return training job metadata
    return objectMapper.createObjectNode().apply {
        put("trainingJobName", trainingJobName)
        put("modelArtifactPath", "s3://fraud-detection-models/$trainingJobName/output/model.tar.gz")
        put("trainingJobStatus", trainingJobStatus.toString())
    }
}
```

**Output** (to S3):
```json
{
  "trainingJobName": "fraud-detection-1234567890",
  "modelArtifactPath": "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz",
  "trainingJobStatus": "Completed"
}
```


#### 3. EvaluateStage (Lambda Handler)

**Purpose**: Evaluate trained model on test dataset and validate performance

**Handler Class**: `EvaluateHandler : WorkflowLambdaHandler()`

**Input** (from TrainStage S3 output):
```json
{
  "trainingJobName": "fraud-detection-1234567890",
  "modelArtifactPath": "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz",
  "trainingJobStatus": "Completed"
}
```

**Processing**:
```kotlin
override fun processData(input: JsonNode): JsonNode {
    // 1. Create temporary SageMaker endpoint for evaluation
    val modelName = "fraud-detection-eval-${System.currentTimeMillis()}"
    val endpointConfigName = "$modelName-config"
    val endpointName = "$modelName-endpoint"
    
    // 2. Create model
    sageMakerClient.createModel(CreateModelRequest.builder()
        .modelName(modelName)
        .primaryContainer(ContainerDefinition.builder()
            .image("382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest")
            .modelDataUrl(input.get("modelArtifactPath").asText())
            .build())
        .executionRoleArn(sageMakerExecutionRoleArn)
        .build())
    
    // 3. Create endpoint configuration
    sageMakerClient.createEndpointConfig(CreateEndpointConfigRequest.builder()
        .endpointConfigName(endpointConfigName)
        .productionVariants(listOf(
            ProductionVariant.builder()
                .variantName("AllTraffic")
                .modelName(modelName)
                .instanceType(ProductionVariantInstanceType.ML_M5_LARGE)
                .initialInstanceCount(1)
                .build()
        ))
        .build())
    
    // 4. Create endpoint
    sageMakerClient.createEndpoint(CreateEndpointRequest.builder()
        .endpointName(endpointName)
        .endpointConfigName(endpointConfigName)
        .build())
    
    // 5. Wait for endpoint to be in service
    val waiter = sageMakerClient.waiter()
    waiter.waitUntilEndpointInService(
        DescribeEndpointRequest.builder()
            .endpointName(endpointName)
            .build()
    )
    
    // 6. Load test data and run predictions
    val testDataPath = readTestDataPathFromPreviousStages()
    val testData = loadTestData(testDataPath)
    val predictions = mutableListOf<Double>()
    val actuals = mutableListOf<Int>()
    
    testData.forEach { record ->
        val prediction = invokeEndpoint(endpointName, record.features)
        predictions.add(prediction)
        actuals.add(record.label)
    }
    
    // 7. Calculate metrics
    val accuracy = calculateAccuracy(predictions, actuals)
    val precision = calculatePrecision(predictions, actuals)
    val recall = calculateRecall(predictions, actuals)
    val f1Score = calculateF1Score(precision, recall)
    val auc = calculateAUC(predictions, actuals)
    
    // 8. Validate model meets minimum accuracy threshold
    if (accuracy < 0.90) {
        throw IllegalStateException(
            "Model accuracy $accuracy is below minimum threshold 0.90. " +
            "Training failed to produce acceptable model."
        )
    }
    
    // 9. Clean up evaluation endpoint
    sageMakerClient.deleteEndpoint(DeleteEndpointRequest.builder()
        .endpointName(endpointName)
        .build())
    sageMakerClient.deleteEndpointConfig(DeleteEndpointConfigRequest.builder()
        .endpointConfigName(endpointConfigName)
        .build())
    sageMakerClient.deleteModel(DeleteModelRequest.builder()
        .modelName(modelName)
        .build())
    
    // 10. Return evaluation metrics
    return objectMapper.createObjectNode().apply {
        put("modelArtifactPath", input.get("modelArtifactPath").asText())
        put("accuracy", accuracy)
        put("precision", precision)
        put("recall", recall)
        put("f1Score", f1Score)
        put("auc", auc)
        put("testRecordCount", testData.size)
    }
}
```

**Output** (to S3):
```json
{
  "modelArtifactPath": "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz",
  "accuracy": 0.9523,
  "precision": 0.8912,
  "recall": 0.8456,
  "f1Score": 0.8678,
  "auc": 0.9234,
  "testRecordCount": 42723
}
```


#### 4. DeployStage (Lambda Handler)

**Purpose**: Deploy trained model to production SageMaker endpoint

**Handler Class**: `DeployHandler : WorkflowLambdaHandler()`

**Input** (from EvaluateStage S3 output):
```json
{
  "modelArtifactPath": "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz",
  "accuracy": 0.9523,
  "precision": 0.8912,
  "recall": 0.8456,
  "f1Score": 0.8678,
  "auc": 0.9234,
  "testRecordCount": 42723
}
```

**Processing**:
```kotlin
override fun processData(input: JsonNode): JsonNode {
    val timestamp = System.currentTimeMillis()
    val modelName = "fraud-detection-prod-$timestamp"
    val endpointConfigName = "$modelName-config"
    val endpointName = "fraud-detection-prod" // Fixed name for inference pipeline
    
    // 1. Create model
    sageMakerClient.createModel(CreateModelRequest.builder()
        .modelName(modelName)
        .primaryContainer(ContainerDefinition.builder()
            .image("382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest")
            .modelDataUrl(input.get("modelArtifactPath").asText())
            .build())
        .executionRoleArn(sageMakerExecutionRoleArn)
        .build())
    
    // 2. Create endpoint configuration
    sageMakerClient.createEndpointConfig(CreateEndpointConfigRequest.builder()
        .endpointConfigName(endpointConfigName)
        .productionVariants(listOf(
            ProductionVariant.builder()
                .variantName("AllTraffic")
                .modelName(modelName)
                .instanceType(ProductionVariantInstanceType.ML_M5_LARGE)
                .initialInstanceCount(1)
                .initialVariantWeight(1.0f)
                .build()
        ))
        .build())
    
    // 3. Check if endpoint exists
    val endpointExists = try {
        sageMakerClient.describeEndpoint(DescribeEndpointRequest.builder()
            .endpointName(endpointName)
            .build())
        true
    } catch (e: ResourceNotFoundException) {
        false
    }
    
    // 4. Create or update endpoint
    if (endpointExists) {
        // Update existing endpoint
        sageMakerClient.updateEndpoint(UpdateEndpointRequest.builder()
            .endpointName(endpointName)
            .endpointConfigName(endpointConfigName)
            .build())
    } else {
        // Create new endpoint
        sageMakerClient.createEndpoint(CreateEndpointRequest.builder()
            .endpointName(endpointName)
            .endpointConfigName(endpointConfigName)
            .build())
    }
    
    // 5. Wait for endpoint to be in service
    val waiter = sageMakerClient.waiter()
    waiter.waitUntilEndpointInService(
        DescribeEndpointRequest.builder()
            .endpointName(endpointName)
            .build(),
        WaiterOverrideConfiguration.builder()
            .maxAttempts(60) // 60 attempts × 30s = 30 minutes
            .build()
    )
    
    // 6. Validate endpoint health with test transaction
    val testTransaction = mapOf(
        "Time" to 0.0,
        "V1" to -1.3598071336738,
        "V2" to -0.0727811733098497,
        // ... other features
        "Amount" to 149.62
    )
    
    val testPrediction = invokeEndpoint(endpointName, testTransaction)
    logger.info("Endpoint health check: prediction=$testPrediction")
    
    // 7. Write endpoint metadata to S3 for inference pipeline
    val endpointMetadata = objectMapper.createObjectNode().apply {
        put("endpointName", endpointName)
        put("modelName", modelName)
        put("deploymentTimestamp", timestamp)
    }
    
    s3Client.putObject(
        PutObjectRequest.builder()
            .bucket("fraud-detection-config")
            .key("current-endpoint.json")
            .build(),
        RequestBody.fromString(objectMapper.writeValueAsString(endpointMetadata))
    )
    
    // 8. Return deployment metadata
    return objectMapper.createObjectNode().apply {
        put("endpointName", endpointName)
        put("modelName", modelName)
        put("endpointConfigName", endpointConfigName)
        put("deploymentTimestamp", timestamp)
        put("modelAccuracy", input.get("accuracy").asDouble())
        put("healthCheckPrediction", testPrediction)
    }
}
```

**Output** (to S3):
```json
{
  "endpointName": "fraud-detection-prod",
  "modelName": "fraud-detection-prod-1234567890",
  "endpointConfigName": "fraud-detection-prod-1234567890-config",
  "deploymentTimestamp": 1234567890,
  "modelAccuracy": 0.9523,
  "healthCheckPrediction": 0.0234
}
```


### Inference Pipeline Components

#### 1. ScoreStage (Lambda Handler)

**Purpose**: Score daily transaction batch using deployed SageMaker endpoint

**Handler Class**: `ScoreHandler : WorkflowLambdaHandler()`

**Input** (from initialData):
```json
{
  "transactionBatchPath": "s3://fraud-detection-data/daily-batches/2024-01-15.json",
  "batchDate": "2024-01-15"
}
```

**Processing**:
```kotlin
override fun processData(input: JsonNode): JsonNode {
    // 1. Read current endpoint name from S3
    val endpointMetadata = s3Client.getObject(
        GetObjectRequest.builder()
            .bucket("fraud-detection-config")
            .key("current-endpoint.json")
            .build()
    ).readAllBytes()
    
    val endpointName = objectMapper.readTree(endpointMetadata)
        .get("endpointName").asText()
    
    // 2. Load transaction batch
    val transactionBatchPath = input.get("transactionBatchPath").asText()
    val transactionBatch = loadTransactionBatch(transactionBatchPath)
    
    logger.info("Scoring ${transactionBatch.size} transactions using endpoint $endpointName")
    
    // 3. Score each transaction
    val scoredTransactions = transactionBatch.map { transaction ->
        val features = extractFeatures(transaction)
        val fraudScore = invokeEndpoint(endpointName, features)
        
        ScoredTransaction(
            transactionId = transaction.id,
            timestamp = transaction.timestamp,
            amount = transaction.amount,
            merchantCategory = transaction.merchantCategory,
            features = features,
            fraudScore = fraudScore,
            scoringTimestamp = System.currentTimeMillis()
        )
    }
    
    // 4. Return scored transactions
    return objectMapper.createObjectNode().apply {
        set<ArrayNode>("scoredTransactions", objectMapper.valueToTree(scoredTransactions))
        put("batchDate", input.get("batchDate").asText())
        put("transactionCount", scoredTransactions.size)
        put("endpointName", endpointName)
    }
}

private fun invokeEndpoint(endpointName: String, features: Map<String, Double>): Double {
    val payload = objectMapper.writeValueAsString(features)
    
    val response = sageMakerRuntimeClient.invokeEndpoint(
        InvokeEndpointRequest.builder()
            .endpointName(endpointName)
            .contentType("application/json")
            .body(SdkBytes.fromUtf8String(payload))
            .build()
    )
    
    val prediction = objectMapper.readTree(response.body().asUtf8String())
    return prediction.asDouble()
}
```

**Output** (to S3):
```json
{
  "scoredTransactions": [
    {
      "transactionId": "txn-001",
      "timestamp": 1705334400000,
      "amount": 149.62,
      "merchantCategory": "retail",
      "features": { "Time": 0.0, "V1": -1.36, "Amount": 149.62 },
      "fraudScore": 0.0234,
      "scoringTimestamp": 1705334401000
    },
    {
      "transactionId": "txn-002",
      "timestamp": 1705334500000,
      "amount": 2500.00,
      "merchantCategory": "online",
      "features": { "Time": 100.0, "V1": 2.45, "Amount": 2500.00 },
      "fraudScore": 0.8912,
      "scoringTimestamp": 1705334501000
    }
  ],
  "batchDate": "2024-01-15",
  "transactionCount": 2,
  "endpointName": "fraud-detection-prod"
}
```


#### 2. StoreStage (Lambda Handler)

**Purpose**: Store scored transactions in DynamoDB for querying and analysis

**Handler Class**: `StoreHandler : WorkflowLambdaHandler()`

**Input** (from ScoreStage S3 output):
```json
{
  "scoredTransactions": [ /* array of scored transactions */ ],
  "batchDate": "2024-01-15",
  "transactionCount": 2,
  "endpointName": "fraud-detection-prod"
}
```

**Processing**:
```kotlin
override fun processData(input: JsonNode): JsonNode {
    val scoredTransactions = input.get("scoredTransactions")
    val batchDate = input.get("batchDate").asText()
    
    // 1. Batch write to DynamoDB (max 25 items per batch)
    val writeRequests = mutableListOf<WriteRequest>()
    var successCount = 0
    var errorCount = 0
    
    scoredTransactions.forEach { transactionNode ->
        val transaction = objectMapper.treeToValue(transactionNode, ScoredTransaction::class.java)
        
        val item = mapOf(
            "transactionId" to AttributeValue.builder().s(transaction.transactionId).build(),
            "timestamp" to AttributeValue.builder().n(transaction.timestamp.toString()).build(),
            "batchDate" to AttributeValue.builder().s(batchDate).build(),
            "amount" to AttributeValue.builder().n(transaction.amount.toString()).build(),
            "merchantCategory" to AttributeValue.builder().s(transaction.merchantCategory).build(),
            "fraudScore" to AttributeValue.builder().n(transaction.fraudScore.toString()).build(),
            "scoringTimestamp" to AttributeValue.builder().n(transaction.scoringTimestamp.toString()).build(),
            "features" to AttributeValue.builder().s(
                objectMapper.writeValueAsString(transaction.features)
            ).build()
        )
        
        writeRequests.add(
            WriteRequest.builder()
                .putRequest(PutRequest.builder().item(item).build())
                .build()
        )
        
        // Batch write when we reach 25 items
        if (writeRequests.size == 25) {
            val result = batchWriteToDynamoDB(writeRequests)
            successCount += result.successCount
            errorCount += result.errorCount
            writeRequests.clear()
        }
    }
    
    // Write remaining items
    if (writeRequests.isNotEmpty()) {
        val result = batchWriteToDynamoDB(writeRequests)
        successCount += result.successCount
        errorCount += result.errorCount
    }
    
    // 2. Calculate summary statistics
    val fraudScores = scoredTransactions.map { it.get("fraudScore").asDouble() }
    val highRiskCount = fraudScores.count { it >= 0.8 }
    val mediumRiskCount = fraudScores.count { it >= 0.5 && it < 0.8 }
    val lowRiskCount = fraudScores.count { it < 0.5 }
    val avgFraudScore = fraudScores.average()
    
    // 3. Return storage summary
    return objectMapper.createObjectNode().apply {
        put("batchDate", batchDate)
        put("totalTransactions", scoredTransactions.size())
        put("successCount", successCount)
        put("errorCount", errorCount)
        set<ObjectNode>("riskDistribution", objectMapper.createObjectNode().apply {
            put("highRisk", highRiskCount)
            put("mediumRisk", mediumRiskCount)
            put("lowRisk", lowRiskCount)
        })
        put("avgFraudScore", avgFraudScore)
    }
}

private fun batchWriteToDynamoDB(writeRequests: List<WriteRequest>): BatchWriteResult {
    val response = dynamoDbClient.batchWriteItem(
        BatchWriteItemRequest.builder()
            .requestItems(mapOf("FraudScores" to writeRequests))
            .build()
    )
    
    val unprocessedCount = response.unprocessedItems()["FraudScores"]?.size ?: 0
    return BatchWriteResult(
        successCount = writeRequests.size - unprocessedCount,
        errorCount = unprocessedCount
    )
}
```

**Output** (to S3):
```json
{
  "batchDate": "2024-01-15",
  "totalTransactions": 2,
  "successCount": 2,
  "errorCount": 0,
  "riskDistribution": {
    "highRisk": 1,
    "mediumRisk": 0,
    "lowRisk": 1
  },
  "avgFraudScore": 0.4573
}
```


#### 3. AlertStage (Lambda Handler)

**Purpose**: Identify high-risk transactions and send alerts via SNS

**Handler Class**: `AlertHandler : WorkflowLambdaHandler()`

**Input** (from ScoreStage S3 output - reads directly from ScoreStage, not StoreStage):
```json
{
  "scoredTransactions": [ /* array of scored transactions */ ],
  "batchDate": "2024-01-15",
  "transactionCount": 2,
  "endpointName": "fraud-detection-prod"
}
```

**Processing**:
```kotlin
override fun processData(input: JsonNode): JsonNode {
    val scoredTransactions = input.get("scoredTransactions")
    val batchDate = input.get("batchDate").asText()
    
    // 1. Filter high-risk transactions (fraud score >= 0.8)
    val highRiskTransactions = scoredTransactions.filter { 
        it.get("fraudScore").asDouble() >= 0.8 
    }
    
    if (highRiskTransactions.isEmpty()) {
        logger.info("No high-risk transactions found for batch $batchDate")
        return objectMapper.createObjectNode().apply {
            put("batchDate", batchDate)
            put("highRiskCount", 0)
            put("alertsSent", 0)
        }
    }
    
    // 2. Batch alerts to avoid SNS rate limits (max 100 per message)
    val alertBatches = highRiskTransactions.chunked(100)
    var alertsSent = 0
    
    alertBatches.forEach { batch ->
        val alertMessage = buildAlertMessage(batch, batchDate)
        
        snsClient.publish(
            PublishRequest.builder()
                .topicArn(fraudAlertTopicArn)
                .subject("Fraud Alert: ${batch.size} High-Risk Transactions Detected")
                .message(alertMessage)
                .messageAttributes(mapOf(
                    "batchDate" to MessageAttributeValue.builder()
                        .dataType("String")
                        .stringValue(batchDate)
                        .build(),
                    "highRiskCount" to MessageAttributeValue.builder()
                        .dataType("Number")
                        .stringValue(batch.size.toString())
                        .build()
                ))
                .build()
        )
        
        alertsSent += batch.size
        logger.info("Sent alert for ${batch.size} high-risk transactions")
    }
    
    // 3. Return alert summary
    return objectMapper.createObjectNode().apply {
        put("batchDate", batchDate)
        put("highRiskCount", highRiskTransactions.size)
        put("alertsSent", alertsSent)
        put("alertBatches", alertBatches.size)
    }
}

private fun buildAlertMessage(transactions: List<JsonNode>, batchDate: String): String {
    val sb = StringBuilder()
    sb.appendLine("High-Risk Fraud Transactions Detected")
    sb.appendLine("Batch Date: $batchDate")
    sb.appendLine("Count: ${transactions.size}")
    sb.appendLine()
    sb.appendLine("Transactions:")
    
    transactions.forEach { txn ->
        sb.appendLine("  - ID: ${txn.get("transactionId").asText()}")
        sb.appendLine("    Amount: $${txn.get("amount").asDouble()}")
        sb.appendLine("    Fraud Score: ${txn.get("fraudScore").asDouble()}")
        sb.appendLine("    Merchant: ${txn.get("merchantCategory").asText()}")
        sb.appendLine()
    }
    
    return sb.toString()
}
```

**Output** (to S3):
```json
{
  "batchDate": "2024-01-15",
  "highRiskCount": 1,
  "alertsSent": 1,
  "alertBatches": 1
}
```

**SNS Alert Message Format**:
```
Subject: Fraud Alert: 1 High-Risk Transactions Detected

High-Risk Fraud Transactions Detected
Batch Date: 2024-01-15
Count: 1

Transactions:
  - ID: txn-002
    Amount: $2500.00
    Fraud Score: 0.8912
    Merchant: online
```


#### 4. MonitorStage (Lambda Handler)

**Purpose**: Monitor model performance and detect distribution drift

**Handler Class**: `MonitorHandler : WorkflowLambdaHandler()`

**Input** (from StoreStage S3 output):
```json
{
  "batchDate": "2024-01-15",
  "totalTransactions": 2,
  "successCount": 2,
  "errorCount": 0,
  "riskDistribution": {
    "highRisk": 1,
    "mediumRisk": 0,
    "lowRisk": 1
  },
  "avgFraudScore": 0.4573
}
```

**Processing**:
```kotlin
override fun processData(input: JsonNode): JsonNode {
    val batchDate = input.get("batchDate").asText()
    val avgFraudScore = input.get("avgFraudScore").asDouble()
    val riskDistribution = input.get("riskDistribution")
    
    // 1. Load historical baseline from S3
    val baseline = loadHistoricalBaseline()
    
    // 2. Calculate distribution metrics
    val highRiskPct = riskDistribution.get("highRisk").asInt().toDouble() / 
                      input.get("totalTransactions").asInt()
    val mediumRiskPct = riskDistribution.get("mediumRisk").asInt().toDouble() / 
                        input.get("totalTransactions").asInt()
    val lowRiskPct = riskDistribution.get("lowRisk").asInt().toDouble() / 
                     input.get("totalTransactions").asInt()
    
    // 3. Compare to baseline (detect drift)
    val avgScoreDrift = Math.abs(avgFraudScore - baseline.avgFraudScore)
    val highRiskDrift = Math.abs(highRiskPct - baseline.highRiskPct)
    
    val driftDetected = avgScoreDrift > 0.1 || highRiskDrift > 0.05
    
    // 4. Send monitoring alert if drift detected
    if (driftDetected) {
        snsClient.publish(
            PublishRequest.builder()
                .topicArn(monitoringAlertTopicArn)
                .subject("Model Drift Detected: $batchDate")
                .message(buildDriftAlertMessage(
                    batchDate, avgFraudScore, baseline.avgFraudScore,
                    highRiskPct, baseline.highRiskPct
                ))
                .build()
        )
        
        logger.warn("Model drift detected: avgScoreDrift=$avgScoreDrift, highRiskDrift=$highRiskDrift")
    }
    
    // 5. Write metrics to S3 for historical tracking
    val metricsPath = "metrics/$batchDate.json"
    val metrics = objectMapper.createObjectNode().apply {
        put("batchDate", batchDate)
        put("avgFraudScore", avgFraudScore)
        put("highRiskPct", highRiskPct)
        put("mediumRiskPct", mediumRiskPct)
        put("lowRiskPct", lowRiskPct)
        put("avgScoreDrift", avgScoreDrift)
        put("highRiskDrift", highRiskDrift)
        put("driftDetected", driftDetected)
    }
    
    s3Client.putObject(
        PutObjectRequest.builder()
            .bucket("fraud-detection-metrics")
            .key(metricsPath)
            .build(),
        RequestBody.fromString(objectMapper.writeValueAsString(metrics))
    )
    
    // 6. Return monitoring summary
    return objectMapper.createObjectNode().apply {
        put("batchDate", batchDate)
        put("avgFraudScore", avgFraudScore)
        put("avgScoreDrift", avgScoreDrift)
        put("highRiskDrift", highRiskDrift)
        put("driftDetected", driftDetected)
        put("metricsPath", metricsPath)
    }
}

private fun loadHistoricalBaseline(): PerformanceBaseline {
    // Load last 30 days of metrics and calculate baseline
    val last30Days = (1..30).map { daysAgo ->
        val date = LocalDate.now().minusDays(daysAgo.toLong())
        loadMetricsForDate(date)
    }.filterNotNull()
    
    return PerformanceBaseline(
        avgFraudScore = last30Days.map { it.avgFraudScore }.average(),
        highRiskPct = last30Days.map { it.highRiskPct }.average()
    )
}
```

**Output** (to S3):
```json
{
  "batchDate": "2024-01-15",
  "avgFraudScore": 0.4573,
  "avgScoreDrift": 0.0234,
  "highRiskDrift": 0.0123,
  "driftDetected": false,
  "metricsPath": "metrics/2024-01-15.json"
}
```


## Data Models

### Core Domain Models

#### Transaction
```kotlin
data class Transaction(
    val id: String,
    val timestamp: Long,
    val amount: Double,
    val merchantCategory: String,
    val features: Map<String, Double> // V1-V28 from Kaggle dataset
)
```

#### ScoredTransaction
```kotlin
data class ScoredTransaction(
    val transactionId: String,
    val timestamp: Long,
    val amount: Double,
    val merchantCategory: String,
    val features: Map<String, Double>,
    val fraudScore: Double, // 0.0 to 1.0
    val scoringTimestamp: Long
)
```

### Workflow Models

#### ExecutionContext
```kotlin
data class ExecutionContext(
    var executionId: String = "",
    var currentStage: String = "",
    var previousStage: String? = null,
    var workflowBucket: String = "",
    var initialData: Map<String, Any>? = null
)
```

#### StageResult
```kotlin
data class StageResult(
    val status: String, // "SUCCESS" or "FAILED"
    val stage: String,
    val recordsProcessed: Int,
    val errorMessage: String? = null
)
```

### Performance Models

#### PerformanceBaseline
```kotlin
data class PerformanceBaseline(
    val avgFraudScore: Double,
    val highRiskPct: Double
)
```

#### BatchWriteResult
```kotlin
data class BatchWriteResult(
    val successCount: Int,
    val errorCount: Int
)
```


### S3 Data Organization

#### Workflow Bucket Structure
```
s3://fraud-detection-workflow-{account-id}/
├── executions/
│   ├── {execution-id-1}/
│   │   ├── DataPrepStage/output.json
│   │   ├── TrainStage/output.json
│   │   ├── EvaluateStage/output.json
│   │   └── DeployStage/output.json
│   └── {execution-id-2}/
│       ├── ScoreStage/output.json
│       ├── StoreStage/output.json
│       ├── AlertStage/output.json
│       └── MonitorStage/output.json
```

#### Data Bucket Structure
```
s3://fraud-detection-data/
├── kaggle-credit-card-fraud.csv          # Raw dataset (284,807 records)
├── prepared/
│   ├── train.parquet                     # 70% split (199,363 records)
│   ├── validation.parquet                # 15% split (42,721 records)
│   └── test.parquet                      # 15% split (42,723 records)
└── daily-batches/
    ├── 2024-01-15.json                   # Daily transaction batches
    ├── 2024-01-16.json
    └── ...
```

#### Model Bucket Structure
```
s3://fraud-detection-models/
├── fraud-detection-1234567890/
│   └── output/
│       └── model.tar.gz                  # Trained model artifacts
└── fraud-detection-1234567891/
    └── output/
        └── model.tar.gz
```

#### Config Bucket Structure
```
s3://fraud-detection-config/
└── current-endpoint.json                 # Current production endpoint metadata
```

#### Metrics Bucket Structure
```
s3://fraud-detection-metrics/
└── metrics/
    ├── 2024-01-15.json                   # Daily performance metrics
    ├── 2024-01-16.json
    └── ...
```

### DynamoDB Schema

#### FraudScores Table

**Primary Key**:
- Partition Key: `transactionId` (String)
- Sort Key: `timestamp` (Number)

**Attributes**:
```
{
  "transactionId": "txn-001",              // Partition key
  "timestamp": 1705334400000,              // Sort key
  "batchDate": "2024-01-15",               // GSI partition key
  "amount": 149.62,
  "merchantCategory": "retail",
  "fraudScore": 0.0234,
  "scoringTimestamp": 1705334401000,
  "features": "{\"Time\":0.0,\"V1\":-1.36,...}" // JSON string
}
```

**Global Secondary Index (GSI)**:
- Name: `BatchDateIndex`
- Partition Key: `batchDate` (String)
- Sort Key: `fraudScore` (Number)
- Purpose: Query all transactions for a specific date, sorted by fraud score

**Capacity Settings**:
- On-Demand billing mode (auto-scaling)
- Estimated daily writes: 10,000 transactions
- Estimated daily reads: 1,000 queries


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, I identified the following redundancies:
- Properties 2.5, 3.5, 5.3, 6.5, 7.5, 9.3, 13.1 all test the same S3 orchestration pattern (writing outputs) → Combined into Property 1
- Properties 3.1, 4.1, 6.1, 7.1, 8.1, 13.2 all test the same S3 orchestration pattern (reading inputs) → Combined into Property 2
- Properties 10.3, 13.1, 13.2 are covered by Properties 1 and 2 → Removed
- Property 13.3 (path convention) is tested implicitly by Properties 1 and 2 → Kept as separate property for clarity
- Properties 7.2 and 7.3 both test DynamoDB writes → Combined into Property 8

The remaining properties provide unique validation value and are not redundant.


### Core Workflow Properties

**Property 1: S3 Output Convention**
*For any* workflow stage execution, when the stage completes successfully, the output data SHALL be written to S3 at path `executions/{executionId}/{currentStage}/output.json` in the workflow bucket.
**Validates: Requirements 2.5, 3.5, 5.3, 6.5, 7.5, 9.3, 13.1**

**Property 2: S3 Input Convention**
*For any* non-first workflow stage execution, when the stage starts, the input data SHALL be read from S3 at path `executions/{executionId}/{previousStage}/output.json` in the workflow bucket.
**Validates: Requirements 3.1, 4.1, 6.1, 7.1, 8.1, 13.2**

**Property 3: S3 Path Construction**
*For any* execution context with executionId and stageName, the constructed S3 path SHALL follow the format `executions/{executionId}/{stageName}/output.json`.
**Validates: Requirements 13.3**

**Property 4: Stage Output Metadata**
*For any* stage output written to S3, the output SHALL contain execution metadata including at minimum the stage name and status.
**Validates: Requirements 13.4**

**Property 5: Error Handling**
*For any* Lambda handler that encounters an exception during processing, the handler SHALL return a StageResult with status "FAILED" and a non-null errorMessage containing the exception details.
**Validates: Requirements 10.4, 15.1**

**Property 6: Partial Results Preservation**
*For any* workflow stage that writes intermediate results before encountering an error, those partial results SHALL remain in S3 to enable manual recovery.
**Validates: Requirements 15.5**


### Data Processing Properties

**Property 7: Data Split Proportions**
*For any* dataset split into training, validation, and test sets with proportions (trainPct, validationPct, testPct), the sum of record counts SHALL equal the original dataset size, and each split SHALL contain approximately trainPct%, validationPct%, and testPct% of records respectively (within 1% tolerance).
**Validates: Requirements 2.3**

**Property 8: SageMaker Output Format**
*For any* prepared dataset written for SageMaker training, the output SHALL be in Parquet format and SHALL contain all feature columns plus the target column specified in the configuration.
**Validates: Requirements 2.4**

**Property 9: Data Quality Validation**
*For any* dataset processed by the data preparation stage, if the dataset contains missing values or outliers beyond acceptable thresholds, the stage SHALL either impute/handle them appropriately or fail with a descriptive error.
**Validates: Requirements 11.4, 11.5**

### Scoring and Storage Properties

**Property 10: Fraud Score Range**
*For any* transaction scored by the inference pipeline, the fraud score SHALL be a number between 0.0 and 1.0 (inclusive).
**Validates: Requirements 6.4**

**Property 11: Endpoint Invocation**
*For any* batch of transactions to be scored, the scoring stage SHALL invoke the SageMaker endpoint exactly once per transaction.
**Validates: Requirements 6.3**

**Property 12: DynamoDB Write Completeness**
*For any* scored transaction written to DynamoDB, the item SHALL contain all required attributes: transactionId, timestamp, batchDate, amount, merchantCategory, fraudScore, scoringTimestamp, and features.
**Validates: Requirements 7.2, 7.3**

**Property 13: High-Risk Transaction Identification**
*For any* set of scored transactions, the transactions identified as high-risk SHALL be exactly those with fraudScore >= 0.8.
**Validates: Requirements 8.2**

**Property 14: Alert Message Completeness**
*For any* high-risk transaction included in an alert, the alert message SHALL contain the transaction ID, fraud score, amount, and timestamp.
**Validates: Requirements 8.4**

**Property 15: Alert Batching**
*For any* set of high-risk transactions to be alerted, if the count exceeds 100, the transactions SHALL be batched into groups of at most 100 transactions per SNS message.
**Validates: Requirements 8.5**


### Model Performance Properties

**Property 16: Distribution Drift Detection**
*For any* daily batch of scored transactions, if the average fraud score deviates from the historical baseline by more than 0.1 OR the high-risk percentage deviates by more than 0.05, the monitoring stage SHALL detect drift and set driftDetected to true.
**Validates: Requirements 14.2**

**Property 17: Metrics Persistence**
*For any* completed inference pipeline execution, the monitoring stage SHALL write performance metrics to S3 at path `metrics/{batchDate}.json`.
**Validates: Requirements 14.3**


## Error Handling

### Lambda Handler Error Handling

All Lambda handlers extend `WorkflowLambdaHandler`, which provides comprehensive error handling:

1. **Exception Catching**: All exceptions are caught in the `handleRequest` method
2. **Error Logging**: Full stack traces are logged to CloudWatch with execution context
3. **Error Response**: Failed stages return `StageResult` with status "FAILED" and detailed error message
4. **Partial Results**: Outputs written before errors are preserved in S3

### S3 Error Handling

The `WorkflowLambdaHandler` base class handles S3 errors with retry logic:

**Transient Errors** (retry via Step Functions):
- 503 SlowDown (throttling)
- 500 InternalError
- Network/SDK errors

**Permanent Errors** (fail immediately):
- 403 Forbidden (IAM permission issue)
- 404 NoSuchKey (previous stage failed or path incorrect)

### SageMaker Error Handling

**Training Job Failures**:
- TrainStage waits for training completion with timeout (1 hour)
- If training fails, throws `IllegalStateException` with training job status
- Step Functions retry logic handles transient SageMaker API errors

**Endpoint Failures**:
- DeployStage waits for endpoint deployment with timeout (30 minutes)
- Health check validates endpoint before marking deployment successful
- If health check fails, throws exception to trigger retry

**Inference Failures**:
- ScoreStage handles endpoint invocation errors per transaction
- Failed transactions are logged but don't fail the entire batch
- Partial results are written to S3 for manual review

### DynamoDB Error Handling

**Batch Write Failures**:
- StoreStage tracks unprocessed items from batch writes
- Returns error count in stage output for monitoring
- Unprocessed items are logged for manual retry

**Throttling**:
- DynamoDB on-demand mode auto-scales to handle load
- Transient throttling errors are retried by AWS SDK

### SNS Error Handling

**Alert Failures**:
- AlertStage logs SNS publish errors but doesn't fail the stage
- Allows pipeline to complete even if alerts fail
- Failed alerts are logged for manual investigation

**Rate Limiting**:
- Alerts are batched (max 100 per message) to avoid rate limits
- Multiple batches are sent sequentially with logging


## Testing Strategy

### Dual Testing Approach

The system requires both unit tests and property-based tests for comprehensive coverage:

**Unit Tests**: Verify specific examples, edge cases, and error conditions
- Specific dataset loading scenarios
- SageMaker API integration with mocked clients
- DynamoDB write operations with specific data
- SNS alert formatting with example transactions
- Error handling with specific exception types

**Property-Based Tests**: Verify universal properties across all inputs
- S3 path construction for any execution context
- Data split proportions for any dataset size
- Fraud score range validation for any model output
- Alert batching for any number of high-risk transactions
- Drift detection for any score distribution

Both approaches are complementary and necessary. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across the input space.

### Property-Based Testing Configuration

**Library**: Use Kotest Property Testing for Kotlin
- Minimum 100 iterations per property test (due to randomization)
- Each test references its design document property via comment tag

**Tag Format**:
```kotlin
// Feature: fraud-detection-ml-pipeline, Property 1: S3 Output Convention
@Test
fun `property test - S3 output convention`() = runTest {
    checkAll(100, Arb.executionContext(), Arb.stageOutput()) { context, output ->
        // Test implementation
    }
}
```

### Test Organization

```
src/test/kotlin/com/frauddetection/
├── properties/
│   ├── WorkflowPropertiesTest.kt        # Properties 1-6
│   ├── DataProcessingPropertiesTest.kt  # Properties 7-9
│   ├── ScoringPropertiesTest.kt         # Properties 10-15
│   └── MonitoringPropertiesTest.kt      # Properties 16-17
├── unit/
│   ├── handlers/
│   │   ├── DataPrepHandlerTest.kt
│   │   ├── TrainHandlerTest.kt
│   │   ├── EvaluateHandlerTest.kt
│   │   ├── DeployHandlerTest.kt
│   │   ├── ScoreHandlerTest.kt
│   │   ├── StoreHandlerTest.kt
│   │   ├── AlertHandlerTest.kt
│   │   └── MonitorHandlerTest.kt
│   └── models/
│       ├── TransactionTest.kt
│       └── ScoredTransactionTest.kt
└── integration/
    ├── TrainingPipelineIntegrationTest.kt
    └── InferencePipelineIntegrationTest.kt
```

### Key Testing Scenarios

**Training Pipeline**:
1. End-to-end training with mock SageMaker
2. Model accuracy below threshold (should fail)
3. S3 orchestration between stages
4. Error handling and retry logic

**Inference Pipeline**:
1. Batch scoring with various transaction counts
2. High-risk transaction alerting
3. DynamoDB storage with batch writes
4. Drift detection with various distributions

**Error Scenarios**:
1. S3 access denied (403)
2. Previous stage output missing (404)
3. SageMaker training failure
4. DynamoDB throttling
5. SNS publish failure


## Deployment Architecture

### Infrastructure Components

**Step Functions Workflows**:
- `FraudDetectionTrainingWorkflow` (Standard): Weekly training pipeline
- `FraudDetectionInferenceWorkflow` (Express): Daily inference pipeline

**Lambda Functions**:
- `fraud-detection-train-handler`: Training job orchestration
- `fraud-detection-evaluate-handler`: Model evaluation
- `fraud-detection-deploy-handler`: Endpoint deployment
- `fraud-detection-score-handler`: Transaction scoring
- `fraud-detection-store-handler`: DynamoDB storage
- `fraud-detection-alert-handler`: SNS alerting
- `fraud-detection-monitor-handler`: Performance monitoring

**Glue Jobs**:
- `fraud-detection-data-prep`: PySpark data preparation

**S3 Buckets**:
- `fraud-detection-workflow-{account-id}`: Workflow orchestration
- `fraud-detection-data`: Raw and prepared datasets
- `fraud-detection-models`: Model artifacts
- `fraud-detection-config`: Endpoint configuration
- `fraud-detection-metrics`: Performance metrics

**DynamoDB Tables**:
- `FraudScores`: Scored transaction storage

**SNS Topics**:
- `fraud-detection-alerts`: High-risk transaction alerts
- `fraud-detection-monitoring`: Drift detection alerts
- `fraud-detection-failures`: Pipeline failure notifications

**EventBridge Rules**:
- `fraud-detection-training-schedule`: Weekly trigger (Sunday 2 AM)
- `fraud-detection-inference-schedule`: Daily trigger (1 AM)

### Deployment Process

1. **Build Application**:
```bash
cd CEAPFraudDetection
./gradlew build
```

2. **Deploy Infrastructure** (using CEAP deployment scripts):
```bash
cd infrastructure

# Deploy training pipeline (Standard workflow with Glue)
./deploy-workflow-simple.sh -n fraud-detection-training -t standard

# Deploy inference pipeline (Express workflow)
./deploy-workflow-simple.sh -n fraud-detection-inference -t express
```

3. **Upload Glue Script**:
```bash
aws s3 cp glue-scripts/data-prep.py s3://fraud-detection-glue-scripts/
```

4. **Upload Dataset**:
```bash
aws s3 cp kaggle-credit-card-fraud.csv s3://fraud-detection-data/
```

5. **Configure EventBridge Schedules**:
```bash
# Training: Weekly on Sunday at 2 AM
aws events put-rule --name fraud-detection-training-schedule \
  --schedule-expression "cron(0 2 ? * SUN *)"

# Inference: Daily at 1 AM
aws events put-rule --name fraud-detection-inference-schedule \
  --schedule-expression "cron(0 1 * * ? *)"
```

### Monitoring and Observability

**CloudWatch Dashboards**:
- Training pipeline metrics (training time, model accuracy)
- Inference pipeline metrics (scoring throughput, fraud score distribution)
- Error rates and failure notifications

**CloudWatch Alarms**:
- Training pipeline failures
- Inference pipeline failures
- Model drift detection
- DynamoDB throttling
- Lambda errors

**X-Ray Tracing**:
- End-to-end workflow tracing
- Lambda function performance
- SageMaker endpoint latency

### Cost Estimation

**Training Pipeline** (Weekly):
- Step Functions: $0.004/week (4 stages × $25/million)
- Glue: ~$0.44/hour × 0.5 hours = $0.22/week
- SageMaker Training: ~$0.269/hour × 1 hour = $0.269/week
- SageMaker Endpoint (deployment): ~$0.228/hour × 24/7 = $163.68/month
- Lambda: Negligible (< $0.01/week)
- S3: ~$0.023/GB/month × 1 GB = $0.023/month
- **Total: ~$164/month**

**Inference Pipeline** (Daily):
- Step Functions: $0.004/day (4 stages × $1/million) = $0.12/month
- Lambda: ~$0.20/day × 30 = $6/month
- DynamoDB: On-demand, ~$1.25/million writes × 0.3 million = $0.375/month
- SNS: ~$0.50/million messages × 0.001 million = $0.0005/month
- S3: ~$0.023/GB/month × 0.5 GB = $0.012/month
- **Total: ~$6.50/month**

**Grand Total: ~$170/month**

