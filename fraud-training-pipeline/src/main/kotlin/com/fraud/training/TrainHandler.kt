package com.fraud.training

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.node.ObjectNode
import com.fraud.common.handler.WorkflowLambdaHandler
import software.amazon.awssdk.services.sagemaker.SageMakerClient
import software.amazon.awssdk.services.sagemaker.model.*
import software.amazon.awssdk.services.sagemaker.waiters.SageMakerWaiter
import software.amazon.awssdk.services.ssm.SsmClient
import software.amazon.awssdk.services.ssm.model.GetParameterRequest

/**
 * Lambda handler for the training stage of the fraud detection ML pipeline.
 * 
 * This handler orchestrates SageMaker training jobs using the XGBoost algorithm
 * to train fraud detection models on prepared transaction data.
 * 
 * **Workflow:**
 * 1. Reads prepared data paths from previous stage (DataPrepStage) output
 * 2. Configures SageMaker training job with XGBoost algorithm
 * 3. Starts training job and waits for completion
 * 4. Returns training job metadata for next stage (EvaluateStage)
 * 
 * **Requirements:**
 * - Requirement 3.1: Read prepared data locations from S3
 * - Requirement 3.2: Configure SageMaker training job with appropriate instance type and hyperparameters
 * - Requirement 3.3: Train the Fraud_Detector using training and validation datasets
 * - Requirement 3.4: Write model artifacts and evaluation metrics to S3
 * - Requirement 3.5: Fail if model accuracy is below 0.90 (handled by EvaluateStage)
 * 
 * **Input Format** (from DataPrepStage):
 * ```json
 * {
 *   "trainDataPath": "s3://fraud-detection-data/prepared/train.parquet",
 *   "validationDataPath": "s3://fraud-detection-data/prepared/validation.parquet",
 *   "testDataPath": "s3://fraud-detection-data/prepared/test.parquet",
 *   "recordCounts": { "train": 199363, "validation": 42721, "test": 42723 },
 *   "features": ["Time", "V1", ..., "V28", "Amount"],
 *   "targetColumn": "Class"
 * }
 * ```
 * 
 * **Output Format** (to S3 for EvaluateStage):
 * ```json
 * {
 *   "trainingJobName": "fraud-detection-1234567890",
 *   "modelArtifactPath": "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz",
 *   "trainingJobStatus": "Completed"
 * }
 * ```
 */
open class TrainHandler(
    private val sageMakerClient: SageMakerClient = SageMakerClient.builder().build(),
    private val ssmClient: SsmClient = SsmClient.builder().build(),
    private val sageMakerExecutionRoleArn: String = System.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
        ?: throw IllegalStateException("SAGEMAKER_EXECUTION_ROLE_ARN environment variable must be set")
) : WorkflowLambdaHandler() {
    
    /**
     * Reads hyperparameters from AWS Parameter Store.
     * Falls back to default values if Parameter Store is unavailable.
     */
    private fun loadHyperparameters(): Map<String, String> {
        val paramPaths = mapOf(
            "objective" to "/fraud-detection/hyperparameters/objective",
            "num_round" to "/fraud-detection/hyperparameters/num_round",
            "max_depth" to "/fraud-detection/hyperparameters/max_depth",
            "eta" to "/fraud-detection/hyperparameters/eta",
            "subsample" to "/fraud-detection/hyperparameters/subsample",
            "colsample_bytree" to "/fraud-detection/hyperparameters/colsample_bytree"
        )

        val defaults = mapOf(
            "objective" to "binary:logistic",
            "num_round" to "150",
            "max_depth" to "7",
            "eta" to "0.2",
            "subsample" to "0.8",
            "colsample_bytree" to "0.8"
        )

        val hyperparameters = mutableMapOf<String, String>()

        for ((name, path) in paramPaths) {
            try {
                val response = ssmClient.getParameter(
                    GetParameterRequest.builder()
                        .name(path)
                        .build()
                )
                val value = response.parameter().value()
                hyperparameters[name] = value
                logger.info("Loaded hyperparameter from Parameter Store: $name = $value")
            } catch (e: Exception) {
                val defaultValue = defaults[name] ?: ""
                hyperparameters[name] = defaultValue
                logger.warn("Failed to read $path from Parameter Store, using default: $name = $defaultValue. Error: ${e.message}")
            }
        }

        return hyperparameters
    }

    /**
     * Processes the training stage by configuring and launching a SageMaker training job.
     * 
     * This method:
     * 1. Extracts data paths from the input (DataPrepStage output)
     * 2. Configures a SageMaker training job with XGBoost algorithm
     * 3. Starts the training job
     * 4. Waits for training completion (with 1 hour timeout)
     * 5. Returns training job metadata
     * 
     * @param input Input data from DataPrepStage containing prepared data paths
     * @return Output data containing training job name and model artifact path
     * @throws IllegalStateException if training job fails
     */
    override fun processData(input: JsonNode): JsonNode {
        // 1. Extract data paths from input
        val trainDataPath = input.get("trainDataPath")?.asText()
            ?: throw IllegalArgumentException("trainDataPath is required in input")
        val validationDataPath = input.get("validationDataPath")?.asText()
            ?: throw IllegalArgumentException("validationDataPath is required in input")
        
        logger.info("Training data path: $trainDataPath")
        logger.info("Validation data path: $validationDataPath")
        
        // 2. Configure SageMaker training job
        val trainingJobName = "fraud-detection-${System.currentTimeMillis()}"
        logger.info("Creating training job: $trainingJobName")
        
        // Load hyperparameters from Parameter Store (with defaults as fallback)
        val hyperparameters = loadHyperparameters()
        logger.info("Using hyperparameters: $hyperparameters")
        
        val trainingJobRequest = CreateTrainingJobRequest.builder()
            .trainingJobName(trainingJobName)
            .algorithmSpecification(
                AlgorithmSpecification.builder()
                    .trainingImage("683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1")
                    .trainingInputMode(TrainingInputMode.FILE)
                    .build()
            )
            .roleArn(sageMakerExecutionRoleArn)
            .inputDataConfig(
                listOf(
                    Channel.builder()
                        .channelName("train")
                        .dataSource(
                            DataSource.builder()
                                .s3DataSource(
                                    S3DataSource.builder()
                                        .s3Uri(trainDataPath)
                                        .s3DataType(S3DataType.S3_PREFIX)
                                        .build()
                                )
                                .build()
                        )
                        .contentType("application/x-parquet")
                        .build(),
                    Channel.builder()
                        .channelName("validation")
                        .dataSource(
                            DataSource.builder()
                                .s3DataSource(
                                    S3DataSource.builder()
                                        .s3Uri(validationDataPath)
                                        .s3DataType(S3DataType.S3_PREFIX)
                                        .build()
                                )
                                .build()
                        )
                        .contentType("application/x-parquet")
                        .build()
                )
            )
            .outputDataConfig(
                OutputDataConfig.builder()
                    .s3OutputPath("s3://fraud-detection-models/")
                    .build()
            )
            .resourceConfig(
                ResourceConfig.builder()
                    .instanceType(TrainingInstanceType.ML_M5_XLARGE)
                    .instanceCount(1)
                    .volumeSizeInGB(30)
                    .build()
            )
            .stoppingCondition(
                StoppingCondition.builder()
                    .maxRuntimeInSeconds(3600) // 1 hour max
                    .build()
            )
            .hyperParameters(hyperparameters)
            .build()
        
        // 3. Start training job
        logger.info("Starting SageMaker training job: $trainingJobName")
        sageMakerClient.createTrainingJob(trainingJobRequest)
        
        // 4. Wait for training completion (with timeout)
        logger.info("Waiting for training job to complete: $trainingJobName")
        val waiter = sageMakerClient.waiter()
        
        try {
            waiter.waitUntilTrainingJobCompletedOrStopped(
                DescribeTrainingJobRequest.builder()
                    .trainingJobName(trainingJobName)
                    .build()
            )
        } catch (e: Exception) {
            logger.error("Error waiting for training job completion: ${e.message}", e)
            throw IllegalStateException(
                "Training job did not complete within timeout: $trainingJobName. Error: ${e.message}",
                e
            )
        }
        
        // 5. Check training status
        val describeResponse = sageMakerClient.describeTrainingJob(
            DescribeTrainingJobRequest.builder()
                .trainingJobName(trainingJobName)
                .build()
        )
        
        val trainingJobStatus = describeResponse.trainingJobStatus()
        logger.info("Training job status: $trainingJobStatus")
        
        if (trainingJobStatus != TrainingJobStatus.COMPLETED) {
            val failureReason = describeResponse.failureReason() ?: "Unknown failure reason"
            throw IllegalStateException(
                "Training job failed with status: $trainingJobStatus. Reason: $failureReason"
            )
        }
        
        // 6. Return training job metadata
        val modelArtifactPath = describeResponse.modelArtifacts().s3ModelArtifacts()
        logger.info("Training completed successfully. Model artifact: $modelArtifactPath")
        
        // Pass through testDataPath for EvaluateStage
        val testDataPath = input.get("testDataPath")?.asText()
        
        return objectMapper.createObjectNode().apply {
            put("trainingJobName", trainingJobName)
            put("modelArtifactPath", modelArtifactPath)
            put("trainingJobStatus", trainingJobStatus.toString())
            if (testDataPath != null) {
                put("testDataPath", testDataPath)
            }
        }
    }
}
