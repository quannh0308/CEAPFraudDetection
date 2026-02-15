package com.fraud.training

import com.fasterxml.jackson.databind.JsonNode
import com.fraud.common.handler.WorkflowLambdaHandler
import software.amazon.awssdk.services.sagemaker.SageMakerClient
import software.amazon.awssdk.services.sagemaker.model.*
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.services.s3.model.PutObjectRequest

/**
 * Lambda handler for the deployment stage of the fraud detection ML pipeline.
 * 
 * This handler deploys trained models to production SageMaker endpoints, performs
 * health checks, and writes endpoint metadata for the inference pipeline to consume.
 * 
 * **Workflow:**
 * 1. Reads evaluation metrics from previous stage (EvaluateStage) output
 * 2. Creates SageMaker model from training artifact
 * 3. Creates endpoint configuration
 * 4. Creates or updates production endpoint
 * 5. Waits for endpoint to be in service
 * 6. Performs health check with test transaction
 * 7. Writes endpoint metadata to config bucket for inference pipeline
 * 8. Returns deployment metadata
 * 
 * **Requirements:**
 * - Requirement 4.1: Read model artifact location from S3
 * - Requirement 4.2: Create or update a SageMaker_Endpoint with the trained model
 * - Requirement 4.3: Configure the endpoint with appropriate instance type for real-time inference
 * - Requirement 4.4: Validate endpoint health by sending a test transaction
 * - Requirement 4.5: Write endpoint name and deployment metadata to S3
 * 
 * **Input Format** (from EvaluateStage):
 * ```json
 * {
 *   "modelArtifactPath": "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz",
 *   "accuracy": 0.9523,
 *   "precision": 0.8912,
 *   "recall": 0.8456,
 *   "f1Score": 0.8678,
 *   "auc": 0.9234,
 *   "testRecordCount": 42723
 * }
 * ```
 * 
 * **Output Format** (to S3 for workflow completion):
 * ```json
 * {
 *   "endpointName": "fraud-detection-prod",
 *   "modelName": "fraud-detection-prod-1234567890",
 *   "endpointConfigName": "fraud-detection-prod-1234567890-config",
 *   "deploymentTimestamp": 1234567890,
 *   "modelAccuracy": 0.9523,
 *   "healthCheckPrediction": 0.0234
 * }
 * ```
 */
open class DeployHandler(
    private val sageMakerClient: SageMakerClient = SageMakerClient.builder().build(),
    private val sageMakerRuntimeClient: SageMakerRuntimeClient = SageMakerRuntimeClient.builder().build(),
    private val sageMakerExecutionRoleArn: String = System.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
        ?: throw IllegalStateException("SAGEMAKER_EXECUTION_ROLE_ARN environment variable must be set"),
    private val configBucket: String = System.getenv("CONFIG_BUCKET") ?: "fraud-detection-config"
) : WorkflowLambdaHandler() {
    
    companion object {
        const val PRODUCTION_ENDPOINT_NAME = "fraud-detection-prod"
        const val CONFIG_KEY = "current-endpoint.json"
    }
    
    /**
     * Processes the deployment stage by creating/updating a production SageMaker endpoint.
     * 
     * This method:
     * 1. Extracts model artifact path from input (EvaluateStage output)
     * 2. Creates SageMaker model from artifact
     * 3. Creates endpoint configuration
     * 4. Creates or updates production endpoint
     * 5. Waits for endpoint to be in service
     * 6. Performs health check with test transaction
     * 7. Writes endpoint metadata to config bucket
     * 8. Returns deployment metadata
     * 
     * @param input Input data from EvaluateStage containing model artifact path and metrics
     * @return Output data containing deployment metadata
     * @throws IllegalStateException if deployment or health check fails
     */
    override fun processData(input: JsonNode): JsonNode {
        // 1. Extract model artifact path from input
        val modelArtifactPath = input.get("modelArtifactPath")?.asText()
            ?: throw IllegalArgumentException("modelArtifactPath is required in input")
        
        val modelAccuracy = input.get("accuracy")?.asDouble() ?: 0.0
        
        logger.info("Deploying model: $modelArtifactPath with accuracy: $modelAccuracy")
        
        // 2. Create SageMaker model
        val timestamp = System.currentTimeMillis()
        val modelName = "fraud-detection-prod-$timestamp"
        val endpointConfigName = "$modelName-config"
        val endpointName = PRODUCTION_ENDPOINT_NAME // Fixed name for inference pipeline
        
        logger.info("Creating SageMaker model: $modelName")
        
        sageMakerClient.createModel(
            CreateModelRequest.builder()
                .modelName(modelName)
                .primaryContainer(
                    ContainerDefinition.builder()
                        .image("683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1")
                        .modelDataUrl(modelArtifactPath)
                        .build()
                )
                .executionRoleArn(sageMakerExecutionRoleArn)
                .build()
        )
        
        logger.info("Successfully created model: $modelName")
        
        // 3. Create endpoint configuration
        logger.info("Creating endpoint configuration: $endpointConfigName")
        
        sageMakerClient.createEndpointConfig(
            CreateEndpointConfigRequest.builder()
                .endpointConfigName(endpointConfigName)
                .productionVariants(
                    listOf(
                        ProductionVariant.builder()
                            .variantName("AllTraffic")
                            .modelName(modelName)
                            .instanceType(ProductionVariantInstanceType.ML_M5_LARGE)
                            .initialInstanceCount(1)
                            .initialVariantWeight(1.0f)
                            .build()
                    )
                )
                .build()
        )
        
        logger.info("Successfully created endpoint configuration: $endpointConfigName")
        
        // 4. Check if endpoint exists
        val endpointExists = try {
            sageMakerClient.describeEndpoint(
                DescribeEndpointRequest.builder()
                    .endpointName(endpointName)
                    .build()
            )
            logger.info("Endpoint $endpointName already exists, will update")
            true
        } catch (e: ResourceNotFoundException) {
            logger.info("Endpoint $endpointName does not exist, will create")
            false
        }
        
        // 5. Create or update endpoint
        if (endpointExists) {
            logger.info("Updating existing endpoint: $endpointName")
            sageMakerClient.updateEndpoint(
                UpdateEndpointRequest.builder()
                    .endpointName(endpointName)
                    .endpointConfigName(endpointConfigName)
                    .build()
            )
        } else {
            logger.info("Creating new endpoint: $endpointName")
            sageMakerClient.createEndpoint(
                CreateEndpointRequest.builder()
                    .endpointName(endpointName)
                    .endpointConfigName(endpointConfigName)
                    .build()
            )
        }
        
        // 6. Wait for endpoint to be in service
        logger.info("Waiting for endpoint to be in service: $endpointName")
        val waiter = sageMakerClient.waiter()
        
        try {
            waiter.waitUntilEndpointInService(
                DescribeEndpointRequest.builder()
                    .endpointName(endpointName)
                    .build()
            )
        } catch (e: Exception) {
            logger.error("Error waiting for endpoint to be in service: ${e.message}", e)
            throw IllegalStateException(
                "Endpoint did not reach InService status within timeout: $endpointName. Error: ${e.message}",
                e
            )
        }
        
        logger.info("Endpoint is in service: $endpointName")
        
        // 7. Validate endpoint health with test transaction
        val testTransaction = mapOf(
            "Time" to 0.0,
            "V1" to -1.3598071336738,
            "V2" to -0.0727811733098497,
            "V3" to 2.53634673796914,
            "V4" to 1.37815522427443,
            "V5" to -0.338320769942518,
            "V6" to 0.462387777762292,
            "V7" to 0.239598554061257,
            "V8" to 0.0986979012610507,
            "V9" to 0.363786969611213,
            "V10" to 0.0907941719789316,
            "V11" to -0.551599533260813,
            "V12" to -0.617800855762348,
            "V13" to -0.991389847235408,
            "V14" to -0.311169353699879,
            "V15" to 1.46817697209427,
            "V16" to -0.470400525259478,
            "V17" to 0.207971241929242,
            "V18" to 0.0257905801985591,
            "V19" to 0.403992960255733,
            "V20" to 0.251412098239705,
            "V21" to -0.018306777944153,
            "V22" to 0.277837575558899,
            "V23" to -0.110473910188767,
            "V24" to 0.0669280749146731,
            "V25" to 0.128539358273528,
            "V26" to -0.189114843888824,
            "V27" to 0.133558376740387,
            "V28" to -0.0210530534538215,
            "Amount" to 149.62
        )
        
        logger.info("Performing health check with test transaction")
        
        val healthCheckPrediction = try {
            invokeEndpoint(endpointName, testTransaction)
        } catch (e: Exception) {
            logger.error("Health check failed: ${e.message}", e)
            throw IllegalStateException(
                "Endpoint health check failed for $endpointName. Error: ${e.message}",
                e
            )
        }
        
        logger.info("Health check passed: prediction=$healthCheckPrediction")
        
        // 8. Write endpoint metadata to config bucket for inference pipeline
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", modelName)
            put("deploymentTimestamp", timestamp)
            put("modelAccuracy", modelAccuracy)
        }
        
        logger.info("Writing endpoint metadata to S3: bucket=$configBucket, key=$CONFIG_KEY")
        
        try {
            val putObjectRequest = PutObjectRequest.builder()
                .bucket(configBucket)
                .key(CONFIG_KEY)
                .contentType("application/json")
                .build()
            
            s3Client.putObject(
                putObjectRequest,
                RequestBody.fromString(objectMapper.writeValueAsString(endpointMetadata))
            )
            
            logger.info("Successfully wrote endpoint metadata to S3")
        } catch (e: Exception) {
            logger.error("Failed to write endpoint metadata to S3: ${e.message}", e)
            throw IllegalStateException(
                "Failed to write endpoint metadata to config bucket. Error: ${e.message}",
                e
            )
        }
        
        // 9. Return deployment metadata
        return objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", modelName)
            put("endpointConfigName", endpointConfigName)
            put("deploymentTimestamp", timestamp)
            put("modelAccuracy", modelAccuracy)
            put("healthCheckPrediction", healthCheckPrediction)
        }
    }
    
    /**
     * Invokes the SageMaker endpoint to get a fraud prediction for a transaction.
     * 
     * @param endpointName Name of the SageMaker endpoint
     * @param features Transaction features
     * @return Fraud score (0.0 to 1.0)
     */
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
}
