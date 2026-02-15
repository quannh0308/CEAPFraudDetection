package com.fraud.training

import com.fasterxml.jackson.databind.JsonNode
import com.fraud.common.handler.WorkflowLambdaHandler
import software.amazon.awssdk.services.sagemaker.SageMakerClient
import software.amazon.awssdk.services.sagemaker.model.DescribeTrainingJobRequest
import software.amazon.awssdk.services.sagemaker.model.TrainingJobStatus

/**
 * Lambda handler to check SageMaker training job status.
 * 
 * This handler polls a SageMaker training job and returns its current status.
 * Used in a Step Functions loop to wait for training completion without
 * blocking Lambda execution.
 * 
 * **Input Format:**
 * ```json
 * {
 *   "trainingJobName": "fraud-detection-1234567890",
 *   "modelArtifactPath": "s3://bucket/path/model.tar.gz",
 *   "trainingJobStatus": "InProgress"
 * }
 * ```
 * 
 * **Output Format:**
 * ```json
 * {
 *   "trainingJobName": "fraud-detection-1234567890",
 *   "modelArtifactPath": "s3://bucket/path/model.tar.gz",
 *   "trainingJobStatus": "Completed",
 *   "isComplete": true,
 *   "isFailed": false
 * }
 * ```
 */
class CheckTrainingStatusHandler(
    private val sageMakerClient: SageMakerClient = SageMakerClient.builder().build()
) : WorkflowLambdaHandler() {
    
    override fun processData(input: JsonNode): JsonNode {
        val trainingJobName = input.get("trainingJobName")?.asText()
            ?: throw IllegalArgumentException("trainingJobName is required")
        
        logger.info("Checking status of training job: $trainingJobName")
        
        // Describe the training job
        val response = sageMakerClient.describeTrainingJob(
            DescribeTrainingJobRequest.builder()
                .trainingJobName(trainingJobName)
                .build()
        )
        
        val status = response.trainingJobStatus()
        val secondaryStatus = response.secondaryStatus()
        
        logger.info("Training job $trainingJobName status: $status, secondary: $secondaryStatus")
        
        // Determine if complete or failed
        val isComplete = status == TrainingJobStatus.COMPLETED
        val isFailed = status == TrainingJobStatus.FAILED || status == TrainingJobStatus.STOPPED
        
        // Get model artifact path if completed
        val modelArtifactPath = if (isComplete) {
            response.modelArtifacts().s3ModelArtifacts()
        } else {
            input.get("modelArtifactPath")?.asText() ?: ""
        }
        
        // Get failure reason if failed
        val failureReason = if (isFailed) {
            response.failureReason() ?: "Unknown failure reason"
        } else {
            null
        }
        
        // Pass through testDataPath
        val testDataPath = input.get("testDataPath")?.asText()
        
        return objectMapper.createObjectNode().apply {
            put("trainingJobName", trainingJobName)
            put("modelArtifactPath", modelArtifactPath)
            put("trainingJobStatus", status.toString())
            put("secondaryStatus", secondaryStatus.toString())
            put("isComplete", isComplete)
            put("isFailed", isFailed)
            if (failureReason != null) {
                put("failureReason", failureReason)
            }
            if (testDataPath != null) {
                put("testDataPath", testDataPath)
            }
        }
    }
}
