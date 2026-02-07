package com.fraud.common.models

import com.fasterxml.jackson.annotation.JsonProperty

/**
 * Represents the execution context for a workflow stage in the fraud detection pipeline.
 * 
 * This data class is used by the WorkflowLambdaHandler base class to manage S3 orchestration
 * between workflow stages. It tracks the current execution state and provides the necessary
 * information for reading inputs from the previous stage and writing outputs for the next stage.
 * 
 * The ExecutionContext follows CEAP's convention-based S3 orchestration pattern where:
 * - First stage reads from initialData
 * - Subsequent stages read from S3 path: executions/{executionId}/{previousStage}/output.json
 * - All stages write to S3 path: executions/{executionId}/{currentStage}/output.json
 * 
 * @property executionId Unique identifier for this workflow execution (e.g., Step Functions execution ARN)
 * @property currentStage Name of the current workflow stage being executed (e.g., "ScoreStage", "TrainStage")
 * @property previousStage Name of the previous workflow stage, or null if this is the first stage
 * @property workflowBucket S3 bucket name where workflow orchestration data is stored
 * @property initialData Initial input data for the first stage, or null for subsequent stages
 */
data class ExecutionContext(
    @JsonProperty("executionId")
    var executionId: String = "",
    
    @JsonProperty("currentStage")
    var currentStage: String = "",
    
    @JsonProperty("previousStage")
    var previousStage: String? = null,
    
    @JsonProperty("workflowBucket")
    var workflowBucket: String = "",
    
    @JsonProperty("initialData")
    var initialData: Map<String, Any>? = null
) {
    /**
     * Determines if this is the first stage in the workflow.
     * The first stage has no previous stage and reads from initialData.
     */
    fun isFirstStage(): Boolean = previousStage == null
    
    /**
     * Constructs the S3 key path for reading input from the previous stage.
     * Returns null if this is the first stage (should read from initialData instead).
     * 
     * @return S3 key path in format "executions/{executionId}/{previousStage}/output.json"
     */
    fun getInputS3Key(): String? {
        return if (previousStage != null) {
            "executions/$executionId/$previousStage/output.json"
        } else {
            null
        }
    }
    
    /**
     * Constructs the S3 key path for writing output from the current stage.
     * 
     * @return S3 key path in format "executions/{executionId}/{currentStage}/output.json"
     */
    fun getOutputS3Key(): String {
        return "executions/$executionId/$currentStage/output.json"
    }
}
