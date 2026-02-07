package com.fraud.common.models

import com.fasterxml.jackson.annotation.JsonIgnore
import com.fasterxml.jackson.annotation.JsonProperty

/**
 * Represents the result of a workflow stage execution in the fraud detection pipeline.
 * 
 * This data class is returned by Lambda handlers to indicate whether a stage completed
 * successfully or failed. It provides essential information for error handling, monitoring,
 * and debugging workflow executions.
 * 
 * The StageResult is used by:
 * - WorkflowLambdaHandler to communicate execution status to Step Functions
 * - Error handling logic to preserve failure details in S3 (Requirement 15.3)
 * - Monitoring systems to track stage performance and failure rates
 * 
 * @property status Execution status: "SUCCESS" or "FAILED"
 * @property stage Name of the workflow stage that executed (e.g., "ScoreStage", "TrainStage")
 * @property recordsProcessed Number of records/items processed by this stage (0 if not applicable)
 * @property errorMessage Detailed error message if status is "FAILED", null otherwise
 */
data class StageResult(
    @JsonProperty("status")
    val status: String,
    
    @JsonProperty("stage")
    val stage: String,
    
    @JsonProperty("recordsProcessed")
    val recordsProcessed: Int,
    
    @JsonProperty("errorMessage")
    val errorMessage: String? = null
) {
    companion object {
        const val STATUS_SUCCESS = "SUCCESS"
        const val STATUS_FAILED = "FAILED"
        
        /**
         * Creates a successful StageResult.
         * 
         * @param stage Name of the workflow stage
         * @param recordsProcessed Number of records processed (default 0)
         * @return StageResult with SUCCESS status
         */
        fun success(stage: String, recordsProcessed: Int = 0): StageResult {
            return StageResult(
                status = STATUS_SUCCESS,
                stage = stage,
                recordsProcessed = recordsProcessed,
                errorMessage = null
            )
        }
        
        /**
         * Creates a failed StageResult with error details.
         * 
         * @param stage Name of the workflow stage
         * @param errorMessage Detailed error message describing the failure
         * @param recordsProcessed Number of records processed before failure (default 0)
         * @return StageResult with FAILED status
         */
        fun failure(stage: String, errorMessage: String, recordsProcessed: Int = 0): StageResult {
            return StageResult(
                status = STATUS_FAILED,
                stage = stage,
                recordsProcessed = recordsProcessed,
                errorMessage = errorMessage
            )
        }
        
        /**
         * Creates a failed StageResult from an exception.
         * 
         * @param stage Name of the workflow stage
         * @param exception The exception that caused the failure
         * @param recordsProcessed Number of records processed before failure (default 0)
         * @return StageResult with FAILED status and exception details
         */
        fun failure(stage: String, exception: Throwable, recordsProcessed: Int = 0): StageResult {
            val errorMessage = buildString {
                append(exception.javaClass.simpleName)
                append(": ")
                append(exception.message ?: "No error message")
                exception.cause?.let { cause ->
                    append(" (caused by: ")
                    append(cause.javaClass.simpleName)
                    append(": ")
                    append(cause.message ?: "No error message")
                    append(")")
                }
            }
            
            return StageResult(
                status = STATUS_FAILED,
                stage = stage,
                recordsProcessed = recordsProcessed,
                errorMessage = errorMessage
            )
        }
    }
    
    init {
        require(status == STATUS_SUCCESS || status == STATUS_FAILED) {
            "Status must be either '$STATUS_SUCCESS' or '$STATUS_FAILED', but was '$status'"
        }
        
        require(recordsProcessed >= 0) {
            "Records processed must be non-negative, but was $recordsProcessed"
        }
        
        if (status == STATUS_FAILED) {
            require(!errorMessage.isNullOrBlank()) {
                "Error message must be provided when status is FAILED"
            }
        }
    }
    
    /**
     * Determines if this stage execution was successful.
     */
    @JsonIgnore
    fun isSuccess(): Boolean = status == STATUS_SUCCESS
    
    /**
     * Determines if this stage execution failed.
     */
    @JsonIgnore
    fun isFailure(): Boolean = status == STATUS_FAILED
}
