package com.fraud.common.handler

import com.amazonaws.services.lambda.runtime.Context
import com.amazonaws.services.lambda.runtime.RequestHandler
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ExecutionContext
import com.fraud.common.models.StageResult
import org.slf4j.LoggerFactory
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.NoSuchKeyException
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.model.S3Exception

/**
 * Abstract base class for all Lambda handlers in the fraud detection pipeline.
 * 
 * This class implements the CEAP WorkflowLambdaHandler pattern, providing:
 * - S3 orchestration for reading inputs and writing outputs
 * - Convention-based S3 path resolution (executions/{executionId}/{stageName}/output.json)
 * - Error handling with StageResult responses
 * - Integration with ExecutionContext for workflow state management
 * 
 * Subclasses must implement the processData method to define stage-specific logic.
 * 
 * **Requirements:**
 * - Requirement 1.3: Lambda handlers SHALL extend WorkflowLambdaHandler base class from CEAP
 * - Requirement 10.2: Handlers SHALL implement handler methods that read inputs from S3 and write outputs to S3
 * - Requirement 10.4: Handlers SHALL handle errors gracefully and return appropriate error responses
 * 
 * **S3 Orchestration Pattern:**
 * 1. First stage reads from ExecutionContext.initialData
 * 2. Subsequent stages read from S3: executions/{executionId}/{previousStage}/output.json
 * 3. All stages write to S3: executions/{executionId}/{currentStage}/output.json
 * 4. Errors are logged and returned as StageResult.failure()
 * 
 * @see ExecutionContext
 * @see StageResult
 */
abstract class WorkflowLambdaHandler : RequestHandler<Map<String, Any>, StageResult> {
    
    protected val logger = LoggerFactory.getLogger(this::class.java)
    protected val objectMapper: ObjectMapper = jacksonObjectMapper()
    protected val s3Client: S3Client = S3Client.builder().build()
    
    /**
     * Abstract method that subclasses must implement to define stage-specific processing logic.
     * 
     * This method receives the input data (either from initialData or previous stage's S3 output)
     * and should return the output data to be written to S3 for the next stage.
     * 
     * @param input Input data as JsonNode (parsed from initialData or S3)
     * @return Output data as JsonNode to be written to S3
     * @throws Exception if processing fails (will be caught and converted to StageResult.failure)
     */
    protected abstract fun processData(input: JsonNode): JsonNode
    
    /**
     * Main Lambda handler method that orchestrates the workflow stage execution.
     * 
     * This method:
     * 1. Parses input to ExecutionContext
     * 2. Calls readInput() to get stage input data
     * 3. Calls processData() (abstract method implemented by subclasses)
     * 4. Calls writeOutput() to save results to S3
     * 5. Returns StageResult.success() or StageResult.failure() on error
     * 
     * @param input Lambda input containing ExecutionContext fields
     * @param context Lambda execution context (provided by AWS)
     * @return StageResult indicating success or failure
     */
    override fun handleRequest(input: Map<String, Any>, context: Context): StageResult {
        var executionContext: ExecutionContext? = null
        
        try {
            // 1. Parse input to ExecutionContext
            executionContext = parseExecutionContext(input)
            
            logger.info(
                "Starting stage execution: executionId=${executionContext.executionId}, " +
                "currentStage=${executionContext.currentStage}, " +
                "previousStage=${executionContext.previousStage}"
            )
            
            // 2. Read input data (from initialData or S3)
            val inputData = readInput(executionContext)
            
            logger.info("Successfully read input data for stage ${executionContext.currentStage}")
            
            // 3. Process data (stage-specific logic)
            val outputData = processData(inputData)
            
            logger.info("Successfully processed data for stage ${executionContext.currentStage}")
            
            // 4. Write output to S3
            writeOutput(executionContext, outputData)
            
            logger.info(
                "Successfully completed stage execution: executionId=${executionContext.executionId}, " +
                "currentStage=${executionContext.currentStage}"
            )
            
            // 5. Return success result
            return StageResult.success(
                stage = executionContext.currentStage,
                recordsProcessed = 0 // Subclasses can override to track records
            )
            
        } catch (e: Exception) {
            val stageName = executionContext?.currentStage ?: "UnknownStage"
            
            logger.error(
                "Stage execution failed: stage=$stageName, " +
                "executionId=${executionContext?.executionId ?: "unknown"}, " +
                "error=${e.message}",
                e
            )
            
            // Return failure result with exception details
            return StageResult.failure(
                stage = stageName,
                exception = e,
                recordsProcessed = 0
            )
        }
    }
    
    /**
     * Parses the Lambda input map into an ExecutionContext object.
     * 
     * @param input Lambda input map
     * @return Parsed ExecutionContext
     * @throws IllegalArgumentException if required fields are missing
     */
    private fun parseExecutionContext(input: Map<String, Any>): ExecutionContext {
        return try {
            val json = objectMapper.writeValueAsString(input)
            objectMapper.readValue(json, ExecutionContext::class.java)
        } catch (e: Exception) {
            throw IllegalArgumentException("Failed to parse ExecutionContext from input: ${e.message}", e)
        }
    }
    
    /**
     * Reads input data for the current stage.
     * 
     * For the first stage (previousStage == null):
     * - Reads from ExecutionContext.initialData
     * 
     * For subsequent stages:
     * - Reads from S3 at path: executions/{executionId}/{previousStage}/output.json
     * 
     * **Error Handling:**
     * - 403 Forbidden: IAM permission issue (permanent error)
     * - 404 NoSuchKey: Previous stage failed or path incorrect (permanent error)
     * - 503 SlowDown: S3 throttling (transient error, will be retried by Step Functions)
     * - 500 InternalError: S3 internal error (transient error, will be retried by Step Functions)
     * 
     * @param executionContext Execution context containing stage information
     * @return Input data as JsonNode
     * @throws IllegalStateException if this is the first stage but initialData is null
     * @throws NoSuchKeyException if S3 object doesn't exist (404)
     * @throws S3Exception for other S3 errors
     */
    protected open fun readInput(executionContext: ExecutionContext): JsonNode {
        // First stage: read from initialData
        if (executionContext.isFirstStage()) {
            val initialData = executionContext.initialData
                ?: throw IllegalStateException(
                    "First stage must have initialData, but it was null. " +
                    "Stage: ${executionContext.currentStage}"
                )
            
            logger.info("Reading input from initialData for first stage: ${executionContext.currentStage}")
            
            return objectMapper.valueToTree(initialData)
        }
        
        // Subsequent stages: read from S3
        val s3Key = executionContext.getInputS3Key()
            ?: throw IllegalStateException(
                "Cannot construct input S3 key: previousStage is null but this is not the first stage"
            )
        
        logger.info(
            "Reading input from S3: bucket=${executionContext.workflowBucket}, key=$s3Key"
        )
        
        try {
            val getObjectRequest = GetObjectRequest.builder()
                .bucket(executionContext.workflowBucket)
                .key(s3Key)
                .build()
            
            val responseBytes = s3Client.getObject(getObjectRequest).readAllBytes()
            val jsonString = String(responseBytes, Charsets.UTF_8)
            
            return objectMapper.readTree(jsonString)
            
        } catch (e: NoSuchKeyException) {
            throw NoSuchKeyException.builder()
                .message(
                    "Previous stage output not found in S3. " +
                    "This usually means the previous stage failed or the S3 path is incorrect. " +
                    "Bucket: ${executionContext.workflowBucket}, Key: $s3Key"
                )
                .cause(e)
                .build()
                
        } catch (e: S3Exception) {
            when (e.statusCode()) {
                403 -> {
                    throw S3Exception.builder()
                        .message(
                            "Access denied to S3 object. Check IAM permissions. " +
                            "Bucket: ${executionContext.workflowBucket}, Key: $s3Key"
                        )
                        .statusCode(403)
                        .cause(e)
                        .build()
                }
                503 -> {
                    throw S3Exception.builder()
                        .message(
                            "S3 throttling (SlowDown). This is a transient error that will be retried. " +
                            "Bucket: ${executionContext.workflowBucket}, Key: $s3Key"
                        )
                        .statusCode(503)
                        .cause(e)
                        .build()
                }
                500 -> {
                    throw S3Exception.builder()
                        .message(
                            "S3 internal error. This is a transient error that will be retried. " +
                            "Bucket: ${executionContext.workflowBucket}, Key: $s3Key"
                        )
                        .statusCode(500)
                        .cause(e)
                        .build()
                }
                else -> {
                    throw S3Exception.builder()
                        .message(
                            "S3 error reading input: ${e.message}. " +
                            "Bucket: ${executionContext.workflowBucket}, Key: $s3Key"
                        )
                        .statusCode(e.statusCode())
                        .cause(e)
                        .build()
                }
            }
        }
    }
    
    /**
     * Writes output data to S3 for the next stage to consume.
     * 
     * Writes to S3 at path: executions/{executionId}/{currentStage}/output.json
     * 
     * **Error Handling:**
     * - 403 Forbidden: IAM permission issue (permanent error)
     * - 503 SlowDown: S3 throttling (transient error, will be retried by Step Functions)
     * - 500 InternalError: S3 internal error (transient error, will be retried by Step Functions)
     * 
     * @param executionContext Execution context containing stage information
     * @param outputData Output data to write
     * @throws S3Exception if S3 write fails
     */
    protected open fun writeOutput(executionContext: ExecutionContext, outputData: JsonNode) {
        val s3Key = executionContext.getOutputS3Key()
        
        logger.info(
            "Writing output to S3: bucket=${executionContext.workflowBucket}, key=$s3Key"
        )
        
        try {
            val jsonString = objectMapper.writeValueAsString(outputData)
            
            val putObjectRequest = PutObjectRequest.builder()
                .bucket(executionContext.workflowBucket)
                .key(s3Key)
                .contentType("application/json")
                .build()
            
            s3Client.putObject(putObjectRequest, RequestBody.fromString(jsonString))
            
            logger.info("Successfully wrote output to S3: key=$s3Key")
            
        } catch (e: S3Exception) {
            when (e.statusCode()) {
                403 -> {
                    throw S3Exception.builder()
                        .message(
                            "Access denied writing to S3. Check IAM permissions. " +
                            "Bucket: ${executionContext.workflowBucket}, Key: $s3Key"
                        )
                        .statusCode(403)
                        .cause(e)
                        .build()
                }
                503 -> {
                    throw S3Exception.builder()
                        .message(
                            "S3 throttling (SlowDown) writing output. This is a transient error that will be retried. " +
                            "Bucket: ${executionContext.workflowBucket}, Key: $s3Key"
                        )
                        .statusCode(503)
                        .cause(e)
                        .build()
                }
                500 -> {
                    throw S3Exception.builder()
                        .message(
                            "S3 internal error writing output. This is a transient error that will be retried. " +
                            "Bucket: ${executionContext.workflowBucket}, Key: $s3Key"
                        )
                        .statusCode(500)
                        .cause(e)
                        .build()
                }
                else -> {
                    throw S3Exception.builder()
                        .message(
                            "S3 error writing output: ${e.message}. " +
                            "Bucket: ${executionContext.workflowBucket}, Key: $s3Key"
                        )
                        .statusCode(e.statusCode())
                        .cause(e)
                        .build()
                }
            }
        }
    }
}
