package com.fraud.common.handler

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ExecutionContext
import com.fraud.common.models.StageResult
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import io.mockk.mockk

/**
 * Property-based tests for error handling in WorkflowLambdaHandler.
 * 
 * This test suite validates:
 * - Property 5: Error Handling
 * 
 * **Requirements:**
 * - Requirement 10.4: Handlers SHALL handle errors gracefully and return appropriate error responses
 * - Requirement 15.1: Log errors with full context
 */
class WorkflowLambdaHandlerErrorHandlingPropertyTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    /**
     * Test handler that throws various types of exceptions during processing.
     */
    class ErrorThrowingHandler(
        private val exceptionToThrow: Exception
    ) : WorkflowLambdaHandler() {
        
        override fun processData(input: JsonNode): JsonNode {
            throw exceptionToThrow
        }
        
        // Override writeOutput to skip S3 write in tests
        override fun writeOutput(executionContext: ExecutionContext, outputData: JsonNode) {
            // Skip actual S3 write in tests
            logger.info("Test mode: Skipping S3 write for ${executionContext.currentStage}")
        }
    }
    
    // ========================================
    // Property 5: Error Handling
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 5: Error Handling
    test("Property 5: Error Handling - handler SHALL return StageResult with status FAILED when exception occurs") {
        checkAll(100, Arb.executionContextInput(), Arb.exception()) { input, exception ->
            val handler = ErrorThrowingHandler(exception)
            val mockContext = mockk<Context>()
            
            // When the handler encounters an exception during processing
            val result = handler.handleRequest(input, mockContext)
            
            // Then the handler SHALL return a StageResult with status "FAILED"
            result.status shouldBe StageResult.STATUS_FAILED
            result.isFailure() shouldBe true
            result.isSuccess() shouldBe false
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 5: Error Handling
    test("Property 5: Error Handling - handler SHALL return non-null errorMessage containing exception details") {
        checkAll(100, Arb.executionContextInput(), Arb.exception()) { input, exception ->
            val handler = ErrorThrowingHandler(exception)
            val mockContext = mockk<Context>()
            
            // When the handler encounters an exception during processing
            val result = handler.handleRequest(input, mockContext)
            
            // Then the handler SHALL return a non-null errorMessage
            result.errorMessage shouldNotBe null
            result.errorMessage!! shouldNotBe ""
            
            // And the errorMessage SHALL contain the exception type
            result.errorMessage shouldContain exception.javaClass.simpleName
            
            // And the errorMessage SHALL contain the exception message (if present)
            if (exception.message != null) {
                result.errorMessage shouldContain exception.message!!
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 5: Error Handling
    test("Property 5: Error Handling - handler SHALL preserve stage name in failure result") {
        checkAll(100, Arb.executionContextInput(), Arb.exception()) { input, exception ->
            val handler = ErrorThrowingHandler(exception)
            val mockContext = mockk<Context>()
            
            val expectedStage = input["currentStage"] as String
            
            // When the handler encounters an exception during processing
            val result = handler.handleRequest(input, mockContext)
            
            // Then the handler SHALL preserve the stage name in the result
            result.stage shouldBe expectedStage
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 5: Error Handling
    test("Property 5: Error Handling - handler SHALL include cause details for nested exceptions") {
        checkAll(100, Arb.executionContextInput(), Arb.nestedExceptionPair()) { input, (outerException, causeException) ->
            val handler = ErrorThrowingHandler(outerException)
            val mockContext = mockk<Context>()
            
            // When the handler encounters a nested exception during processing
            val result = handler.handleRequest(input, mockContext)
            
            // Then the errorMessage SHALL contain both outer and cause exception details
            result.errorMessage shouldNotBe null
            result.errorMessage!! shouldContain outerException.javaClass.simpleName
            result.errorMessage shouldContain causeException.javaClass.simpleName
            
            // And SHALL contain both exception messages
            if (outerException.message != null) {
                result.errorMessage shouldContain outerException.message!!
            }
            if (causeException.message != null) {
                result.errorMessage shouldContain causeException.message!!
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 5: Error Handling
    test("Property 5: Error Handling - handler SHALL handle all exception types consistently") {
        checkAll(100, Arb.executionContextInput(), Arb.exceptionType()) { input, exceptionType ->
            val exception = createExceptionOfType(exceptionType)
            val handler = ErrorThrowingHandler(exception)
            val mockContext = mockk<Context>()
            
            // When the handler encounters any type of exception
            val result = handler.handleRequest(input, mockContext)
            
            // Then the handler SHALL always return a FAILED StageResult
            result.status shouldBe StageResult.STATUS_FAILED
            result.errorMessage shouldNotBe null
            
            // And SHALL include the specific exception type in the error message
            result.errorMessage shouldContain exceptionType
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 5: Error Handling
    test("Property 5: Error Handling - handler SHALL handle exceptions with null messages gracefully") {
        checkAll(100, Arb.executionContextInput(), Arb.exceptionType()) { input, exceptionType ->
            // Create exception with null message
            val exception = createExceptionOfType(exceptionType, message = null)
            val handler = ErrorThrowingHandler(exception)
            val mockContext = mockk<Context>()
            
            // When the handler encounters an exception with null message
            val result = handler.handleRequest(input, mockContext)
            
            // Then the handler SHALL still return a valid FAILED result
            result.status shouldBe StageResult.STATUS_FAILED
            result.errorMessage shouldNotBe null
            result.errorMessage!! shouldNotBe ""
            
            // And SHALL include the exception type even without a message
            result.errorMessage shouldContain exceptionType
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 5: Error Handling
    test("Property 5: Error Handling - handler SHALL work for all pipeline stages") {
        checkAll(100, Arb.stageName(), Arb.exception()) { stageName, exception ->
            val input = mapOf<String, Any>(
                "executionId" to "exec-test",
                "currentStage" to stageName,
                "workflowBucket" to "test-bucket",
                "initialData" to mapOf("test" to "data")
            )
            
            val handler = ErrorThrowingHandler(exception)
            val mockContext = mockk<Context>()
            
            // When any pipeline stage encounters an exception
            val result = handler.handleRequest(input, mockContext)
            
            // Then the handler SHALL return a FAILED result with the correct stage name
            result.status shouldBe StageResult.STATUS_FAILED
            result.stage shouldBe stageName
            result.errorMessage shouldNotBe null
        }
    }
})

// ========================================
// Arbitrary Generators for Property Tests
// ========================================

/**
 * Generates arbitrary execution context input maps for Lambda handlers.
 */
private fun Arb.Companion.executionContextInput(): Arb<Map<String, Any>> {
    return arbitrary {
        mapOf<String, Any>(
            "executionId" to Arb.executionId().bind(),
            "currentStage" to Arb.stageName().bind(),
            "workflowBucket" to Arb.bucketName().bind(),
            "initialData" to mapOf(
                "test" to "data",
                "value" to Arb.int(0..1000).bind()
            )
        )
    }
}

/**
 * Generates arbitrary execution IDs.
 */
private fun Arb.Companion.executionId(): Arb<String> {
    return Arb.choice(
        Arb.string(10..50, Codepoint.alphanumeric()).map { "exec-$it" },
        Arb.string(10..20, Codepoint.alphanumeric()).map { id ->
            "arn:aws:states:us-east-1:123456789012:execution:fraud-workflow:$id"
        }
    )
}

/**
 * Generates arbitrary stage names from the fraud detection pipeline.
 */
private fun Arb.Companion.stageName(): Arb<String> {
    return Arb.of(
        "DataPrepStage",
        "TrainStage",
        "EvaluateStage",
        "DeployStage",
        "ScoreStage",
        "StoreStage",
        "AlertStage",
        "MonitorStage"
    )
}

/**
 * Generates arbitrary S3 bucket names.
 */
private fun Arb.Companion.bucketName(): Arb<String> {
    return Arb.choice(
        Arb.string(10..30, Codepoint.az()).map { "fraud-detection-$it" },
        Arb.string(10..20, Codepoint.az()).map { id ->
            "fraud-detection-workflow-$id"
        }
    )
}

/**
 * Generates arbitrary exceptions with various types and messages.
 */
private fun Arb.Companion.exception(): Arb<Exception> {
    return arbitrary {
        val exceptionType = Arb.exceptionType().bind()
        val message = Arb.exceptionMessage().bind()
        createExceptionOfType(exceptionType, message)
    }
}

/**
 * Generates arbitrary exception types commonly encountered in Lambda handlers.
 */
private fun Arb.Companion.exceptionType(): Arb<String> {
    return Arb.of(
        "RuntimeException",
        "IllegalArgumentException",
        "IllegalStateException",
        "NullPointerException",
        "IndexOutOfBoundsException",
        "UnsupportedOperationException",
        "ArithmeticException",
        "ClassCastException",
        "NumberFormatException"
    )
}

/**
 * Generates arbitrary exception messages.
 */
private fun Arb.Companion.exceptionMessage(): Arb<String> {
    return Arb.of(
        "Processing failed",
        "Invalid input data",
        "Missing required field",
        "Endpoint not found",
        "Model artifact not available",
        "S3 object not found",
        "DynamoDB write failed",
        "SageMaker invocation error",
        "Timeout exceeded",
        "Resource not available",
        "Configuration error",
        "Data validation failed",
        "Unexpected null value",
        "Index out of range",
        "Invalid format",
        "Operation not supported"
    )
}

/**
 * Generates pairs of nested exceptions (outer exception with cause).
 */
private fun Arb.Companion.nestedExceptionPair(): Arb<Pair<Exception, Exception>> {
    return arbitrary {
        val causeType = Arb.exceptionType().bind()
        val causeMessage = Arb.exceptionMessage().bind()
        val causeException = createExceptionOfType(causeType, causeMessage)
        
        val outerType = Arb.exceptionType().bind()
        val outerMessage = Arb.exceptionMessage().bind()
        val outerException = createExceptionOfType(outerType, outerMessage, causeException)
        
        Pair(outerException, causeException)
    }
}

/**
 * Creates an exception instance of the specified type.
 */
private fun createExceptionOfType(
    exceptionType: String,
    message: String? = "Test error message",
    cause: Throwable? = null
): Exception {
    return when (exceptionType) {
        "RuntimeException" -> RuntimeException(message, cause)
        "IllegalArgumentException" -> IllegalArgumentException(message, cause)
        "IllegalStateException" -> IllegalStateException(message, cause)
        "NullPointerException" -> NullPointerException(message).apply { if (cause != null) initCause(cause) }
        "IndexOutOfBoundsException" -> IndexOutOfBoundsException(message).apply { if (cause != null) initCause(cause) }
        "UnsupportedOperationException" -> UnsupportedOperationException(message, cause)
        "ArithmeticException" -> ArithmeticException(message).apply { if (cause != null) initCause(cause) }
        "ClassCastException" -> ClassCastException(message).apply { if (cause != null) initCause(cause) }
        "NumberFormatException" -> NumberFormatException(message).apply { if (cause != null) initCause(cause) }
        else -> RuntimeException(message, cause)
    }
}
