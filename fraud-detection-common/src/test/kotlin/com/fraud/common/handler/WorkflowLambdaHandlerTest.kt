package com.fraud.common.handler

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ExecutionContext
import com.fraud.common.models.StageResult
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.mockk.mockk

/**
 * Unit tests for WorkflowLambdaHandler base class.
 * 
 * Tests cover:
 * - Successful execution flow (first stage with initialData)
 * - Error handling (processing errors, parsing errors)
 * - ExecutionContext parsing
 * - Input reading from initialData
 */
class WorkflowLambdaHandlerTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    /**
     * Test implementation of WorkflowLambdaHandler that echoes input to output.
     */
    class TestHandler(
        private val shouldThrowError: Boolean = false,
        private val skipS3Write: Boolean = true
    ) : WorkflowLambdaHandler() {
        
        override fun processData(input: JsonNode): JsonNode {
            if (shouldThrowError) {
                throw RuntimeException("Test processing error")
            }
            
            // Echo input to output with additional field
            return objectMapper.createObjectNode().apply {
                set<JsonNode>("input", input)
                put("processed", true)
            }
        }
        
        // Override writeOutput to skip S3 write in tests
        override fun writeOutput(executionContext: ExecutionContext, outputData: JsonNode) {
            if (skipS3Write) {
                // Skip actual S3 write in tests
                logger.info("Test mode: Skipping S3 write for ${executionContext.currentStage}")
            } else {
                super.writeOutput(executionContext, outputData)
            }
        }
        
        // Expose protected methods for testing
        public override fun readInput(executionContext: ExecutionContext): JsonNode {
            return super.readInput(executionContext)
        }
    }
    
    test("handleRequest should successfully process first stage with initialData") {
        val handler = TestHandler()
        val mockContext = mockk<Context>()
        
        val input = mapOf<String, Any>(
            "executionId" to "exec-123",
            "currentStage" to "DataPrepStage",
            "workflowBucket" to "test-bucket",
            "initialData" to mapOf(
                "datasetPath" to "s3://data/dataset.csv",
                "outputPrefix" to "s3://data/prepared/"
            )
        )
        
        val result = handler.handleRequest(input, mockContext)
        
        result.status shouldBe "SUCCESS"
        result.stage shouldBe "DataPrepStage"
        result.isSuccess() shouldBe true
        result.errorMessage shouldBe null
    }
    
    test("handleRequest should return failure when processing throws exception") {
        val handler = TestHandler(shouldThrowError = true)
        val mockContext = mockk<Context>()
        
        val input = mapOf<String, Any>(
            "executionId" to "exec-123",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-bucket",
            "initialData" to mapOf("test" to "data")
        )
        
        val result = handler.handleRequest(input, mockContext)
        
        result.status shouldBe "FAILED"
        result.stage shouldBe "ScoreStage"
        result.isFailure() shouldBe true
        result.errorMessage shouldNotBe null
        result.errorMessage shouldContain "RuntimeException"
        result.errorMessage shouldContain "Test processing error"
    }
    
    test("handleRequest should return failure when ExecutionContext parsing fails") {
        val handler = TestHandler()
        val mockContext = mockk<Context>()
        
        // Invalid input: missing required fields
        val input = mapOf<String, Any>(
            "invalidField" to "value"
        )
        
        val result = handler.handleRequest(input, mockContext)
        
        result.status shouldBe "FAILED"
        result.stage shouldBe "UnknownStage"
        result.isFailure() shouldBe true
        result.errorMessage shouldNotBe null
    }
    
    test("readInput should read from initialData for first stage") {
        val handler = TestHandler()
        
        val executionContext = ExecutionContext(
            executionId = "exec-123",
            currentStage = "DataPrepStage",
            previousStage = null,
            workflowBucket = "test-bucket",
            initialData = mapOf(
                "datasetPath" to "s3://data/dataset.csv",
                "trainSplit" to 0.7
            )
        )
        
        val inputData = handler.readInput(executionContext)
        
        inputData shouldNotBe null
        inputData.get("datasetPath").asText() shouldBe "s3://data/dataset.csv"
        inputData.get("trainSplit").asDouble() shouldBe 0.7
    }
    
    test("readInput should throw exception when first stage has no initialData") {
        val handler = TestHandler()
        
        val executionContext = ExecutionContext(
            executionId = "exec-123",
            currentStage = "DataPrepStage",
            previousStage = null,
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        val exception = shouldThrow<IllegalStateException> {
            handler.readInput(executionContext)
        }
        
        exception.message shouldContain "First stage must have initialData"
        exception.message shouldContain "DataPrepStage"
    }
    
    test("handleRequest should work for all pipeline stages") {
        val stages = listOf(
            "DataPrepStage", "TrainStage", "EvaluateStage", "DeployStage",
            "ScoreStage", "StoreStage", "AlertStage", "MonitorStage"
        )
        
        val mockContext = mockk<Context>()
        
        stages.forEach { stage ->
            val handler = TestHandler()
            
            val input = mapOf<String, Any>(
                "executionId" to "exec-test",
                "currentStage" to stage,
                "workflowBucket" to "test-bucket",
                "initialData" to mapOf("test" to "data")
            )
            
            val result = handler.handleRequest(input, mockContext)
            
            result.status shouldBe "SUCCESS"
            result.stage shouldBe stage
            result.isSuccess() shouldBe true
        }
    }
    
    test("handleRequest should handle complex initialData") {
        val handler = TestHandler()
        val mockContext = mockk<Context>()
        
        val input = mapOf<String, Any>(
            "executionId" to "exec-complex",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to "s3://data/batch.json",
                "batchDate" to "2024-01-15",
                "config" to mapOf(
                    "threshold" to 0.8,
                    "maxBatchSize" to 10000
                )
            )
        )
        
        val result = handler.handleRequest(input, mockContext)
        
        result.status shouldBe "SUCCESS"
        result.stage shouldBe "ScoreStage"
        result.isSuccess() shouldBe true
    }
    
    test("handleRequest should preserve error details") {
        val handler = object : WorkflowLambdaHandler() {
            override fun processData(input: JsonNode): JsonNode {
                throw IllegalArgumentException("Invalid input: missing required field 'modelPath'")
            }
        }
        
        val mockContext = mockk<Context>()
        
        val input = mapOf<String, Any>(
            "executionId" to "exec-error",
            "currentStage" to "DeployStage",
            "workflowBucket" to "test-bucket",
            "initialData" to mapOf("incomplete" to "data")
        )
        
        val result = handler.handleRequest(input, mockContext)
        
        result.status shouldBe "FAILED"
        result.stage shouldBe "DeployStage"
        result.errorMessage shouldContain "IllegalArgumentException"
        result.errorMessage shouldContain "Invalid input"
        result.errorMessage shouldContain "modelPath"
    }
    
    test("handleRequest should handle nested exceptions") {
        val handler = object : WorkflowLambdaHandler() {
            override fun processData(input: JsonNode): JsonNode {
                val cause = IllegalStateException("Endpoint not found")
                throw RuntimeException("Failed to invoke endpoint", cause)
            }
        }
        
        val mockContext = mockk<Context>()
        
        val input = mapOf<String, Any>(
            "executionId" to "exec-nested",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-bucket",
            "initialData" to mapOf("test" to "data")
        )
        
        val result = handler.handleRequest(input, mockContext)
        
        result.status shouldBe "FAILED"
        result.errorMessage shouldContain "RuntimeException"
        result.errorMessage shouldContain "Failed to invoke endpoint"
        result.errorMessage shouldContain "IllegalStateException"
        result.errorMessage shouldContain "Endpoint not found"
    }
    
    test("ExecutionContext should be parsed correctly from input") {
        val handler = TestHandler()
        val mockContext = mockk<Context>()
        
        val input = mapOf<String, Any>(
            "executionId" to "arn:aws:states:us-east-1:123456789012:execution:fraud-training:exec-123",
            "currentStage" to "DataPrepStage",
            "workflowBucket" to "fraud-detection-workflow-123456789012",
            "initialData" to mapOf("test" to "data")
        )
        
        val result = handler.handleRequest(input, mockContext)
        
        result.status shouldBe "SUCCESS"
        result.stage shouldBe "DataPrepStage"
    }
    
    test("readInput should handle various data types in initialData") {
        val handler = TestHandler()
        
        val executionContext = ExecutionContext(
            executionId = "exec-types",
            currentStage = "TestStage",
            previousStage = null,
            workflowBucket = "test-bucket",
            initialData = mapOf(
                "stringValue" to "test",
                "intValue" to 42,
                "doubleValue" to 3.14,
                "boolValue" to true,
                "listValue" to listOf("a", "b", "c"),
                "mapValue" to mapOf("nested" to "value")
            )
        )
        
        val inputData = handler.readInput(executionContext)
        
        inputData.get("stringValue").asText() shouldBe "test"
        inputData.get("intValue").asInt() shouldBe 42
        inputData.get("doubleValue").asDouble() shouldBe 3.14
        inputData.get("boolValue").asBoolean() shouldBe true
        inputData.get("listValue").size() shouldBe 3
        inputData.get("mapValue").get("nested").asText() shouldBe "value"
    }
})
