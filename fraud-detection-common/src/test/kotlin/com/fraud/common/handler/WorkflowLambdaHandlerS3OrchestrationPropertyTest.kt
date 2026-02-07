package com.fraud.common.handler

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ExecutionContext
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import io.mockk.*
import software.amazon.awssdk.core.ResponseInputStream
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.http.AbortableInputStream
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.*
import java.io.ByteArrayInputStream

/**
 * Property-based tests for S3 orchestration in WorkflowLambdaHandler.
 * 
 * This test suite validates the three core S3 orchestration properties:
 * - Property 1: S3 Output Convention
 * - Property 2: S3 Input Convention
 * - Property 3: S3 Path Construction
 * 
 * **Requirements:**
 * - Requirement 2.5: Write output data and metadata to S3
 * - Requirement 3.1: Read prepared data locations from S3
 * - Requirement 13.1: Write output data and metadata to S3
 * - Requirement 13.2: Read input data and metadata from S3
 * - Requirement 13.3: Use consistent S3 key naming conventions
 */
class WorkflowLambdaHandlerS3OrchestrationPropertyTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    /**
     * Test handler that exposes protected methods and allows S3 client mocking.
     */
    class TestHandlerWithMockS3(
        private val mockS3Client: S3Client
    ) : WorkflowLambdaHandler() {
        
        init {
            // Replace the S3 client with mock
            val s3Field = WorkflowLambdaHandler::class.java.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(this, mockS3Client)
        }
        
        override fun processData(input: JsonNode): JsonNode {
            return objectMapper.createObjectNode().apply {
                set<JsonNode>("input", input)
                put("processed", true)
            }
        }
        
        // Expose protected methods for testing
        public override fun readInput(executionContext: ExecutionContext): JsonNode {
            return super.readInput(executionContext)
        }
        
        public override fun writeOutput(executionContext: ExecutionContext, outputData: JsonNode) {
            super.writeOutput(executionContext, outputData)
        }
    }
    
    // ========================================
    // Property 1: S3 Output Convention
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 1: S3 Output Convention
    test("Property 1: S3 Output Convention - output SHALL be written to executions/{executionId}/{currentStage}/output.json") {
        checkAll(100, Arb.executionContext(), Arb.outputData()) { context, outputData ->
            val mockS3Client = mockk<S3Client>()
            val handler = TestHandlerWithMockS3(mockS3Client)
            
            // Mock successful S3 write
            every { 
                mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
            } returns PutObjectResponse.builder().build()
            
            // When the stage completes successfully and writes output
            handler.writeOutput(context, outputData)
            
            // Then the output SHALL be written to the correct S3 path
            val expectedKey = "executions/${context.executionId}/${context.currentStage}/output.json"
            
            verify {
                mockS3Client.putObject(
                    match<PutObjectRequest> { request ->
                        request.bucket() == context.workflowBucket &&
                        request.key() == expectedKey &&
                        request.contentType() == "application/json"
                    },
                    any<RequestBody>()
                )
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 1: S3 Output Convention
    test("Property 1: S3 Output Convention - output data SHALL be valid JSON") {
        checkAll(100, Arb.executionContext(), Arb.outputData()) { context, outputData ->
            val mockS3Client = mockk<S3Client>()
            val handler = TestHandlerWithMockS3(mockS3Client)
            
            var capturedJson: String? = null
            
            // Capture the JSON that was written
            every { 
                mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
            } answers {
                val body = secondArg<RequestBody>()
                capturedJson = String(body.contentStreamProvider().newStream().readAllBytes())
                PutObjectResponse.builder().build()
            }
            
            // When the stage writes output
            handler.writeOutput(context, outputData)
            
            // Then the output SHALL be valid JSON that can be parsed
            capturedJson shouldBe objectMapper.writeValueAsString(outputData)
            
            val parsedBack = objectMapper.readTree(capturedJson)
            parsedBack shouldBe outputData
        }
    }
    
    // ========================================
    // Property 2: S3 Input Convention
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 2: S3 Input Convention
    test("Property 2: S3 Input Convention - input SHALL be read from executions/{executionId}/{previousStage}/output.json") {
        checkAll(100, Arb.nonFirstStageContext(), Arb.inputData()) { context, inputData ->
            val mockS3Client = mockk<S3Client>()
            val handler = TestHandlerWithMockS3(mockS3Client)
            
            val inputJson = objectMapper.writeValueAsString(inputData)
            
            // Mock S3 response
            val responseStream = ResponseInputStream(
                GetObjectResponse.builder().build(),
                AbortableInputStream.create(ByteArrayInputStream(inputJson.toByteArray()))
            )
            
            every { 
                mockS3Client.getObject(any<GetObjectRequest>()) 
            } returns responseStream
            
            // When a non-first stage starts and reads input
            val result = handler.readInput(context)
            
            // Then the input SHALL be read from the correct S3 path
            val expectedKey = "executions/${context.executionId}/${context.previousStage}/output.json"
            
            verify {
                mockS3Client.getObject(
                    match<GetObjectRequest> { request ->
                        request.bucket() == context.workflowBucket &&
                        request.key() == expectedKey
                    }
                )
            }
            
            // And the input data SHALL match what was stored
            result shouldBe inputData
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 2: S3 Input Convention
    test("Property 2: S3 Input Convention - first stage SHALL read from initialData, not S3") {
        checkAll(100, Arb.firstStageContext()) { context ->
            val mockS3Client = mockk<S3Client>()
            val handler = TestHandlerWithMockS3(mockS3Client)
            
            // When the first stage reads input
            val result = handler.readInput(context)
            
            // Then it SHALL read from initialData
            result shouldBe objectMapper.valueToTree(context.initialData)
            
            // And it SHALL NOT call S3
            verify(exactly = 0) {
                mockS3Client.getObject(any<GetObjectRequest>())
            }
        }
    }
    
    // ========================================
    // Property 3: S3 Path Construction
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 3: S3 Path Construction
    test("Property 3: S3 Path Construction - getInputS3Key SHALL follow format executions/{executionId}/{previousStage}/output.json") {
        checkAll(100, Arb.nonFirstStageContext()) { context ->
            // For any non-first stage execution context
            val inputKey = context.getInputS3Key()
            
            // The constructed S3 path SHALL follow the format
            inputKey shouldBe "executions/${context.executionId}/${context.previousStage}/output.json"
            
            // And SHALL contain all required components
            inputKey shouldContain "executions/"
            inputKey shouldContain context.executionId
            inputKey shouldContain context.previousStage!!
            inputKey shouldContain "/output.json"
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 3: S3 Path Construction
    test("Property 3: S3 Path Construction - getOutputS3Key SHALL follow format executions/{executionId}/{currentStage}/output.json") {
        checkAll(100, Arb.executionContext()) { context ->
            // For any execution context
            val outputKey = context.getOutputS3Key()
            
            // The constructed S3 path SHALL follow the format
            outputKey shouldBe "executions/${context.executionId}/${context.currentStage}/output.json"
            
            // And SHALL contain all required components
            outputKey shouldContain "executions/"
            outputKey shouldContain context.executionId
            outputKey shouldContain context.currentStage
            outputKey shouldContain "/output.json"
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 3: S3 Path Construction
    test("Property 3: S3 Path Construction - getInputS3Key SHALL return null for first stage") {
        checkAll(100, Arb.firstStageContext()) { context ->
            // For any first stage execution context
            val inputKey = context.getInputS3Key()
            
            // The input S3 key SHALL be null (reads from initialData instead)
            inputKey shouldBe null
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 3: S3 Path Construction
    test("Property 3: S3 Path Construction - paths SHALL be consistent across read and write operations") {
        checkAll(100, Arb.pipelineStages()) { stages ->
            // For any sequence of pipeline stages
            stages.zipWithNext().forEach { (previousStage, currentStage) ->
                val writeContext = ExecutionContext(
                    executionId = "exec-consistency-test",
                    currentStage = previousStage,
                    previousStage = null,
                    workflowBucket = "test-bucket",
                    initialData = null
                )
                
                val readContext = ExecutionContext(
                    executionId = "exec-consistency-test",
                    currentStage = currentStage,
                    previousStage = previousStage,
                    workflowBucket = "test-bucket",
                    initialData = null
                )
                
                // The output path of the previous stage
                val outputPath = writeContext.getOutputS3Key()
                
                // SHALL match the input path of the current stage
                val inputPath = readContext.getInputS3Key()
                
                outputPath shouldBe inputPath
            }
        }
    }
})

// ========================================
// Arbitrary Generators for Property Tests
// ========================================

/**
 * Generates arbitrary ExecutionContext instances for property testing.
 */
private fun Arb.Companion.executionContext(): Arb<ExecutionContext> {
    return arbitrary { rs ->
        ExecutionContext(
            executionId = Arb.executionId().bind(),
            currentStage = Arb.stageName().bind(),
            previousStage = Arb.stageName().orNull().bind(),
            workflowBucket = Arb.bucketName().bind(),
            initialData = Arb.initialDataMap().orNull().bind()
        )
    }
}

/**
 * Generates arbitrary ExecutionContext instances for non-first stages (has previousStage).
 */
private fun Arb.Companion.nonFirstStageContext(): Arb<ExecutionContext> {
    return arbitrary { rs ->
        ExecutionContext(
            executionId = Arb.executionId().bind(),
            currentStage = Arb.stageName().bind(),
            previousStage = Arb.stageName().bind(), // Always has previousStage
            workflowBucket = Arb.bucketName().bind(),
            initialData = null // Non-first stages don't have initialData
        )
    }
}

/**
 * Generates arbitrary ExecutionContext instances for first stages (no previousStage, has initialData).
 */
private fun Arb.Companion.firstStageContext(): Arb<ExecutionContext> {
    return arbitrary { rs ->
        ExecutionContext(
            executionId = Arb.executionId().bind(),
            currentStage = Arb.stageName().bind(),
            previousStage = null, // First stage has no previousStage
            workflowBucket = Arb.bucketName().bind(),
            initialData = Arb.initialDataMap().bind() // First stage has initialData
        )
    }
}

/**
 * Generates arbitrary execution IDs (can include AWS ARNs or simple IDs).
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
 * Generates arbitrary output data (JsonNode).
 */
private fun Arb.Companion.outputData(): Arb<JsonNode> {
    val objectMapper = jacksonObjectMapper()
    return arbitrary { rs ->
        val data = mapOf(
            "stage" to Arb.stageName().bind(),
            "status" to Arb.of("SUCCESS", "COMPLETED", "PROCESSED").bind(),
            "recordsProcessed" to Arb.int(0..10000).bind(),
            "timestamp" to System.currentTimeMillis(),
            "data" to mapOf(
                "key1" to Arb.string(5..20).bind(),
                "key2" to Arb.double(0.0, 1.0).bind()
            )
        )
        objectMapper.valueToTree(data)
    }
}

/**
 * Generates arbitrary input data (JsonNode).
 */
private fun Arb.Companion.inputData(): Arb<JsonNode> {
    val objectMapper = jacksonObjectMapper()
    return arbitrary { rs ->
        val data = mapOf(
            "dataPath" to "s3://bucket/${Arb.string(10..30).bind()}",
            "config" to mapOf(
                "threshold" to Arb.double(0.0, 1.0).bind(),
                "maxSize" to Arb.int(100..10000).bind()
            ),
            "metadata" to mapOf(
                "version" to "1.0",
                "timestamp" to System.currentTimeMillis()
            )
        )
        objectMapper.valueToTree(data)
    }
}

/**
 * Generates arbitrary sequences of pipeline stages.
 */
private fun Arb.Companion.pipelineStages(): Arb<List<String>> {
    return Arb.of(
        listOf("DataPrepStage", "TrainStage", "EvaluateStage", "DeployStage"),
        listOf("ScoreStage", "StoreStage", "AlertStage", "MonitorStage"),
        listOf("DataPrepStage", "TrainStage"),
        listOf("ScoreStage", "StoreStage"),
        listOf("DataPrepStage", "TrainStage", "EvaluateStage")
    )
}

/**
 * Generates arbitrary values of any type for initialData.
 */
private fun Arb.Companion.any(): Arb<Any> {
    return Arb.choice(
        Arb.string(),
        Arb.int(),
        Arb.double(),
        Arb.boolean(),
        Arb.list(Arb.string(), range = 0..5)
    )
}

/**
 * Generates arbitrary initialData maps.
 */
private fun Arb.Companion.initialDataMap(): Arb<Map<String, Any>> {
    return arbitrary { rs ->
        mapOf(
            "key1" to Arb.string(5..20).bind(),
            "key2" to Arb.int(0..1000).bind(),
            "key3" to Arb.double(0.0, 1.0).bind()
        )
    }
}
