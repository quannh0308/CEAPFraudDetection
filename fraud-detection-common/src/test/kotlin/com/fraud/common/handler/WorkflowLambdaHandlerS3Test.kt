package com.fraud.common.handler

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ExecutionContext
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.mockk.*
import software.amazon.awssdk.core.ResponseInputStream
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.http.AbortableInputStream
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.*
import java.io.ByteArrayInputStream

/**
 * Tests for S3 input reading logic in WorkflowLambdaHandler.
 * 
 * This test suite validates:
 * - Reading from S3 for non-first stages
 * - S3 error handling (403, 404, 503, 500)
 * - Convention-based path resolution
 * - Error messages with context
 * 
 * **Requirements:**
 * - Requirement 2.3: Split dataset into training (70%), validation (15%), and test (15%) sets
 * - Requirement 2.4: Write prepared datasets to S3 in format compatible with SageMaker
 * - Requirement 3.1: Read prepared data locations from S3
 * - Requirement 13.2: Read input data and metadata from S3
 * - Requirement 15.1: Log errors with full context
 */
class WorkflowLambdaHandlerS3Test : FunSpec({
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
        
        // Override writeOutput to skip S3 write in tests
        override fun writeOutput(executionContext: ExecutionContext, outputData: JsonNode) {
            logger.info("Test mode: Skipping S3 write for ${executionContext.currentStage}")
        }
        
        // Expose protected methods for testing
        public override fun readInput(executionContext: ExecutionContext): JsonNode {
            return super.readInput(executionContext)
        }
    }
    
    test("readInput should read from S3 for non-first stage") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-123",
            currentStage = "TrainStage",
            previousStage = "DataPrepStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        val s3Output = mapOf(
            "trainDataPath" to "s3://data/train.parquet",
            "validationDataPath" to "s3://data/validation.parquet",
            "testDataPath" to "s3://data/test.parquet"
        )
        val s3OutputJson = objectMapper.writeValueAsString(s3Output)
        
        // Mock S3 response
        val responseStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(s3OutputJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(any<GetObjectRequest>()) 
        } returns responseStream
        
        val inputData = handler.readInput(executionContext)
        
        inputData.get("trainDataPath").asText() shouldBe "s3://data/train.parquet"
        inputData.get("validationDataPath").asText() shouldBe "s3://data/validation.parquet"
        inputData.get("testDataPath").asText() shouldBe "s3://data/test.parquet"
        
        // Verify S3 was called with correct path
        verify {
            mockS3Client.getObject(match<GetObjectRequest> { request ->
                request.bucket() == "test-bucket" &&
                request.key() == "executions/exec-123/DataPrepStage/output.json"
            })
        }
    }
    
    test("readInput should throw NoSuchKeyException with context when S3 object not found (404)") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-404",
            currentStage = "TrainStage",
            previousStage = "DataPrepStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        // Mock S3 404 error
        every { 
            mockS3Client.getObject(any<GetObjectRequest>()) 
        } throws NoSuchKeyException.builder()
            .message("The specified key does not exist")
            .build()
        
        val exception = shouldThrow<NoSuchKeyException> {
            handler.readInput(executionContext)
        }
        
        exception.message shouldContain "Previous stage output not found in S3"
        exception.message shouldContain "test-bucket"
        exception.message shouldContain "executions/exec-404/DataPrepStage/output.json"
    }
    
    test("readInput should throw S3Exception with context when access denied (403)") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-403",
            currentStage = "ScoreStage",
            previousStage = "DeployStage",
            workflowBucket = "restricted-bucket",
            initialData = null
        )
        
        // Mock S3 403 error
        every { 
            mockS3Client.getObject(any<GetObjectRequest>()) 
        } throws S3Exception.builder()
            .message("Access Denied")
            .statusCode(403)
            .build()
        
        val exception = shouldThrow<S3Exception> {
            handler.readInput(executionContext)
        }
        
        exception.statusCode() shouldBe 403
        exception.message shouldContain "Access denied to S3 object"
        exception.message shouldContain "Check IAM permissions"
        exception.message shouldContain "restricted-bucket"
        exception.message shouldContain "executions/exec-403/DeployStage/output.json"
    }
    
    test("readInput should throw S3Exception with context when S3 throttling occurs (503)") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-503",
            currentStage = "StoreStage",
            previousStage = "ScoreStage",
            workflowBucket = "busy-bucket",
            initialData = null
        )
        
        // Mock S3 503 error (SlowDown)
        every { 
            mockS3Client.getObject(any<GetObjectRequest>()) 
        } throws S3Exception.builder()
            .message("Please reduce your request rate")
            .statusCode(503)
            .build()
        
        val exception = shouldThrow<S3Exception> {
            handler.readInput(executionContext)
        }
        
        exception.statusCode() shouldBe 503
        exception.message shouldContain "S3 throttling (SlowDown)"
        exception.message shouldContain "transient error that will be retried"
        exception.message shouldContain "busy-bucket"
        exception.message shouldContain "executions/exec-503/ScoreStage/output.json"
    }
    
    test("readInput should throw S3Exception with context when S3 internal error occurs (500)") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-500",
            currentStage = "AlertStage",
            previousStage = "StoreStage",
            workflowBucket = "error-bucket",
            initialData = null
        )
        
        // Mock S3 500 error
        every { 
            mockS3Client.getObject(any<GetObjectRequest>()) 
        } throws S3Exception.builder()
            .message("We encountered an internal error. Please try again.")
            .statusCode(500)
            .build()
        
        val exception = shouldThrow<S3Exception> {
            handler.readInput(executionContext)
        }
        
        exception.statusCode() shouldBe 500
        exception.message shouldContain "S3 internal error"
        exception.message shouldContain "transient error that will be retried"
        exception.message shouldContain "error-bucket"
        exception.message shouldContain "executions/exec-500/StoreStage/output.json"
    }
    
    test("readInput should handle other S3 errors with generic message") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-other",
            currentStage = "MonitorStage",
            previousStage = "AlertStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        // Mock S3 error with different status code
        every { 
            mockS3Client.getObject(any<GetObjectRequest>()) 
        } throws S3Exception.builder()
            .message("Invalid request")
            .statusCode(400)
            .build()
        
        val exception = shouldThrow<S3Exception> {
            handler.readInput(executionContext)
        }
        
        exception.statusCode() shouldBe 400
        exception.message shouldContain "S3 error reading input"
        exception.message shouldContain "test-bucket"
        exception.message shouldContain "executions/exec-other/AlertStage/output.json"
    }
    
    test("readInput should use convention-based S3 path for all stages") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        val stages = listOf(
            "DataPrepStage" to "InitialStage",
            "TrainStage" to "DataPrepStage",
            "EvaluateStage" to "TrainStage",
            "DeployStage" to "EvaluateStage",
            "ScoreStage" to "DeployStage",
            "StoreStage" to "ScoreStage",
            "AlertStage" to "ScoreStage",  // AlertStage reads from ScoreStage, not StoreStage
            "MonitorStage" to "StoreStage"
        )
        
        stages.forEach { (currentStage, previousStage) ->
            val executionContext = ExecutionContext(
                executionId = "exec-convention",
                currentStage = currentStage,
                previousStage = previousStage,
                workflowBucket = "convention-bucket",
                initialData = null
            )
            
            val s3Output = mapOf("data" to "test")
            val s3OutputJson = objectMapper.writeValueAsString(s3Output)
            
            val responseStream = ResponseInputStream(
                GetObjectResponse.builder().build(),
                AbortableInputStream.create(ByteArrayInputStream(s3OutputJson.toByteArray()))
            )
            
            every { 
                mockS3Client.getObject(any<GetObjectRequest>()) 
            } returns responseStream
            
            handler.readInput(executionContext)
            
            // Verify correct S3 path was used
            verify {
                mockS3Client.getObject(match<GetObjectRequest> { request ->
                    request.bucket() == "convention-bucket" &&
                    request.key() == "executions/exec-convention/$previousStage/output.json"
                })
            }
            
            clearMocks(mockS3Client, answers = false)
        }
    }
    
    test("readInput should parse complex JSON from S3") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-complex",
            currentStage = "ScoreStage",
            previousStage = "DeployStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        val complexOutput = mapOf(
            "endpointName" to "fraud-detection-prod",
            "modelName" to "fraud-detection-prod-1234567890",
            "deploymentTimestamp" to 1234567890,
            "modelAccuracy" to 0.9523,
            "healthCheckPrediction" to 0.0234,
            "metadata" to mapOf(
                "version" to "1.0",
                "features" to listOf("V1", "V2", "V3")
            )
        )
        val s3OutputJson = objectMapper.writeValueAsString(complexOutput)
        
        val responseStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(s3OutputJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(any<GetObjectRequest>()) 
        } returns responseStream
        
        val inputData = handler.readInput(executionContext)
        
        inputData.get("endpointName").asText() shouldBe "fraud-detection-prod"
        inputData.get("modelAccuracy").asDouble() shouldBe 0.9523
        inputData.get("metadata").get("version").asText() shouldBe "1.0"
        inputData.get("metadata").get("features").size() shouldBe 3
    }
    
    test("readInput should handle large JSON payloads from S3") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-large",
            currentStage = "StoreStage",
            previousStage = "ScoreStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        // Create large payload with 1000 transactions
        val transactions = (1..1000).map { i ->
            mapOf(
                "transactionId" to "txn-$i",
                "timestamp" to 1705334400000L + i,
                "amount" to (100.0 + i),
                "fraudScore" to (i % 100) / 100.0,
                "features" to mapOf("V1" to i.toDouble(), "V2" to (i * 2).toDouble())
            )
        }
        
        val largeOutput = mapOf(
            "scoredTransactions" to transactions,
            "batchDate" to "2024-01-15",
            "transactionCount" to 1000
        )
        val s3OutputJson = objectMapper.writeValueAsString(largeOutput)
        
        val responseStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(s3OutputJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(any<GetObjectRequest>()) 
        } returns responseStream
        
        val inputData = handler.readInput(executionContext)
        
        inputData.get("transactionCount").asInt() shouldBe 1000
        inputData.get("scoredTransactions").size() shouldBe 1000
        inputData.get("scoredTransactions").get(0).get("transactionId").asText() shouldBe "txn-1"
        inputData.get("scoredTransactions").get(999).get("transactionId").asText() shouldBe "txn-1000"
    }
    
    test("handleRequest should fail gracefully when S3 read fails") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        val mockContext = mockk<Context>()
        
        val input = mapOf<String, Any>(
            "executionId" to "exec-fail",
            "currentStage" to "TrainStage",
            "previousStage" to "DataPrepStage",
            "workflowBucket" to "test-bucket"
        )
        
        // Mock S3 404 error
        every { 
            mockS3Client.getObject(any<GetObjectRequest>()) 
        } throws NoSuchKeyException.builder()
            .message("The specified key does not exist")
            .build()
        
        val result = handler.handleRequest(input, mockContext)
        
        result.status shouldBe "FAILED"
        result.stage shouldBe "TrainStage"
        result.isFailure() shouldBe true
        result.errorMessage shouldContain "NoSuchKeyException"
        result.errorMessage shouldContain "Previous stage output not found"
    }
    
    test("readInput should throw exception when previousStage is null for non-first stage") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerWithMockS3(mockS3Client)
        
        // This is an invalid state: not first stage (has no initialData) but no previousStage
        val executionContext = ExecutionContext(
            executionId = "exec-invalid",
            currentStage = "TrainStage",
            previousStage = null,  // Invalid: should have previousStage
            workflowBucket = "test-bucket",
            initialData = null  // Invalid: should have initialData if first stage
        )
        
        val exception = shouldThrow<IllegalStateException> {
            handler.readInput(executionContext)
        }
        
        exception.message shouldContain "First stage must have initialData"
    }
    
    // ========================================
    // S3 Output Writing Tests
    // ========================================
    
    /**
     * Test handler that exposes writeOutput method for testing.
     */
    class TestHandlerForWriteOutput(
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
                put("processed", true)
            }
        }
        
        // Expose protected method for testing
        public override fun writeOutput(executionContext: ExecutionContext, outputData: JsonNode) {
            super.writeOutput(executionContext, outputData)
        }
    }
    
    test("writeOutput should write to S3 with convention-based path") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-write-123",
            currentStage = "TrainStage",
            previousStage = "DataPrepStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        val outputData = objectMapper.createObjectNode().apply {
            put("trainingJobName", "fraud-detection-1234567890")
            put("modelArtifactPath", "s3://models/model.tar.gz")
            put("trainingJobStatus", "Completed")
        }
        
        // Mock successful S3 write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        handler.writeOutput(executionContext, outputData)
        
        // Verify S3 was called with correct path and content
        verify {
            mockS3Client.putObject(
                match<PutObjectRequest> { request ->
                    request.bucket() == "test-bucket" &&
                    request.key() == "executions/exec-write-123/TrainStage/output.json" &&
                    request.contentType() == "application/json"
                },
                match<RequestBody> { body ->
                    val content = String(body.contentStreamProvider().newStream().readAllBytes())
                    content.contains("fraud-detection-1234567890") &&
                    content.contains("s3://models/model.tar.gz")
                }
            )
        }
    }
    
    test("writeOutput should throw S3Exception with context when access denied (403)") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-write-403",
            currentStage = "DeployStage",
            previousStage = "EvaluateStage",
            workflowBucket = "restricted-bucket",
            initialData = null
        )
        
        val outputData = objectMapper.createObjectNode().apply {
            put("endpointName", "fraud-detection-prod")
        }
        
        // Mock S3 403 error
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } throws S3Exception.builder()
            .message("Access Denied")
            .statusCode(403)
            .build()
        
        val exception = shouldThrow<S3Exception> {
            handler.writeOutput(executionContext, outputData)
        }
        
        exception.statusCode() shouldBe 403
        exception.message shouldContain "Access denied writing to S3"
        exception.message shouldContain "Check IAM permissions"
        exception.message shouldContain "restricted-bucket"
        exception.message shouldContain "executions/exec-write-403/DeployStage/output.json"
    }
    
    test("writeOutput should throw S3Exception with context when S3 throttling occurs (503)") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-write-503",
            currentStage = "ScoreStage",
            previousStage = "DeployStage",
            workflowBucket = "busy-bucket",
            initialData = null
        )
        
        val outputData = objectMapper.createObjectNode().apply {
            put("transactionCount", 1000)
        }
        
        // Mock S3 503 error (SlowDown)
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } throws S3Exception.builder()
            .message("Please reduce your request rate")
            .statusCode(503)
            .build()
        
        val exception = shouldThrow<S3Exception> {
            handler.writeOutput(executionContext, outputData)
        }
        
        exception.statusCode() shouldBe 503
        exception.message shouldContain "S3 throttling (SlowDown) writing output"
        exception.message shouldContain "transient error that will be retried"
        exception.message shouldContain "busy-bucket"
        exception.message shouldContain "executions/exec-write-503/ScoreStage/output.json"
    }
    
    test("writeOutput should throw S3Exception with context when S3 internal error occurs (500)") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-write-500",
            currentStage = "StoreStage",
            previousStage = "ScoreStage",
            workflowBucket = "error-bucket",
            initialData = null
        )
        
        val outputData = objectMapper.createObjectNode().apply {
            put("successCount", 100)
        }
        
        // Mock S3 500 error
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } throws S3Exception.builder()
            .message("We encountered an internal error. Please try again.")
            .statusCode(500)
            .build()
        
        val exception = shouldThrow<S3Exception> {
            handler.writeOutput(executionContext, outputData)
        }
        
        exception.statusCode() shouldBe 500
        exception.message shouldContain "S3 internal error writing output"
        exception.message shouldContain "transient error that will be retried"
        exception.message shouldContain "error-bucket"
        exception.message shouldContain "executions/exec-write-500/StoreStage/output.json"
    }
    
    test("writeOutput should handle other S3 errors with generic message") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-write-other",
            currentStage = "AlertStage",
            previousStage = "ScoreStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        val outputData = objectMapper.createObjectNode().apply {
            put("alertsSent", 5)
        }
        
        // Mock S3 error with different status code
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } throws S3Exception.builder()
            .message("Invalid request")
            .statusCode(400)
            .build()
        
        val exception = shouldThrow<S3Exception> {
            handler.writeOutput(executionContext, outputData)
        }
        
        exception.statusCode() shouldBe 400
        exception.message shouldContain "S3 error writing output"
        exception.message shouldContain "test-bucket"
        exception.message shouldContain "executions/exec-write-other/AlertStage/output.json"
    }
    
    test("writeOutput should use convention-based S3 path for all stages") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val stages = listOf(
            "DataPrepStage",
            "TrainStage",
            "EvaluateStage",
            "DeployStage",
            "ScoreStage",
            "StoreStage",
            "AlertStage",
            "MonitorStage"
        )
        
        stages.forEach { currentStage ->
            val executionContext = ExecutionContext(
                executionId = "exec-write-convention",
                currentStage = currentStage,
                previousStage = "PreviousStage",
                workflowBucket = "convention-bucket",
                initialData = null
            )
            
            val outputData = objectMapper.createObjectNode().apply {
                put("stage", currentStage)
                put("data", "test")
            }
            
            // Mock successful S3 write
            every { 
                mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
            } returns PutObjectResponse.builder().build()
            
            handler.writeOutput(executionContext, outputData)
            
            // Verify correct S3 path was used
            verify {
                mockS3Client.putObject(
                    match<PutObjectRequest> { request ->
                        request.bucket() == "convention-bucket" &&
                        request.key() == "executions/exec-write-convention/$currentStage/output.json"
                    },
                    any<RequestBody>()
                )
            }
            
            clearMocks(mockS3Client, answers = false)
        }
    }
    
    test("writeOutput should serialize complex JSON correctly") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-write-complex",
            currentStage = "ScoreStage",
            previousStage = "DeployStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        val complexOutput = objectMapper.createObjectNode().apply {
            put("batchDate", "2024-01-15")
            put("transactionCount", 1000)
            put("endpointName", "fraud-detection-prod")
            set<JsonNode>("metadata", objectMapper.createObjectNode().apply {
                put("version", "1.0")
                set<JsonNode>("features", objectMapper.createArrayNode().apply {
                    add("V1")
                    add("V2")
                    add("V3")
                })
            })
        }
        
        // Mock successful S3 write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        handler.writeOutput(executionContext, complexOutput)
        
        // Verify JSON was serialized correctly
        verify {
            mockS3Client.putObject(
                any<PutObjectRequest>(),
                match<RequestBody> { body ->
                    val content = String(body.contentStreamProvider().newStream().readAllBytes())
                    val parsed = objectMapper.readTree(content)
                    parsed.get("batchDate").asText() == "2024-01-15" &&
                    parsed.get("transactionCount").asInt() == 1000 &&
                    parsed.get("metadata").get("features").size() == 3
                }
            )
        }
    }
    
    test("writeOutput should handle large JSON payloads") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-write-large",
            currentStage = "ScoreStage",
            previousStage = "DeployStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        // Create large payload with 1000 transactions
        val largeOutput = objectMapper.createObjectNode().apply {
            put("batchDate", "2024-01-15")
            put("transactionCount", 1000)
            set<JsonNode>("scoredTransactions", objectMapper.createArrayNode().apply {
                (1..1000).forEach { i ->
                    add(objectMapper.createObjectNode().apply {
                        put("transactionId", "txn-$i")
                        put("timestamp", 1705334400000L + i)
                        put("amount", 100.0 + i)
                        put("fraudScore", (i % 100) / 100.0)
                    })
                }
            })
        }
        
        // Mock successful S3 write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        handler.writeOutput(executionContext, largeOutput)
        
        // Verify large payload was written
        verify {
            mockS3Client.putObject(
                any<PutObjectRequest>(),
                match<RequestBody> { body ->
                    val content = String(body.contentStreamProvider().newStream().readAllBytes())
                    val parsed = objectMapper.readTree(content)
                    parsed.get("transactionCount").asInt() == 1000 &&
                    parsed.get("scoredTransactions").size() == 1000
                }
            )
        }
    }
    
    test("writeOutput should set correct content type") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-write-content-type",
            currentStage = "TrainStage",
            previousStage = "DataPrepStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        val outputData = objectMapper.createObjectNode().apply {
            put("status", "success")
        }
        
        // Mock successful S3 write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        handler.writeOutput(executionContext, outputData)
        
        // Verify content type is set to application/json
        verify {
            mockS3Client.putObject(
                match<PutObjectRequest> { request ->
                    request.contentType() == "application/json"
                },
                any<RequestBody>()
            )
        }
    }
    
    test("handleRequest should fail gracefully when S3 write fails") {
        val mockS3Client = mockk<S3Client>()
        
        // Create handler that doesn't override writeOutput
        val handler = object : WorkflowLambdaHandler() {
            init {
                val s3Field = WorkflowLambdaHandler::class.java.getDeclaredField("s3Client")
                s3Field.isAccessible = true
                s3Field.set(this, mockS3Client)
            }
            
            override fun processData(input: JsonNode): JsonNode {
                return objectMapper.createObjectNode().apply {
                    put("processed", true)
                }
            }
        }
        
        val mockContext = mockk<Context>()
        
        val input = mutableMapOf<String, Any?>(
            "executionId" to "exec-write-fail",
            "currentStage" to "TrainStage",
            "previousStage" to null,
            "workflowBucket" to "test-bucket",
            "initialData" to mapOf("data" to "test")
        )
        
        // Mock S3 403 error on write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } throws S3Exception.builder()
            .message("Access Denied")
            .statusCode(403)
            .build()
        
        @Suppress("UNCHECKED_CAST")
        val result = handler.handleRequest(input as Map<String, Any>, mockContext)
        
        result.status shouldBe "FAILED"
        result.stage shouldBe "TrainStage"
        result.isFailure() shouldBe true
        result.errorMessage shouldContain "S3Exception"
        result.errorMessage shouldContain "Access denied writing to S3"
    }
    
    test("writeOutput should write valid JSON that can be read back") {
        val mockS3Client = mockk<S3Client>()
        val handler = TestHandlerForWriteOutput(mockS3Client)
        
        val executionContext = ExecutionContext(
            executionId = "exec-write-roundtrip",
            currentStage = "EvaluateStage",
            previousStage = "TrainStage",
            workflowBucket = "test-bucket",
            initialData = null
        )
        
        val outputData = objectMapper.createObjectNode().apply {
            put("modelArtifactPath", "s3://models/model.tar.gz")
            put("accuracy", 0.9523)
            put("precision", 0.8912)
            put("recall", 0.8456)
            put("f1Score", 0.8678)
            put("auc", 0.9234)
        }
        
        var capturedJson: String? = null
        
        // Capture the JSON that was written
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } answers {
            val body = secondArg<RequestBody>()
            capturedJson = String(body.contentStreamProvider().newStream().readAllBytes())
            PutObjectResponse.builder().build()
        }
        
        handler.writeOutput(executionContext, outputData)
        
        // Verify we can parse the JSON back
        capturedJson shouldBe objectMapper.writeValueAsString(outputData)
        
        val parsedBack = objectMapper.readTree(capturedJson)
        parsedBack.get("accuracy").asDouble() shouldBe 0.9523
        parsedBack.get("precision").asDouble() shouldBe 0.8912
    }
})
