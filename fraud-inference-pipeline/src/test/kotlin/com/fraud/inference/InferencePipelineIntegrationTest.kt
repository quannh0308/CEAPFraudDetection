package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ScoredTransaction
import com.fraud.common.models.Transaction
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.mockk.*
import software.amazon.awssdk.core.ResponseInputStream
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.http.AbortableInputStream
import software.amazon.awssdk.services.dynamodb.DynamoDbClient
import software.amazon.awssdk.services.dynamodb.model.*
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.*
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointResponse
import software.amazon.awssdk.services.sns.SnsClient
import software.amazon.awssdk.services.sns.model.PublishRequest
import software.amazon.awssdk.services.sns.model.PublishResponse
import java.io.ByteArrayInputStream

/**
 * Integration tests for the complete Inference Pipeline workflow.
 * 
 * This test suite validates:
 * - End-to-end inference workflow with mocked AWS services
 * - S3 orchestration between Score, Store, Alert, and Monitor stages
 * - DynamoDB storage and SNS alerting
 * 
 * **Requirements:**
 * - Requirement 16.2: Include integration tests that validate end-to-end workflow execution
 * - Requirement 16.3: Include tests that validate S3 orchestration patterns
 */
class InferencePipelineIntegrationTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    test("should execute complete inference pipeline workflow successfully") {
        // Setup mocked AWS clients
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockDynamoDbClient = mockk<DynamoDbClient>()
        val mockSnsClient = mockk<SnsClient>()
        
        val executionId = "exec-inference-integration-001"
        val workflowBucket = "test-workflow-bucket"
        val batchDate = "2024-01-15"
        
        // ========================================
        // Stage 1: Score Handler
        // ========================================
        
        val scoreHandler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3FieldScore = scoreHandler.javaClass.superclass.getDeclaredField("s3Client")
        s3FieldScore.isAccessible = true
        s3FieldScore.set(scoreHandler, mockS3Client)
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", "fraud-detection-prod")
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "test-config-bucket" && 
                    it.key() == "current-endpoint.json" 
                }
            ) 
        } returns endpointMetadataStream
        
        // Create test transactions
        val transactions = listOf(
            Transaction(
                id = "txn-001",
                timestamp = 1705334400000L,
                amount = 149.62,
                merchantCategory = "retail",
                features = mapOf("Time" to 0.0, "V1" to -1.36, "Amount" to 149.62)
            ),
            Transaction(
                id = "txn-002",
                timestamp = 1705334500000L,
                amount = 2500.00,
                merchantCategory = "online",
                features = mapOf("Time" to 100.0, "V1" to 2.45, "Amount" to 2500.00)
            ),
            Transaction(
                id = "txn-003",
                timestamp = 1705334600000L,
                amount = 5000.00,
                merchantCategory = "online",
                features = mapOf("Time" to 200.0, "V1" to 3.5, "Amount" to 5000.00)
            )
        )
        
        // Mock transaction batch read
        val transactionBatchJson = objectMapper.writeValueAsString(transactions)
        val transactionBatchStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(transactionBatchJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && 
                    it.key() == "daily-batches/$batchDate.json" 
                }
            ) 
        } returns transactionBatchStream
        
        // Mock SageMaker endpoint invocations with different scores
        val fraudScores = listOf(0.1, 0.6, 0.9) // low, medium, high risk
        var invocationCount = 0
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } answers {
            val score = fraudScores[invocationCount % fraudScores.size]
            invocationCount++
            InvokeEndpointResponse.builder()
                .body(SdkBytes.fromUtf8String(score.toString()))
                .build()
        }
        
        // Mock S3 write for Score stage output
        var scoreStageOutput: String? = null
        every { 
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                },
                any<RequestBody>()
            ) 
        } answers {
            val requestBody = secondArg<RequestBody>()
            scoreStageOutput = requestBody.contentStreamProvider().newStream().readAllBytes().toString(Charsets.UTF_8)
            PutObjectResponse.builder().build()
        }
        
        // Execute Score stage
        val scoreInput = mapOf(
            "executionId" to executionId,
            "currentStage" to "ScoreStage",
            "workflowBucket" to workflowBucket,
            "initialData" to mapOf(
                "transactionBatchPath" to "s3://fraud-detection-data/daily-batches/$batchDate.json",
                "batchDate" to batchDate
            )
        )
        
        val mockContext = mockk<Context>()
        val scoreResult = scoreHandler.handleRequest(scoreInput, mockContext)
        
        // Verify Score stage succeeded
        scoreResult.status shouldBe "SUCCESS"
        scoreResult.stage shouldBe "ScoreStage"
        scoreStageOutput shouldNotBe null
        
        val scoreOutputNode = objectMapper.readTree(scoreStageOutput)
        scoreOutputNode.get("batchDate").asText() shouldBe batchDate
        scoreOutputNode.get("transactionCount").asInt() shouldBe transactions.size
        scoreOutputNode.get("endpointName").asText() shouldBe "fraud-detection-prod"
        
        // ========================================
        // Stage 2: Store Handler
        // ========================================
        
        val storeHandler = StoreHandler(
            dynamoDbClient = mockDynamoDbClient,
            tableName = "FraudScores"
        )
        
        // Replace S3 client with mock
        val s3FieldStore = storeHandler.javaClass.superclass.getDeclaredField("s3Client")
        s3FieldStore.isAccessible = true
        s3FieldStore.set(storeHandler, mockS3Client)
        
        // Mock S3 read for Score stage output (input to Store stage)
        val scoreStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(scoreStageOutput!!.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                }
            ) 
        } returns scoreStream
        
        // Mock DynamoDB batch write
        every { 
            mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) 
        } returns BatchWriteItemResponse.builder()
            .unprocessedItems(emptyMap())
            .build()
        
        // Mock S3 write for Store stage output
        var storeStageOutput: String? = null
        every { 
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/StoreStage/output.json" 
                },
                any<RequestBody>()
            ) 
        } answers {
            val requestBody = secondArg<RequestBody>()
            storeStageOutput = requestBody.contentStreamProvider().newStream().readAllBytes().toString(Charsets.UTF_8)
            PutObjectResponse.builder().build()
        }
        
        // Execute Store stage
        val storeInput = mapOf(
            "executionId" to executionId,
            "currentStage" to "StoreStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to workflowBucket
        )
        
        val storeResult = storeHandler.handleRequest(storeInput, mockContext)
        
        // Verify Store stage succeeded
        storeResult.status shouldBe "SUCCESS"
        storeResult.stage shouldBe "StoreStage"
        storeStageOutput shouldNotBe null
        
        val storeOutputNode = objectMapper.readTree(storeStageOutput)
        storeOutputNode.get("batchDate").asText() shouldBe batchDate
        storeOutputNode.get("totalTransactions").asInt() shouldBe transactions.size
        storeOutputNode.get("successCount").asInt() shouldBe transactions.size
        storeOutputNode.get("errorCount").asInt() shouldBe 0
        
        // ========================================
        // Stage 3: Alert Handler
        // ========================================
        
        val alertHandler = AlertHandler(
            snsClient = mockSnsClient,
            fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
        )
        
        // Replace S3 client with mock
        val s3FieldAlert = alertHandler.javaClass.superclass.getDeclaredField("s3Client")
        s3FieldAlert.isAccessible = true
        s3FieldAlert.set(alertHandler, mockS3Client)
        
        // Mock S3 read for Score stage output (Alert reads from Score, not Store)
        val scoreStream2 = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(scoreStageOutput!!.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                }
            ) 
        } returns scoreStream2
        
        // Mock SNS publish
        var alertsSent = 0
        every { 
            mockSnsClient.publish(any<PublishRequest>()) 
        } answers {
            alertsSent++
            PublishResponse.builder().messageId("msg-$alertsSent").build()
        }
        
        // Mock S3 write for Alert stage output
        var alertStageOutput: String? = null
        every { 
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/AlertStage/output.json" 
                },
                any<RequestBody>()
            ) 
        } answers {
            val requestBody = secondArg<RequestBody>()
            alertStageOutput = requestBody.contentStreamProvider().newStream().readAllBytes().toString(Charsets.UTF_8)
            PutObjectResponse.builder().build()
        }
        
        // Execute Alert stage
        val alertInput = mapOf(
            "executionId" to executionId,
            "currentStage" to "AlertStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to workflowBucket
        )
        
        val alertResult = alertHandler.handleRequest(alertInput, mockContext)
        
        // Verify Alert stage succeeded
        alertResult.status shouldBe "SUCCESS"
        alertResult.stage shouldBe "AlertStage"
        alertStageOutput shouldNotBe null
        
        val alertOutputNode = objectMapper.readTree(alertStageOutput)
        alertOutputNode.get("batchDate").asText() shouldBe batchDate
        alertOutputNode.get("highRiskCount").asInt() shouldBe 1 // Only txn-003 with score 0.9
        
        // Verify SNS was called for high-risk transactions
        verify(atLeast = 1) {
            mockSnsClient.publish(any<PublishRequest>())
        }
        
        // ========================================
        // Stage 4: Monitor Handler
        // ========================================
        
        val monitorHandler = MonitorHandler(
            snsClient = mockSnsClient,
            monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-monitoring",
            metricsBucket = "test-metrics-bucket"
        )
        
        // Replace S3 client with mock
        val s3FieldMonitor = monitorHandler.javaClass.superclass.getDeclaredField("s3Client")
        s3FieldMonitor.isAccessible = true
        s3FieldMonitor.set(monitorHandler, mockS3Client)
        
        // Mock S3 read for Store stage output (input to Monitor stage)
        val storeStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(storeStageOutput!!.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/StoreStage/output.json" 
                }
            ) 
        } returns storeStream
        
        // Mock S3 read for Score stage output (Monitor needs scored transactions)
        val scoreStream3 = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(scoreStageOutput!!.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                }
            ) 
        } returns scoreStream3
        
        // Mock historical baseline read (no baseline exists for first run)
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "test-metrics-bucket" && 
                    it.key() == "baseline-metrics.json" 
                }
            ) 
        } throws NoSuchKeyException.builder().message("No baseline exists").build()
        
        // Mock S3 write for metrics
        var metricsWritten: String? = null
        every { 
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == "test-metrics-bucket" && 
                    it.key().startsWith("metrics/")
                },
                any<RequestBody>()
            ) 
        } answers {
            val requestBody = secondArg<RequestBody>()
            metricsWritten = requestBody.contentStreamProvider().newStream().readAllBytes().toString(Charsets.UTF_8)
            PutObjectResponse.builder().build()
        }
        
        // Mock S3 write for Monitor stage output
        var monitorStageOutput: String? = null
        every { 
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/MonitorStage/output.json" 
                },
                any<RequestBody>()
            ) 
        } answers {
            val requestBody = secondArg<RequestBody>()
            monitorStageOutput = requestBody.contentStreamProvider().newStream().readAllBytes().toString(Charsets.UTF_8)
            PutObjectResponse.builder().build()
        }
        
        // Execute Monitor stage
        val monitorInput = mapOf(
            "executionId" to executionId,
            "currentStage" to "MonitorStage",
            "previousStage" to "StoreStage",
            "workflowBucket" to workflowBucket
        )
        
        val monitorResult = monitorHandler.handleRequest(monitorInput, mockContext)
        
        // Verify Monitor stage succeeded
        monitorResult.status shouldBe "SUCCESS"
        monitorResult.stage shouldBe "MonitorStage"
        monitorStageOutput shouldNotBe null
        metricsWritten shouldNotBe null
        
        val monitorOutputNode = objectMapper.readTree(monitorStageOutput)
        monitorOutputNode.get("batchDate").asText() shouldBe batchDate
        
        // ========================================
        // Verify S3 Orchestration
        // ========================================
        
        // Verify Score stage wrote output
        verify(exactly = 1) {
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                },
                any<RequestBody>()
            )
        }
        
        // Verify Store stage read Score output
        verify(atLeast = 1) {
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                }
            )
        }
        
        // Verify Store stage wrote output
        verify(exactly = 1) {
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/StoreStage/output.json" 
                },
                any<RequestBody>()
            )
        }
        
        // Verify Alert stage read Score output (not Store output)
        verify(atLeast = 1) {
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                }
            )
        }
        
        // Verify Alert stage wrote output
        verify(exactly = 1) {
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/AlertStage/output.json" 
                },
                any<RequestBody>()
            )
        }
        
        // Verify Monitor stage read Store output
        verify(atLeast = 1) {
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/StoreStage/output.json" 
                }
            )
        }
        
        // Verify Monitor stage wrote output
        verify(exactly = 1) {
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/MonitorStage/output.json" 
                },
                any<RequestBody>()
            )
        }
        
        // Verify metrics were written
        verify(exactly = 1) {
            mockS3Client.putObject(
                match<PutObjectRequest> { 
                    it.bucket() == "test-metrics-bucket" && 
                    it.key().startsWith("metrics/")
                },
                any<RequestBody>()
            )
        }
        
        // ========================================
        // Verify DynamoDB Storage
        // ========================================
        
        // Verify DynamoDB batch write was called
        verify(atLeast = 1) {
            mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>())
        }
        
        // ========================================
        // Verify SNS Alerting
        // ========================================
        
        // Verify SNS publish was called for high-risk alerts
        verify(atLeast = 1) {
            mockSnsClient.publish(
                match<PublishRequest> { 
                    it.topicArn() == "arn:aws:sns:us-east-1:123456789012:fraud-alerts" 
                }
            )
        }
    }
    
    test("should handle DynamoDB write failures gracefully") {
        // Setup mocked AWS clients
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockDynamoDbClient = mockk<DynamoDbClient>()
        
        val executionId = "exec-dynamodb-failure-001"
        val workflowBucket = "test-workflow-bucket"
        val batchDate = "2024-01-15"
        
        val storeHandler = StoreHandler(
            dynamoDbClient = mockDynamoDbClient,
            tableName = "FraudScores"
        )
        
        // Replace S3 client with mock
        val s3Field = storeHandler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(storeHandler, mockS3Client)
        
        // Create scored transactions
        val scoredTransactions = listOf(
            ScoredTransaction(
                transactionId = "txn-001",
                timestamp = 1705334400000L,
                amount = 149.62,
                merchantCategory = "retail",
                features = mapOf("Time" to 0.0, "V1" to -1.36, "Amount" to 149.62),
                fraudScore = 0.1,
                scoringTimestamp = System.currentTimeMillis()
            )
        )
        
        val scoreOutput = objectMapper.createObjectNode().apply {
            replace("scoredTransactions", objectMapper.valueToTree(scoredTransactions))
            put("batchDate", batchDate)
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        // Mock S3 read for Score stage output
        val scoreStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreOutput).toByteArray())
            )
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                }
            ) 
        } returns scoreStream
        
        // Mock DynamoDB batch write failure
        every { 
            mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) 
        } throws DynamoDbException.builder().message("Provisioned throughput exceeded").build()
        
        // Execute Store stage
        val storeInput = mapOf(
            "executionId" to executionId,
            "currentStage" to "StoreStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to workflowBucket
        )
        
        val mockContext = mockk<Context>()
        val storeResult = storeHandler.handleRequest(storeInput, mockContext)
        
        // Verify Store stage failed with DynamoDB error
        storeResult.status shouldBe "FAILED"
        storeResult.errorMessage shouldContain "DynamoDB"
    }
    
    test("should handle SNS publish failures gracefully") {
        // Setup mocked AWS clients
        val mockS3Client = mockk<S3Client>()
        val mockSnsClient = mockk<SnsClient>()
        
        val executionId = "exec-sns-failure-001"
        val workflowBucket = "test-workflow-bucket"
        val batchDate = "2024-01-15"
        
        val alertHandler = AlertHandler(
            snsClient = mockSnsClient,
            fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
        )
        
        // Replace S3 client with mock
        val s3Field = alertHandler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(alertHandler, mockS3Client)
        
        // Create high-risk scored transactions
        val scoredTransactions = listOf(
            ScoredTransaction(
                transactionId = "txn-001",
                timestamp = 1705334400000L,
                amount = 5000.00,
                merchantCategory = "online",
                features = mapOf("Time" to 0.0, "V1" to 3.5, "Amount" to 5000.00),
                fraudScore = 0.9,
                scoringTimestamp = System.currentTimeMillis()
            )
        )
        
        val scoreOutput = objectMapper.createObjectNode().apply {
            replace("scoredTransactions", objectMapper.valueToTree(scoredTransactions))
            put("batchDate", batchDate)
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        // Mock S3 read for Score stage output
        val scoreStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreOutput).toByteArray())
            )
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                }
            ) 
        } returns scoreStream
        
        // Mock SNS publish failure
        every { 
            mockSnsClient.publish(any<PublishRequest>()) 
        } throws software.amazon.awssdk.services.sns.model.SnsException.builder()
            .message("Topic does not exist")
            .build()
        
        // Execute Alert stage
        val alertInput = mapOf(
            "executionId" to executionId,
            "currentStage" to "AlertStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to workflowBucket
        )
        
        val mockContext = mockk<Context>()
        val alertResult = alertHandler.handleRequest(alertInput, mockContext)
        
        // Verify Alert stage failed with SNS error
        alertResult.status shouldBe "FAILED"
        alertResult.errorMessage shouldContain "SNS"
    }
    
    test("should detect distribution drift and send monitoring alert") {
        // Setup mocked AWS clients
        val mockS3Client = mockk<S3Client>()
        val mockSnsClient = mockk<SnsClient>()
        
        val executionId = "exec-drift-detection-001"
        val workflowBucket = "test-workflow-bucket"
        val batchDate = "2024-01-15"
        
        val monitorHandler = MonitorHandler(
            snsClient = mockSnsClient,
            monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-monitoring",
            metricsBucket = "test-metrics-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = monitorHandler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(monitorHandler, mockS3Client)
        
        // Create store output with high-risk distribution
        val storeOutput = objectMapper.createObjectNode().apply {
            put("batchDate", batchDate)
            put("totalTransactions", 100)
            put("successCount", 100)
            put("errorCount", 0)
            putObject("riskDistribution").apply {
                put("highRisk", 50) // 50% high risk (significant drift)
                put("mediumRisk", 30)
                put("lowRisk", 20)
            }
            put("avgFraudScore", 0.7)
        }
        
        // Mock S3 read for Store stage output
        val storeStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(storeOutput).toByteArray())
            )
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/StoreStage/output.json" 
                }
            ) 
        } returns storeStream
        
        // Create scored transactions for drift calculation
        val scoredTransactions = (1..100).map { i ->
            ScoredTransaction(
                transactionId = "txn-$i",
                timestamp = System.currentTimeMillis(),
                amount = 100.0,
                merchantCategory = "retail",
                features = emptyMap(),
                fraudScore = if (i <= 50) 0.9 else 0.3,
                scoringTimestamp = System.currentTimeMillis()
            )
        }
        
        val scoreOutput = objectMapper.createObjectNode().apply {
            replace("scoredTransactions", objectMapper.valueToTree(scoredTransactions))
            put("batchDate", batchDate)
            put("transactionCount", scoredTransactions.size)
        }
        
        val scoreStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreOutput).toByteArray())
            )
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == workflowBucket && 
                    it.key() == "executions/$executionId/ScoreStage/output.json" 
                }
            ) 
        } returns scoreStream
        
        // Mock historical baseline with normal distribution
        val baseline = objectMapper.createObjectNode().apply {
            put("avgFraudScore", 0.3) // Baseline avg is 0.3, current is 0.7 (drift > 0.1)
            putObject("riskDistribution").apply {
                put("highRisk", 5) // Baseline 5%, current 50% (drift > 0.05)
                put("mediumRisk", 20)
                put("lowRisk", 75)
            }
        }
        
        val baselineStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(baseline).toByteArray())
            )
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "test-metrics-bucket" && 
                    it.key() == "baseline-metrics.json" 
                }
            ) 
        } returns baselineStream
        
        // Mock SNS publish for drift alert
        var driftAlertSent = false
        every { 
            mockSnsClient.publish(any<PublishRequest>()) 
        } answers {
            driftAlertSent = true
            PublishResponse.builder().messageId("msg-drift-001").build()
        }
        
        // Mock S3 writes
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        // Execute Monitor stage
        val monitorInput = mapOf(
            "executionId" to executionId,
            "currentStage" to "MonitorStage",
            "previousStage" to "StoreStage",
            "workflowBucket" to workflowBucket
        )
        
        val mockContext = mockk<Context>()
        val monitorResult = monitorHandler.handleRequest(monitorInput, mockContext)
        
        // Verify Monitor stage succeeded
        monitorResult.status shouldBe "SUCCESS"
        
        // Verify drift alert was sent
        driftAlertSent shouldBe true
        verify(atLeast = 1) {
            mockSnsClient.publish(
                match<PublishRequest> { 
                    it.topicArn() == "arn:aws:sns:us-east-1:123456789012:fraud-monitoring" 
                }
            )
        }
    }
})
