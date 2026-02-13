package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ScoredTransaction
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.mockk.*
import software.amazon.awssdk.core.ResponseInputStream
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.http.AbortableInputStream
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.GetObjectResponse
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.model.PutObjectResponse
import software.amazon.awssdk.services.sns.SnsClient
import software.amazon.awssdk.services.sns.model.PublishRequest
import software.amazon.awssdk.services.sns.model.PublishResponse
import java.io.ByteArrayInputStream

/**
 * Unit tests for AlertHandler.
 * 
 * These tests validate specific examples and edge cases for:
 * - High-risk filtering (Requirement 8.2)
 * - Alert message formatting (Requirement 8.3, 8.4)
 * - SNS publish with mocked client (Requirement 8.3)
 */
class AlertHandlerTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    test("should filter high-risk transactions with fraud score >= 0.8") {
        val mockS3Client = mockk<S3Client>()
        val mockSnsClient = mockk<SnsClient>()
        
        val handler = AlertHandler(
            snsClient = mockSnsClient,
            fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        // Create mix of high-risk and low-risk transactions
        val scoredTransactions = listOf(
            ScoredTransaction(
                transactionId = "txn-001",
                timestamp = 1705334400000,
                amount = 149.62,
                merchantCategory = "retail",
                features = mapOf("V1" to -1.36),
                fraudScore = 0.0234, // Low-risk
                scoringTimestamp = 1705334401000
            ),
            ScoredTransaction(
                transactionId = "txn-002",
                timestamp = 1705334500000,
                amount = 2500.00,
                merchantCategory = "online",
                features = mapOf("V1" to 2.45),
                fraudScore = 0.8912, // High-risk
                scoringTimestamp = 1705334501000
            ),
            ScoredTransaction(
                transactionId = "txn-003",
                timestamp = 1705334600000,
                amount = 500.00,
                merchantCategory = "restaurant",
                features = mapOf("V1" to 0.5),
                fraudScore = 0.7999, // Low-risk (just below threshold)
                scoringTimestamp = 1705334601000
            ),
            ScoredTransaction(
                transactionId = "txn-004",
                timestamp = 1705334700000,
                amount = 3000.00,
                merchantCategory = "travel",
                features = mapOf("V1" to 3.0),
                fraudScore = 0.8, // High-risk (exactly at threshold)
                scoringTimestamp = 1705334701000
            )
        )
        
        val scoreStageOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("batchDate", "2024-01-15")
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        val publishedMessages = mutableListOf<String>()
        
        every { mockSnsClient.publish(any<PublishRequest>()) } answers {
            val request = firstArg<PublishRequest>()
            publishedMessages.add(request.message())
            PublishResponse.builder().messageId("msg-${publishedMessages.size}").build()
        }
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "AlertStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        handler.handleRequest(input, mockContext)
        
        // Verify only high-risk transactions are in the alert
        val allAlertText = publishedMessages.joinToString("\n")
        
        // High-risk transactions should be present
        allAlertText shouldContain "txn-002"
        allAlertText shouldContain "txn-004"
        
        // Low-risk transactions should NOT be present
        allAlertText.contains("txn-001") shouldBe false
        allAlertText.contains("txn-003") shouldBe false
        
        // Verify output
        val capturedOutput = slot<RequestBody>()
        verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
        
        val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
        val resultNode = objectMapper.readTree(outputJson)
        
        resultNode.get("highRiskCount").asInt() shouldBe 2
        resultNode.get("alertsSent").asInt() shouldBe 2
        resultNode.get("alertBatches").asInt() shouldBe 1
    }
    
    test("should format alert message with all required transaction details") {
        val mockS3Client = mockk<S3Client>()
        val mockSnsClient = mockk<SnsClient>()
        
        val handler = AlertHandler(
            snsClient = mockSnsClient,
            fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val scoredTransactions = listOf(
            ScoredTransaction(
                transactionId = "txn-high-001",
                timestamp = 1705334400000,
                amount = 2500.00,
                merchantCategory = "online",
                features = mapOf("V1" to 2.45),
                fraudScore = 0.8912,
                scoringTimestamp = 1705334401000
            )
        )
        
        val scoreStageOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("batchDate", "2024-01-15")
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        val publishedMessages = mutableListOf<String>()
        
        every { mockSnsClient.publish(any<PublishRequest>()) } answers {
            val request = firstArg<PublishRequest>()
            publishedMessages.add(request.message())
            PublishResponse.builder().messageId("msg-${publishedMessages.size}").build()
        }
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "AlertStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        handler.handleRequest(input, mockContext)
        
        // Verify alert message format
        publishedMessages.size shouldBe 1
        val alertMessage = publishedMessages[0]
        
        // Should contain header
        alertMessage shouldContain "High-Risk Fraud Transactions Detected"
        alertMessage shouldContain "Batch Date: 2024-01-15"
        alertMessage shouldContain "Count: 1"
        
        // Should contain key transaction identifiers
        alertMessage shouldContain "txn-high-001"
        alertMessage shouldContain "online"
    }
    
    test("should publish to SNS with correct message attributes") {
        val mockS3Client = mockk<S3Client>()
        val mockSnsClient = mockk<SnsClient>()
        
        val handler = AlertHandler(
            snsClient = mockSnsClient,
            fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val scoredTransactions = listOf(
            ScoredTransaction(
                transactionId = "txn-001",
                timestamp = 1705334400000,
                amount = 1000.00,
                merchantCategory = "online",
                features = mapOf("V1" to 1.0),
                fraudScore = 0.9,
                scoringTimestamp = 1705334401000
            ),
            ScoredTransaction(
                transactionId = "txn-002",
                timestamp = 1705334500000,
                amount = 2000.00,
                merchantCategory = "travel",
                features = mapOf("V1" to 2.0),
                fraudScore = 0.95,
                scoringTimestamp = 1705334501000
            )
        )
        
        val scoreStageOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("batchDate", "2024-01-15")
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        val capturedPublishRequest = slot<PublishRequest>()
        
        every { mockSnsClient.publish(capture(capturedPublishRequest)) } returns 
            PublishResponse.builder().messageId("msg-123").build()
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "AlertStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        handler.handleRequest(input, mockContext)
        
        // Verify SNS publish request
        val publishRequest = capturedPublishRequest.captured
        
        publishRequest.topicArn() shouldBe "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
        publishRequest.subject() shouldBe "Fraud Alert: 2 High-Risk Transactions Detected"
        
        // Verify message attributes
        val attributes = publishRequest.messageAttributes()
        attributes["batchDate"]?.stringValue() shouldBe "2024-01-15"
        attributes["highRiskCount"]?.stringValue() shouldBe "2"
        attributes["batchIndex"]?.stringValue() shouldBe "0"
    }
    
    test("should return zero counts when no high-risk transactions found") {
        val mockS3Client = mockk<S3Client>()
        val mockSnsClient = mockk<SnsClient>()
        
        val handler = AlertHandler(
            snsClient = mockSnsClient,
            fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        // All low-risk transactions
        val scoredTransactions = listOf(
            ScoredTransaction(
                transactionId = "txn-001",
                timestamp = 1705334400000,
                amount = 100.00,
                merchantCategory = "retail",
                features = mapOf("V1" to 0.1),
                fraudScore = 0.1,
                scoringTimestamp = 1705334401000
            ),
            ScoredTransaction(
                transactionId = "txn-002",
                timestamp = 1705334500000,
                amount = 200.00,
                merchantCategory = "grocery",
                features = mapOf("V1" to 0.2),
                fraudScore = 0.2,
                scoringTimestamp = 1705334501000
            )
        )
        
        val scoreStageOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("batchDate", "2024-01-15")
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "AlertStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        handler.handleRequest(input, mockContext)
        
        // Verify no SNS publish was called
        verify(exactly = 0) { mockSnsClient.publish(any<PublishRequest>()) }
        
        // Verify output
        val capturedOutput = slot<RequestBody>()
        verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
        
        val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
        val resultNode = objectMapper.readTree(outputJson)
        
        resultNode.get("highRiskCount").asInt() shouldBe 0
        resultNode.get("alertsSent").asInt() shouldBe 0
        resultNode.get("alertBatches").asInt() shouldBe 0
    }
    
    test("should continue processing remaining batches even if one SNS publish fails") {
        val mockS3Client = mockk<S3Client>()
        val mockSnsClient = mockk<SnsClient>()
        
        val handler = AlertHandler(
            snsClient = mockSnsClient,
            fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        // Create 150 high-risk transactions (will create 2 batches)
        val scoredTransactions = List(150) { index ->
            ScoredTransaction(
                transactionId = "txn-$index",
                timestamp = 1705334400000 + index,
                amount = 1000.0,
                merchantCategory = "online",
                features = mapOf("V1" to 1.0),
                fraudScore = 0.9,
                scoringTimestamp = 1705334401000 + index
            )
        }
        
        val scoreStageOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("batchDate", "2024-01-15")
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        var callCount = 0
        every { mockSnsClient.publish(any<PublishRequest>()) } answers {
            callCount++
            if (callCount == 1) {
                // First batch fails
                throw RuntimeException("SNS publish failed")
            } else {
                // Second batch succeeds
                PublishResponse.builder().messageId("msg-$callCount").build()
            }
        }
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "AlertStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        handler.handleRequest(input, mockContext)
        
        // Verify both batches were attempted
        verify(exactly = 2) { mockSnsClient.publish(any<PublishRequest>()) }
        
        // Verify output shows partial success
        val capturedOutput = slot<RequestBody>()
        verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
        
        val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
        val resultNode = objectMapper.readTree(outputJson)
        
        resultNode.get("highRiskCount").asInt() shouldBe 150
        resultNode.get("alertsSent").asInt() shouldBe 50 // Only second batch succeeded
        resultNode.get("alertBatches").asInt() shouldBe 2
    }
})
