package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ScoredTransaction
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.ints.shouldBeGreaterThanOrEqual
import io.kotest.matchers.ints.shouldBeLessThanOrEqual
import io.kotest.matchers.shouldBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
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
 * Property-based tests for alert batching in AlertHandler.
 * 
 * This test suite validates:
 * - Property 15: Alert Batching
 * 
 * **Requirements:**
 * - Requirement 8.5: Batch alerts to avoid SNS rate limits (maximum 100 alerts per message)
 */
class AlertHandlerAlertBatchingPropertyTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    // ========================================
    // Property 15: Alert Batching
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 15: Alert Batching
    test("Property 15: Alert Batching - if high-risk count exceeds 100, transactions SHALL be batched into groups of at most 100 per SNS message") {
        checkAll(100, Arb.int(101..500)) { highRiskCount ->
            val mockS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = AlertHandler(
                snsClient = mockSnsClient,
                fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            // Create transactions with fraud scores >= 0.8 (all high-risk)
            val scoredTransactions = List(highRiskCount) { index ->
                ScoredTransaction(
                    transactionId = "txn-high-risk-$index",
                    timestamp = System.currentTimeMillis(),
                    amount = 1000.0,
                    merchantCategory = "online",
                    features = mapOf("V1" to 1.0),
                    fraudScore = 0.9, // High-risk
                    scoringTimestamp = System.currentTimeMillis()
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
            
            val publishedBatches = mutableListOf<Int>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                val batchSize = request.messageAttributes()["highRiskCount"]?.stringValue()?.toInt() ?: 0
                publishedBatches.add(batchSize)
                PublishResponse.builder().messageId("msg-${publishedBatches.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "AlertStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Then the number of batches SHALL be ceil(highRiskCount / 100)
            val expectedBatches = (highRiskCount + 99) / 100
            publishedBatches.size shouldBe expectedBatches
            
            // And each batch SHALL contain at most 100 transactions
            publishedBatches.forEach { batchSize ->
                batchSize shouldBeLessThanOrEqual 100
                batchSize shouldBeGreaterThanOrEqual 1
            }
            
            // And the total count SHALL equal the high-risk count
            publishedBatches.sum() shouldBe highRiskCount
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 15: Alert Batching
    test("Property 15: Alert Batching - if high-risk count is exactly 100, there SHALL be exactly 1 batch") {
        checkAll(50, Arb.constant(100)) { highRiskCount ->
            val mockS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = AlertHandler(
                snsClient = mockSnsClient,
                fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            val scoredTransactions = List(highRiskCount) { index ->
                ScoredTransaction(
                    transactionId = "txn-boundary-$index",
                    timestamp = System.currentTimeMillis(),
                    amount = 1000.0,
                    merchantCategory = "online",
                    features = mapOf("V1" to 1.0),
                    fraudScore = 0.8, // Exactly at threshold
                    scoringTimestamp = System.currentTimeMillis()
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
            
            val publishedBatches = mutableListOf<Int>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                val batchSize = request.messageAttributes()["highRiskCount"]?.stringValue()?.toInt() ?: 0
                publishedBatches.add(batchSize)
                PublishResponse.builder().messageId("msg-${publishedBatches.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "AlertStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Then there SHALL be exactly 1 batch
            publishedBatches.size shouldBe 1
            
            // And the batch SHALL contain exactly 100 transactions
            publishedBatches[0] shouldBe 100
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 15: Alert Batching
    test("Property 15: Alert Batching - if high-risk count is less than 100, there SHALL be exactly 1 batch") {
        checkAll(100, Arb.int(1..99)) { highRiskCount ->
            val mockS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = AlertHandler(
                snsClient = mockSnsClient,
                fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            val scoredTransactions = List(highRiskCount) { index ->
                ScoredTransaction(
                    transactionId = "txn-small-batch-$index",
                    timestamp = System.currentTimeMillis(),
                    amount = 1000.0,
                    merchantCategory = "online",
                    features = mapOf("V1" to 1.0),
                    fraudScore = 0.85,
                    scoringTimestamp = System.currentTimeMillis()
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
            
            val publishedBatches = mutableListOf<Int>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                val batchSize = request.messageAttributes()["highRiskCount"]?.stringValue()?.toInt() ?: 0
                publishedBatches.add(batchSize)
                PublishResponse.builder().messageId("msg-${publishedBatches.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "AlertStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Then there SHALL be exactly 1 batch
            publishedBatches.size shouldBe 1
            
            // And the batch SHALL contain exactly highRiskCount transactions
            publishedBatches[0] shouldBe highRiskCount
        }
    }
})
