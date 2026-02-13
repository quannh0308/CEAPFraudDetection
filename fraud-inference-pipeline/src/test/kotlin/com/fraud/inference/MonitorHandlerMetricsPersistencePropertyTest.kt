package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
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
 * Property-based tests for metrics persistence in MonitorHandler.
 * 
 * This test suite validates:
 * - Property 17: Metrics Persistence
 * 
 * **Requirements:**
 * - Requirement 14.3: Write performance metrics to S3 for analysis
 */
class MonitorHandlerMetricsPersistencePropertyTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    // ========================================
    // Property 17: Metrics Persistence
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 17: Metrics Persistence
    test("Property 17: Metrics Persistence - metrics SHALL be written to S3 at path metrics/{batchDate}.json") {
        checkAll(100, Arb.performanceMetricsForPersistence(), Arb.batchDateForPersistence()) { currentMetrics, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockMetricsS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = MonitorHandler(
                metricsS3Client = mockMetricsS3Client,
                snsClient = mockSnsClient,
                metricsBucket = "test-metrics-bucket",
                monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:monitoring-alerts"
            )
            
            // Replace S3 client with mock
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            // Mock S3 input read (from StoreStage)
            val storeStageOutput = objectMapper.createObjectNode().apply {
                put("batchDate", batchDate)
                put("totalTransactions", 100)
                put("successCount", 100)
                put("errorCount", 0)
                set<com.fasterxml.jackson.databind.node.ObjectNode>(
                    "riskDistribution",
                    objectMapper.createObjectNode().apply {
                        put("highRisk", (currentMetrics.highRiskPct * 100).toInt())
                        put("mediumRisk", 20)
                        put("lowRisk", 80 - (currentMetrics.highRiskPct * 100).toInt())
                    }
                )
                put("avgFraudScore", currentMetrics.avgFraudScore)
            }
            
            val storeStageStream = ResponseInputStream(
                GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
                )
            )
            
            every { 
                mockS3Client.getObject(any<GetObjectRequest>()) 
            } returns storeStageStream
            
            // Mock historical metrics loading (return baseline)
            val baseline = PerformanceMetricsData(
                avgFraudScore = currentMetrics.avgFraudScore + 0.02, // Small drift
                highRiskPct = currentMetrics.highRiskPct + 0.01 // Small drift
            )
            
            every {
                mockMetricsS3Client.getObject(any<GetObjectRequest>())
            } answers {
                val historicalMetrics = objectMapper.createObjectNode().apply {
                    put("avgFraudScore", baseline.avgFraudScore)
                    put("highRiskPct", baseline.highRiskPct)
                }
                ResponseInputStream(
                    GetObjectResponse.builder().build(),
                    AbortableInputStream.create(
                        ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                    )
                )
            }
            
            // Mock S3 output write (for both workflow output and metrics)
            every { 
                mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
            } returns PutObjectResponse.builder().build()
            
            val capturedMetricsWrites = mutableListOf<Pair<PutObjectRequest, RequestBody>>()
            
            every {
                mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>())
            } answers {
                val request = firstArg<PutObjectRequest>()
                val body = secondArg<RequestBody>()
                capturedMetricsWrites.add(request to body)
                PutObjectResponse.builder().build()
            }
            
            // Mock SNS publish (may or may not be called depending on drift)
            every { 
                mockSnsClient.publish(any<PublishRequest>()) 
            } returns PublishResponse.builder().messageId("msg-1").build()
            
            // Create input for handler
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "MonitorStage",
                "previousStage" to "StoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            
            // When the monitoring stage processes the metrics
            handler.handleRequest(input, mockContext)
            
            // Then metrics SHALL be written to S3
            capturedMetricsWrites.size shouldBe 1
            
            val (metricsRequest, metricsBody) = capturedMetricsWrites[0]
            
            // And the path SHALL be metrics/{batchDate}.json
            metricsRequest.bucket() shouldBe "test-metrics-bucket"
            metricsRequest.key() shouldBe "metrics/$batchDate.json"
            
            // And the metrics SHALL contain required fields
            val metricsJson = metricsBody.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val metricsNode = objectMapper.readTree(metricsJson)
            
            metricsNode.get("batchDate").asText() shouldBe batchDate
            metricsNode.get("avgFraudScore") shouldNotBe null
            metricsNode.get("highRiskPct") shouldNotBe null
            metricsNode.get("mediumRiskPct") shouldNotBe null
            metricsNode.get("lowRiskPct") shouldNotBe null
            metricsNode.get("avgScoreDrift") shouldNotBe null
            metricsNode.get("highRiskDrift") shouldNotBe null
            metricsNode.get("driftDetected") shouldNotBe null
            metricsNode.get("timestamp") shouldNotBe null
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 17: Metrics Persistence
    test("Property 17: Metrics Persistence - metrics SHALL contain all required fields") {
        checkAll(100, Arb.performanceMetricsForPersistence(), Arb.batchDateForPersistence()) { currentMetrics, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockMetricsS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = MonitorHandler(
                metricsS3Client = mockMetricsS3Client,
                snsClient = mockSnsClient,
                metricsBucket = "test-metrics-bucket",
                monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:monitoring-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            val storeStageOutput = objectMapper.createObjectNode().apply {
                put("batchDate", batchDate)
                put("totalTransactions", 100)
                put("successCount", 100)
                put("errorCount", 0)
                set<com.fasterxml.jackson.databind.node.ObjectNode>(
                    "riskDistribution",
                    objectMapper.createObjectNode().apply {
                        put("highRisk", (currentMetrics.highRiskPct * 100).toInt())
                        put("mediumRisk", 20)
                        put("lowRisk", 80 - (currentMetrics.highRiskPct * 100).toInt())
                    }
                )
                put("avgFraudScore", currentMetrics.avgFraudScore)
            }
            
            val storeStageStream = ResponseInputStream(
                GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
                )
            )
            
            every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
            
            val baseline = PerformanceMetricsData(
                avgFraudScore = currentMetrics.avgFraudScore + 0.02,
                highRiskPct = currentMetrics.highRiskPct + 0.01
            )
            
            every {
                mockMetricsS3Client.getObject(any<GetObjectRequest>())
            } answers {
                val historicalMetrics = objectMapper.createObjectNode().apply {
                    put("avgFraudScore", baseline.avgFraudScore)
                    put("highRiskPct", baseline.highRiskPct)
                }
                ResponseInputStream(
                    GetObjectResponse.builder().build(),
                    AbortableInputStream.create(
                        ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                    )
                )
            }
            
            every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            
            val capturedMetricsWrites = mutableListOf<Pair<PutObjectRequest, RequestBody>>()
            
            every {
                mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>())
            } answers {
                val request = firstArg<PutObjectRequest>()
                val body = secondArg<RequestBody>()
                capturedMetricsWrites.add(request to body)
                PutObjectResponse.builder().build()
            }
            
            every { mockSnsClient.publish(any<PublishRequest>()) } returns PublishResponse.builder().messageId("msg-1").build()
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "MonitorStage",
                "previousStage" to "StoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Then metrics SHALL contain all required fields
            val (_, metricsBody) = capturedMetricsWrites[0]
            val metricsJson = metricsBody.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val metricsNode = objectMapper.readTree(metricsJson)
            
            // Verify all required fields are present
            val requiredFields = listOf(
                "batchDate", "avgFraudScore", "highRiskPct", "mediumRiskPct", "lowRiskPct",
                "avgScoreDrift", "highRiskDrift", "driftDetected", "timestamp"
            )
            
            requiredFields.forEach { field ->
                metricsNode.has(field) shouldBe true
            }
            
            // Verify field types
            metricsNode.get("batchDate").isTextual shouldBe true
            metricsNode.get("avgFraudScore").isNumber shouldBe true
            metricsNode.get("highRiskPct").isNumber shouldBe true
            metricsNode.get("mediumRiskPct").isNumber shouldBe true
            metricsNode.get("lowRiskPct").isNumber shouldBe true
            metricsNode.get("avgScoreDrift").isNumber shouldBe true
            metricsNode.get("highRiskDrift").isNumber shouldBe true
            metricsNode.get("driftDetected").isBoolean shouldBe true
            metricsNode.get("timestamp").isNumber shouldBe true
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 17: Metrics Persistence
    test("Property 17: Metrics Persistence - metrics SHALL be persisted for every execution") {
        checkAll(50, Arb.performanceMetricsForPersistence(), Arb.batchDateForPersistence()) { currentMetrics, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockMetricsS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = MonitorHandler(
                metricsS3Client = mockMetricsS3Client,
                snsClient = mockSnsClient,
                metricsBucket = "test-metrics-bucket",
                monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:monitoring-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            val storeStageOutput = objectMapper.createObjectNode().apply {
                put("batchDate", batchDate)
                put("totalTransactions", 100)
                put("successCount", 100)
                put("errorCount", 0)
                set<com.fasterxml.jackson.databind.node.ObjectNode>(
                    "riskDistribution",
                    objectMapper.createObjectNode().apply {
                        put("highRisk", (currentMetrics.highRiskPct * 100).toInt())
                        put("mediumRisk", 20)
                        put("lowRisk", 80 - (currentMetrics.highRiskPct * 100).toInt())
                    }
                )
                put("avgFraudScore", currentMetrics.avgFraudScore)
            }
            
            val storeStageStream = ResponseInputStream(
                GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
                )
            )
            
            every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
            
            val baseline = PerformanceMetricsData(
                avgFraudScore = currentMetrics.avgFraudScore + 0.02,
                highRiskPct = currentMetrics.highRiskPct + 0.01
            )
            
            every {
                mockMetricsS3Client.getObject(any<GetObjectRequest>())
            } answers {
                val historicalMetrics = objectMapper.createObjectNode().apply {
                    put("avgFraudScore", baseline.avgFraudScore)
                    put("highRiskPct", baseline.highRiskPct)
                }
                ResponseInputStream(
                    GetObjectResponse.builder().build(),
                    AbortableInputStream.create(
                        ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                    )
                )
            }
            
            every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            
            var metricsWriteCount = 0
            
            every {
                mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>())
            } answers {
                metricsWriteCount++
                PutObjectResponse.builder().build()
            }
            
            every { mockSnsClient.publish(any<PublishRequest>()) } returns PublishResponse.builder().messageId("msg-1").build()
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "MonitorStage",
                "previousStage" to "StoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Then metrics SHALL be written exactly once per execution
            metricsWriteCount shouldBe 1
        }
    }
})

// ========================================
// Arbitrary Generators for Property Tests
// ========================================

/**
 * Generates arbitrary performance metrics.
 */
private fun Arb.Companion.performanceMetricsForPersistence(): Arb<PerformanceMetricsData> {
    return arbitrary { rs ->
        PerformanceMetricsData(
            avgFraudScore = Arb.double(0.0, 1.0).bind(),
            highRiskPct = Arb.double(0.0, 0.5).bind() // 0-50% high risk
        )
    }
}

/**
 * Generates arbitrary batch dates in YYYY-MM-DD format.
 */
private fun Arb.Companion.batchDateForPersistence(): Arb<String> {
    return arbitrary { rs ->
        val year = Arb.int(2024..2025).bind()
        val month = Arb.int(1..12).bind()
        val day = Arb.int(1..28).bind() // Use 28 to avoid month-specific logic
        String.format("%04d-%02d-%02d", year, month, day)
    }
}

/**
 * Performance metrics data class for testing.
 */
private data class PerformanceMetricsData(
    val avgFraudScore: Double,
    val highRiskPct: Double
)
