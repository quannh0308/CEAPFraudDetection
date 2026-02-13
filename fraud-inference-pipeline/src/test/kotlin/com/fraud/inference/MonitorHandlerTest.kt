package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
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
import software.amazon.awssdk.services.s3.model.NoSuchKeyException
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.model.PutObjectResponse
import software.amazon.awssdk.services.sns.SnsClient
import software.amazon.awssdk.services.sns.model.PublishRequest
import software.amazon.awssdk.services.sns.model.PublishResponse
import java.io.ByteArrayInputStream

/**
 * Unit tests for MonitorHandler.
 * 
 * These tests validate specific examples and edge cases for:
 * - Baseline calculation (Requirement 14.1)
 * - Drift detection with various distributions (Requirement 14.2)
 * - Metrics writing to S3 (Requirement 14.3, 14.4)
 */
class MonitorHandlerTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    test("should calculate baseline from historical metrics") {
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
            put("batchDate", "2024-01-15")
            put("totalTransactions", 100)
            put("successCount", 100)
            put("errorCount", 0)
            set<com.fasterxml.jackson.databind.node.ObjectNode>(
                "riskDistribution",
                objectMapper.createObjectNode().apply {
                    put("highRisk", 10)
                    put("mediumRisk", 20)
                    put("lowRisk", 70)
                }
            )
            put("avgFraudScore", 0.35)
        }
        
        val storeStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
        
        // Mock historical metrics (3 days of data)
        var getObjectCallCount = 0
        every { mockMetricsS3Client.getObject(any<GetObjectRequest>()) } answers {
            getObjectCallCount++
            if (getObjectCallCount <= 3) {
                val historicalMetrics = objectMapper.createObjectNode().apply {
                    put("avgFraudScore", 0.30 + (getObjectCallCount * 0.01)) // 0.31, 0.32, 0.33
                    put("highRiskPct", 0.08 + (getObjectCallCount * 0.01)) // 0.09, 0.10, 0.11
                }
                ResponseInputStream(
                    GetObjectResponse.builder().build(),
                    AbortableInputStream.create(
                        ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                    )
                )
            } else {
                throw NoSuchKeyException.builder().message("No metrics found").build()
            }
        }
        
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        every { mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        every { mockSnsClient.publish(any<PublishRequest>()) } returns PublishResponse.builder().messageId("msg-1").build()
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "MonitorStage",
            "previousStage" to "StoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        handler.handleRequest(input, mockContext)
        
        // Verify baseline was calculated from historical data
        // Baseline avgFraudScore should be average of 0.31, 0.32, 0.33 = 0.32
        // Baseline highRiskPct should be average of 0.09, 0.10, 0.11 = 0.10
        
        val capturedOutput = slot<RequestBody>()
        verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
        
        val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
        val resultNode = objectMapper.readTree(outputJson)
        
        // Current avgFraudScore = 0.35, baseline = 0.32, drift = 0.03 (< 0.1)
        // Current highRiskPct = 0.10, baseline = 0.10, drift = 0.00 (< 0.05)
        resultNode.get("driftDetected").asBoolean() shouldBe false
    }
    
    test("should detect drift when average score differs by more than 0.1") {
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
            put("batchDate", "2024-01-15")
            put("totalTransactions", 100)
            put("successCount", 100)
            put("errorCount", 0)
            set<com.fasterxml.jackson.databind.node.ObjectNode>(
                "riskDistribution",
                objectMapper.createObjectNode().apply {
                    put("highRisk", 10)
                    put("mediumRisk", 20)
                    put("lowRisk", 70)
                }
            )
            put("avgFraudScore", 0.50) // Current score
        }
        
        val storeStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
        
        // Mock historical metrics with baseline avgFraudScore = 0.35
        every { mockMetricsS3Client.getObject(any<GetObjectRequest>()) } answers {
            val historicalMetrics = objectMapper.createObjectNode().apply {
                put("avgFraudScore", 0.35) // Baseline
                put("highRiskPct", 0.10)
            }
            ResponseInputStream(
                GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                )
            )
        }
        
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        every { mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        val publishedAlerts = mutableListOf<String>()
        
        every { mockSnsClient.publish(any<PublishRequest>()) } answers {
            val request = firstArg<PublishRequest>()
            publishedAlerts.add(request.message())
            PublishResponse.builder().messageId("msg-${publishedAlerts.size}").build()
        }
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "MonitorStage",
            "previousStage" to "StoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        handler.handleRequest(input, mockContext)
        
        // Verify drift was detected (0.50 - 0.35 = 0.15 > 0.1)
        val capturedOutput = slot<RequestBody>()
        verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
        
        val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
        val resultNode = objectMapper.readTree(outputJson)
        
        resultNode.get("driftDetected").asBoolean() shouldBe true
        
        // Use approximate comparison for floating point
        val avgScoreDrift = resultNode.get("avgScoreDrift").asDouble()
        (avgScoreDrift >= 0.14 && avgScoreDrift <= 0.16) shouldBe true
        
        // Verify alert was sent
        publishedAlerts.size shouldBe 1
        publishedAlerts[0] shouldContain "Model Distribution Drift Detected"
        publishedAlerts[0] shouldContain "2024-01-15"
    }
    
    test("should detect drift when high risk percentage differs by more than 0.05") {
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
            put("batchDate", "2024-01-15")
            put("totalTransactions", 100)
            put("successCount", 100)
            put("errorCount", 0)
            set<com.fasterxml.jackson.databind.node.ObjectNode>(
                "riskDistribution",
                objectMapper.createObjectNode().apply {
                    put("highRisk", 20) // 20% high risk
                    put("mediumRisk", 20)
                    put("lowRisk", 60)
                }
            )
            put("avgFraudScore", 0.35)
        }
        
        val storeStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
        
        // Mock historical metrics with baseline highRiskPct = 0.12 (12%)
        every { mockMetricsS3Client.getObject(any<GetObjectRequest>()) } answers {
            val historicalMetrics = objectMapper.createObjectNode().apply {
                put("avgFraudScore", 0.35)
                put("highRiskPct", 0.12) // Baseline
            }
            ResponseInputStream(
                GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                )
            )
        }
        
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        every { mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        val publishedAlerts = mutableListOf<String>()
        
        every { mockSnsClient.publish(any<PublishRequest>()) } answers {
            val request = firstArg<PublishRequest>()
            publishedAlerts.add(request.message())
            PublishResponse.builder().messageId("msg-${publishedAlerts.size}").build()
        }
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "MonitorStage",
            "previousStage" to "StoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        handler.handleRequest(input, mockContext)
        
        // Verify drift was detected (0.20 - 0.12 = 0.08 > 0.05)
        val capturedOutput = slot<RequestBody>()
        verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
        
        val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
        val resultNode = objectMapper.readTree(outputJson)
        
        resultNode.get("driftDetected").asBoolean() shouldBe true
        
        // Use approximate comparison for floating point
        val highRiskDrift = resultNode.get("highRiskDrift").asDouble()
        (highRiskDrift >= 0.07 && highRiskDrift <= 0.09) shouldBe true
        
        // Verify alert was sent
        publishedAlerts.size shouldBe 1
        publishedAlerts[0] shouldContain "Model Distribution Drift Detected"
        publishedAlerts[0] shouldContain "High Risk Percentage"
    }
    
    test("should write metrics to S3 with all required fields") {
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
            put("batchDate", "2024-01-15")
            put("totalTransactions", 100)
            put("successCount", 100)
            put("errorCount", 0)
            set<com.fasterxml.jackson.databind.node.ObjectNode>(
                "riskDistribution",
                objectMapper.createObjectNode().apply {
                    put("highRisk", 10)
                    put("mediumRisk", 20)
                    put("lowRisk", 70)
                }
            )
            put("avgFraudScore", 0.35)
        }
        
        val storeStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
        
        // Mock historical metrics
        every { mockMetricsS3Client.getObject(any<GetObjectRequest>()) } answers {
            val historicalMetrics = objectMapper.createObjectNode().apply {
                put("avgFraudScore", 0.32)
                put("highRiskPct", 0.10)
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
        
        every { mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } answers {
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
        
        // Verify metrics were written to S3
        capturedMetricsWrites.size shouldBe 1
        
        val (metricsRequest, metricsBody) = capturedMetricsWrites[0]
        
        metricsRequest.bucket() shouldBe "test-metrics-bucket"
        metricsRequest.key() shouldBe "metrics/2024-01-15.json"
        
        val metricsJson = metricsBody.contentStreamProvider().newStream().readAllBytes().decodeToString()
        val metricsNode = objectMapper.readTree(metricsJson)
        
        // Verify all required fields are present
        metricsNode.get("batchDate").asText() shouldBe "2024-01-15"
        metricsNode.get("avgFraudScore").asDouble() shouldBe 0.35
        metricsNode.get("highRiskPct").asDouble() shouldBe 0.10
        metricsNode.get("mediumRiskPct").asDouble() shouldBe 0.20
        metricsNode.get("lowRiskPct").asDouble() shouldBe 0.70
        
        // Use approximate comparison for floating point
        val avgScoreDrift = metricsNode.get("avgScoreDrift").asDouble()
        (avgScoreDrift >= 0.02 && avgScoreDrift <= 0.04) shouldBe true
        
        val highRiskDrift = metricsNode.get("highRiskDrift").asDouble()
        (highRiskDrift >= -0.01 && highRiskDrift <= 0.01) shouldBe true
        
        metricsNode.get("driftDetected").asBoolean() shouldBe false
        metricsNode.has("timestamp") shouldBe true
    }
    
    test("should handle first run with no historical data") {
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
            put("batchDate", "2024-01-15")
            put("totalTransactions", 100)
            put("successCount", 100)
            put("errorCount", 0)
            set<com.fasterxml.jackson.databind.node.ObjectNode>(
                "riskDistribution",
                objectMapper.createObjectNode().apply {
                    put("highRisk", 10)
                    put("mediumRisk", 20)
                    put("lowRisk", 70)
                }
            )
            put("avgFraudScore", 0.35)
        }
        
        val storeStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
        
        // Mock no historical metrics (first run)
        every { mockMetricsS3Client.getObject(any<GetObjectRequest>()) } throws NoSuchKeyException.builder().message("No metrics found").build()
        
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        every { mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        every { mockSnsClient.publish(any<PublishRequest>()) } returns PublishResponse.builder().messageId("msg-1").build()
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "MonitorStage",
            "previousStage" to "StoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        handler.handleRequest(input, mockContext)
        
        // Verify handler completes successfully with default baseline (0.0, 0.0)
        val capturedOutput = slot<RequestBody>()
        verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
        
        val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
        val resultNode = objectMapper.readTree(outputJson)
        
        // With baseline (0.0, 0.0), current metrics will trigger drift
        // avgFraudScore = 0.35, drift = 0.35 > 0.1
        resultNode.get("driftDetected").asBoolean() shouldBe true
    }
})
