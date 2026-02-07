package com.fraud.common.models

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain

class StageResultTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()

    test("StageResult should be created with SUCCESS status") {
        val result = StageResult(
            status = "SUCCESS",
            stage = "ScoreStage",
            recordsProcessed = 100,
            errorMessage = null
        )

        result.status shouldBe "SUCCESS"
        result.stage shouldBe "ScoreStage"
        result.recordsProcessed shouldBe 100
        result.errorMessage shouldBe null
        result.isSuccess() shouldBe true
        result.isFailure() shouldBe false
    }

    test("StageResult should be created with FAILED status") {
        val result = StageResult(
            status = "FAILED",
            stage = "TrainStage",
            recordsProcessed = 50,
            errorMessage = "Training job failed due to insufficient data"
        )

        result.status shouldBe "FAILED"
        result.stage shouldBe "TrainStage"
        result.recordsProcessed shouldBe 50
        result.errorMessage shouldBe "Training job failed due to insufficient data"
        result.isSuccess() shouldBe false
        result.isFailure() shouldBe true
    }

    test("StageResult should reject invalid status") {
        val exception = shouldThrow<IllegalArgumentException> {
            StageResult(
                status = "PENDING",
                stage = "ScoreStage",
                recordsProcessed = 0,
                errorMessage = null
            )
        }

        exception.message shouldContain "Status must be either 'SUCCESS' or 'FAILED'"
        exception.message shouldContain "PENDING"
    }

    test("StageResult should reject negative recordsProcessed") {
        val exception = shouldThrow<IllegalArgumentException> {
            StageResult(
                status = "SUCCESS",
                stage = "ScoreStage",
                recordsProcessed = -10,
                errorMessage = null
            )
        }

        exception.message shouldContain "Records processed must be non-negative"
        exception.message shouldContain "-10"
    }

    test("StageResult should reject FAILED status without error message") {
        val exception = shouldThrow<IllegalArgumentException> {
            StageResult(
                status = "FAILED",
                stage = "ScoreStage",
                recordsProcessed = 0,
                errorMessage = null
            )
        }

        exception.message shouldContain "Error message must be provided when status is FAILED"
    }

    test("StageResult should reject FAILED status with blank error message") {
        val exception = shouldThrow<IllegalArgumentException> {
            StageResult(
                status = "FAILED",
                stage = "ScoreStage",
                recordsProcessed = 0,
                errorMessage = "   "
            )
        }

        exception.message shouldContain "Error message must be provided when status is FAILED"
    }

    test("StageResult.success should create successful result") {
        val result = StageResult.success("ScoreStage", 150)

        result.status shouldBe "SUCCESS"
        result.stage shouldBe "ScoreStage"
        result.recordsProcessed shouldBe 150
        result.errorMessage shouldBe null
        result.isSuccess() shouldBe true
    }

    test("StageResult.success should create successful result with default recordsProcessed") {
        val result = StageResult.success("DeployStage")

        result.status shouldBe "SUCCESS"
        result.stage shouldBe "DeployStage"
        result.recordsProcessed shouldBe 0
        result.errorMessage shouldBe null
    }

    test("StageResult.failure should create failed result with error message") {
        val result = StageResult.failure(
            stage = "TrainStage",
            errorMessage = "SageMaker training job timed out",
            recordsProcessed = 0
        )

        result.status shouldBe "FAILED"
        result.stage shouldBe "TrainStage"
        result.recordsProcessed shouldBe 0
        result.errorMessage shouldBe "SageMaker training job timed out"
        result.isFailure() shouldBe true
    }

    test("StageResult.failure should create failed result from exception") {
        val exception = RuntimeException("S3 access denied")
        val result = StageResult.failure("ScoreStage", exception, 25)

        result.status shouldBe "FAILED"
        result.stage shouldBe "ScoreStage"
        result.recordsProcessed shouldBe 25
        result.errorMessage shouldNotBe null
        result.errorMessage shouldContain "RuntimeException"
        result.errorMessage shouldContain "S3 access denied"
    }

    test("StageResult.failure should handle exception with cause") {
        val cause = IllegalStateException("Invalid endpoint configuration")
        val exception = RuntimeException("Endpoint invocation failed", cause)
        val result = StageResult.failure("ScoreStage", exception)

        result.status shouldBe "FAILED"
        result.errorMessage shouldNotBe null
        result.errorMessage shouldContain "RuntimeException"
        result.errorMessage shouldContain "Endpoint invocation failed"
        result.errorMessage shouldContain "IllegalStateException"
        result.errorMessage shouldContain "Invalid endpoint configuration"
    }

    test("StageResult.failure should handle exception without message") {
        val exception = RuntimeException()
        val result = StageResult.failure("AlertStage", exception)

        result.status shouldBe "FAILED"
        result.errorMessage shouldNotBe null
        result.errorMessage shouldContain "RuntimeException"
        result.errorMessage shouldContain "No error message"
    }

    test("StageResult should serialize to JSON correctly") {
        val result = StageResult(
            status = "SUCCESS",
            stage = "StoreStage",
            recordsProcessed = 200,
            errorMessage = null
        )

        val json = objectMapper.writeValueAsString(result)
        
        json shouldNotBe null
        json.contains("\"status\":\"SUCCESS\"") shouldBe true
        json.contains("\"stage\":\"StoreStage\"") shouldBe true
        json.contains("\"recordsProcessed\":200") shouldBe true
    }

    test("StageResult should deserialize from JSON correctly") {
        val json = """
            {
                "status": "FAILED",
                "stage": "TrainStage",
                "recordsProcessed": 0,
                "errorMessage": "Model accuracy below threshold: 0.85 < 0.90"
            }
        """.trimIndent()

        val result: StageResult = objectMapper.readValue(json)

        result.status shouldBe "FAILED"
        result.stage shouldBe "TrainStage"
        result.recordsProcessed shouldBe 0
        result.errorMessage shouldBe "Model accuracy below threshold: 0.85 < 0.90"
        result.isFailure() shouldBe true
    }

    test("StageResult should support data class copy") {
        val original = StageResult.success("ScoreStage", 100)
        val copied = original.copy(recordsProcessed = 150)

        copied.status shouldBe "SUCCESS"
        copied.stage shouldBe "ScoreStage"
        copied.recordsProcessed shouldBe 150
        copied.errorMessage shouldBe null
    }

    test("StageResult constants should be correct") {
        StageResult.STATUS_SUCCESS shouldBe "SUCCESS"
        StageResult.STATUS_FAILED shouldBe "FAILED"
    }

    test("StageResult should accept zero recordsProcessed") {
        val result = StageResult.success("DeployStage", 0)

        result.recordsProcessed shouldBe 0
        result.isSuccess() shouldBe true
    }

    test("StageResult should handle large recordsProcessed values") {
        val result = StageResult.success("ScoreStage", 1_000_000)

        result.recordsProcessed shouldBe 1_000_000
        result.isSuccess() shouldBe true
    }

    test("StageResult should preserve detailed error messages") {
        val detailedError = """
            Failed to invoke SageMaker endpoint: fraud-detection-prod
            Reason: EndpointNotFoundException
            Details: Could not find endpoint arn:aws:sagemaker:us-east-1:123456789012:endpoint/fraud-detection-prod
            Suggestion: Verify endpoint exists and is in InService status
        """.trimIndent()

        val result = StageResult.failure("ScoreStage", detailedError, 0)

        result.errorMessage shouldBe detailedError
        result.errorMessage shouldContain "EndpointNotFoundException"
        result.errorMessage shouldContain "fraud-detection-prod"
    }

    test("StageResult should work for all pipeline stages") {
        val stages = listOf(
            "DataPrepStage", "TrainStage", "EvaluateStage", "DeployStage",
            "ScoreStage", "StoreStage", "AlertStage", "MonitorStage"
        )

        stages.forEach { stage ->
            val successResult = StageResult.success(stage, 100)
            successResult.stage shouldBe stage
            successResult.isSuccess() shouldBe true

            val failureResult = StageResult.failure(stage, "Test error")
            failureResult.stage shouldBe stage
            failureResult.isFailure() shouldBe true
        }
    }

    test("StageResult should handle partial processing failures") {
        // Scenario: 75 out of 100 records processed before failure
        val result = StageResult.failure(
            stage = "StoreStage",
            errorMessage = "DynamoDB throttling exception after 75 records",
            recordsProcessed = 75
        )

        result.status shouldBe "FAILED"
        result.recordsProcessed shouldBe 75
        result.errorMessage shouldContain "75 records"
        result.isFailure() shouldBe true
    }

    test("StageResult should serialize and deserialize with null errorMessage") {
        val original = StageResult.success("ScoreStage", 50)
        val json = objectMapper.writeValueAsString(original)
        val deserialized: StageResult = objectMapper.readValue(json)

        deserialized.status shouldBe original.status
        deserialized.stage shouldBe original.stage
        deserialized.recordsProcessed shouldBe original.recordsProcessed
        deserialized.errorMessage shouldBe null
    }
})
