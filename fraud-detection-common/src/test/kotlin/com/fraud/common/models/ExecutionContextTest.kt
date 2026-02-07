package com.fraud.common.models

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe

class ExecutionContextTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()

    test("ExecutionContext should be created with all properties") {
        val context = ExecutionContext(
            executionId = "exec-123",
            currentStage = "ScoreStage",
            previousStage = "DataPrepStage",
            workflowBucket = "fraud-detection-workflow",
            initialData = mapOf("key" to "value")
        )

        context.executionId shouldBe "exec-123"
        context.currentStage shouldBe "ScoreStage"
        context.previousStage shouldBe "DataPrepStage"
        context.workflowBucket shouldBe "fraud-detection-workflow"
        context.initialData shouldBe mapOf("key" to "value")
    }

    test("ExecutionContext should be created with default values") {
        val context = ExecutionContext()

        context.executionId shouldBe ""
        context.currentStage shouldBe ""
        context.previousStage shouldBe null
        context.workflowBucket shouldBe ""
        context.initialData shouldBe null
    }

    test("isFirstStage should return true when previousStage is null") {
        val context = ExecutionContext(
            executionId = "exec-123",
            currentStage = "DataPrepStage",
            previousStage = null,
            workflowBucket = "fraud-detection-workflow"
        )

        context.isFirstStage() shouldBe true
    }

    test("isFirstStage should return false when previousStage is not null") {
        val context = ExecutionContext(
            executionId = "exec-123",
            currentStage = "TrainStage",
            previousStage = "DataPrepStage",
            workflowBucket = "fraud-detection-workflow"
        )

        context.isFirstStage() shouldBe false
    }

    test("getInputS3Key should return null for first stage") {
        val context = ExecutionContext(
            executionId = "exec-123",
            currentStage = "DataPrepStage",
            previousStage = null,
            workflowBucket = "fraud-detection-workflow"
        )

        context.getInputS3Key() shouldBe null
    }

    test("getInputS3Key should return correct path for non-first stage") {
        val context = ExecutionContext(
            executionId = "exec-456",
            currentStage = "TrainStage",
            previousStage = "DataPrepStage",
            workflowBucket = "fraud-detection-workflow"
        )

        context.getInputS3Key() shouldBe "executions/exec-456/DataPrepStage/output.json"
    }

    test("getOutputS3Key should return correct path") {
        val context = ExecutionContext(
            executionId = "exec-789",
            currentStage = "ScoreStage",
            previousStage = null,
            workflowBucket = "fraud-detection-workflow"
        )

        context.getOutputS3Key() shouldBe "executions/exec-789/ScoreStage/output.json"
    }

    test("getOutputS3Key should work for any stage") {
        val stages = listOf("DataPrepStage", "TrainStage", "EvaluateStage", "DeployStage", 
                           "ScoreStage", "StoreStage", "AlertStage", "MonitorStage")
        
        stages.forEach { stage ->
            val context = ExecutionContext(
                executionId = "exec-test",
                currentStage = stage,
                workflowBucket = "test-bucket"
            )
            
            context.getOutputS3Key() shouldBe "executions/exec-test/$stage/output.json"
        }
    }

    test("ExecutionContext should serialize to JSON correctly") {
        val context = ExecutionContext(
            executionId = "exec-serialize",
            currentStage = "ScoreStage",
            previousStage = "DataPrepStage",
            workflowBucket = "fraud-detection-workflow",
            initialData = mapOf("batchDate" to "2024-01-15", "count" to 100)
        )

        val json = objectMapper.writeValueAsString(context)
        
        json shouldNotBe null
        json.contains("\"executionId\":\"exec-serialize\"") shouldBe true
        json.contains("\"currentStage\":\"ScoreStage\"") shouldBe true
        json.contains("\"previousStage\":\"DataPrepStage\"") shouldBe true
        json.contains("\"workflowBucket\":\"fraud-detection-workflow\"") shouldBe true
    }

    test("ExecutionContext should deserialize from JSON correctly") {
        val json = """
            {
                "executionId": "exec-deserialize",
                "currentStage": "TrainStage",
                "previousStage": "DataPrepStage",
                "workflowBucket": "fraud-detection-workflow",
                "initialData": {
                    "trainDataPath": "s3://bucket/train.parquet",
                    "validationDataPath": "s3://bucket/validation.parquet"
                }
            }
        """.trimIndent()

        val context: ExecutionContext = objectMapper.readValue(json)

        context.executionId shouldBe "exec-deserialize"
        context.currentStage shouldBe "TrainStage"
        context.previousStage shouldBe "DataPrepStage"
        context.workflowBucket shouldBe "fraud-detection-workflow"
        context.initialData shouldNotBe null
        context.initialData?.get("trainDataPath") shouldBe "s3://bucket/train.parquet"
    }

    test("ExecutionContext should deserialize from JSON with null previousStage") {
        val json = """
            {
                "executionId": "exec-first-stage",
                "currentStage": "DataPrepStage",
                "previousStage": null,
                "workflowBucket": "fraud-detection-workflow",
                "initialData": {
                    "datasetS3Path": "s3://bucket/dataset.csv"
                }
            }
        """.trimIndent()

        val context: ExecutionContext = objectMapper.readValue(json)

        context.executionId shouldBe "exec-first-stage"
        context.currentStage shouldBe "DataPrepStage"
        context.previousStage shouldBe null
        context.isFirstStage() shouldBe true
        context.getInputS3Key() shouldBe null
    }

    test("ExecutionContext should support data class copy") {
        val original = ExecutionContext(
            executionId = "exec-original",
            currentStage = "ScoreStage",
            previousStage = null,
            workflowBucket = "original-bucket"
        )

        val copied = original.copy(
            currentStage = "StoreStage",
            previousStage = "ScoreStage"
        )

        copied.executionId shouldBe "exec-original"
        copied.currentStage shouldBe "StoreStage"
        copied.previousStage shouldBe "ScoreStage"
        copied.workflowBucket shouldBe "original-bucket"
        copied.isFirstStage() shouldBe false
    }

    test("ExecutionContext should handle complex initialData") {
        val complexData = mapOf(
            "string" to "value",
            "number" to 42,
            "boolean" to true,
            "list" to listOf(1, 2, 3),
            "nested" to mapOf("key" to "nestedValue")
        )

        val context = ExecutionContext(
            executionId = "exec-complex",
            currentStage = "DataPrepStage",
            workflowBucket = "test-bucket",
            initialData = complexData
        )

        context.initialData shouldBe complexData
        context.initialData?.get("string") shouldBe "value"
        context.initialData?.get("number") shouldBe 42
        context.initialData?.get("boolean") shouldBe true
    }

    test("ExecutionContext path construction should follow convention") {
        val context = ExecutionContext(
            executionId = "arn:aws:states:us-east-1:123456789012:execution:MyWorkflow:exec-abc",
            currentStage = "ScoreStage",
            previousStage = "DataPrepStage",
            workflowBucket = "fraud-detection-workflow-123456789012"
        )

        // Input path should follow convention: executions/{executionId}/{previousStage}/output.json
        context.getInputS3Key() shouldBe 
            "executions/arn:aws:states:us-east-1:123456789012:execution:MyWorkflow:exec-abc/DataPrepStage/output.json"
        
        // Output path should follow convention: executions/{executionId}/{currentStage}/output.json
        context.getOutputS3Key() shouldBe 
            "executions/arn:aws:states:us-east-1:123456789012:execution:MyWorkflow:exec-abc/ScoreStage/output.json"
    }
})
