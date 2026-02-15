package com.frauddetection.infrastructure

import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import software.amazon.awscdk.App
import software.amazon.awscdk.cloudassembly.schema.ArtifactType

/**
 * Unit tests for TrainingPipelineStack
 * 
 * Tests verify:
 * - Stack synthesis succeeds
 * - All required resources are created
 * - Resource configurations match requirements
 * 
 * Validates: Requirements 12.1, 12.2
 */
class TrainingPipelineStackTest : FunSpec({
    
    test("stack should synthesize successfully") {
        // Given
        val app = App()
        
        // When
        val stack = TrainingPipelineStack(
            app,
            "TestTrainingStack",
            software.amazon.awscdk.StackProps.builder().build(),
            envName = "test",
            bucketSuffix = "test-suffix"
        )
        
        // Then - should not throw exception
        val assembly = app.synth()
        assembly shouldNotBe null
        
        // Verify stack exists in assembly
        val stackArtifact = assembly.getStackByName(stack.stackName)
        stackArtifact shouldNotBe null
        stackArtifact.template shouldNotBe null
    }
    
    test("stack should create required resources") {
        // Given
        val app = App()
        val stack = TrainingPipelineStack(
            app,
            "TestTrainingStack",
            software.amazon.awscdk.StackProps.builder().build(),
            envName = "test",
            bucketSuffix = "test-suffix"
        )
        
        // When
        val assembly = app.synth()
        val stackArtifact = assembly.getStackByName(stack.stackName)
        val template = stackArtifact.template as Map<*, *>
        val resources = template["Resources"] as? Map<*, *>
        
        // Then
        resources shouldNotBe null
        
        // Count resource types - CDK creates additional IAM roles/policies
        val s3Buckets = resources!!.values.count { 
            (it as? Map<*, *>)?.get("Type") == "AWS::S3::Bucket" 
        }
        val glueJobs = resources.values.count { 
            (it as? Map<*, *>)?.get("Type") == "AWS::Glue::Job" 
        }
        val lambdas = resources.values.count { 
            (it as? Map<*, *>)?.get("Type") == "AWS::Lambda::Function" 
        }
        val stateMachines = resources.values.count { 
            (it as? Map<*, *>)?.get("Type") == "AWS::StepFunctions::StateMachine" 
        }
        val eventRules = resources.values.count { 
            (it as? Map<*, *>)?.get("Type") == "AWS::Events::Rule" 
        }
        
        // Verify minimum required resources exist
        (s3Buckets >= 4) shouldBe true
        (glueJobs >= 1) shouldBe true
        (lambdas >= 3) shouldBe true
        (stateMachines >= 1) shouldBe true
        (eventRules >= 1) shouldBe true
    }
    
    test("stack should export required outputs") {
        // Given
        val app = App()
        val stack = TrainingPipelineStack(
            app,
            "TestTrainingStack",
            software.amazon.awscdk.StackProps.builder().build(),
            envName = "test",
            bucketSuffix = "test-suffix"
        )
        
        // When
        val assembly = app.synth()
        val stackArtifact = assembly.getStackByName(stack.stackName)
        val template = stackArtifact.template as Map<*, *>
        val outputs = template["Outputs"] as? Map<*, *>
        
        // Then
        outputs shouldNotBe null
        outputs!!.keys.any { it.toString().contains("TrainingWorkflowArn") } shouldBe true
        outputs.keys.any { it.toString().contains("WorkflowBucketName") } shouldBe true
        outputs.keys.any { it.toString().contains("DataBucketName") } shouldBe true
        outputs.keys.any { it.toString().contains("ModelsBucketName") } shouldBe true
        outputs.keys.any { it.toString().contains("ConfigBucketName") } shouldBe true
    }
})
