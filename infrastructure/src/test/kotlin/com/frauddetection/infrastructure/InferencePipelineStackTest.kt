package com.frauddetection.infrastructure

import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import software.amazon.awscdk.App

/**
 * Unit tests for InferencePipelineStack
 * 
 * Tests verify:
 * - Stack synthesis succeeds
 * - All required resources are created
 * - Resource configurations match requirements
 * 
 * Validates: Requirements 12.1, 12.2
 */
class InferencePipelineStackTest : FunSpec({
    
    test("stack should synthesize successfully") {
        // Given
        val app = App()
        
        // When
        val stack = InferencePipelineStack(
            app,
            "TestInferenceStack",
            software.amazon.awscdk.StackProps.builder().build(),
            "test",
            "TestTrainingStack"
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
        val stack = InferencePipelineStack(
            app,
            "TestInferenceStack",
            software.amazon.awscdk.StackProps.builder().build(),
            "test",
            "TestTrainingStack"
        )
        
        // When
        val assembly = app.synth()
        val stackArtifact = assembly.getStackByName(stack.stackName)
        val template = stackArtifact.template as Map<*, *>
        val resources = template["Resources"] as? Map<*, *>
        
        // Then
        resources shouldNotBe null
        
        // Count resource types - CDK creates additional IAM roles/policies
        val dynamoTables = resources!!.values.count { 
            (it as? Map<*, *>)?.get("Type") == "AWS::DynamoDB::Table" 
        }
        val snsTopics = resources.values.count { 
            (it as? Map<*, *>)?.get("Type") == "AWS::SNS::Topic" 
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
        (dynamoTables >= 1) shouldBe true
        (snsTopics >= 2) shouldBe true
        (lambdas >= 4) shouldBe true
        (stateMachines >= 1) shouldBe true
        (eventRules >= 1) shouldBe true
    }
    
    test("stack should export required outputs") {
        // Given
        val app = App()
        val stack = InferencePipelineStack(
            app,
            "TestInferenceStack",
            software.amazon.awscdk.StackProps.builder().build(),
            "test",
            "TestTrainingStack"
        )
        
        // When
        val assembly = app.synth()
        val stackArtifact = assembly.getStackByName(stack.stackName)
        val template = stackArtifact.template as Map<*, *>
        val outputs = template["Outputs"] as? Map<*, *>
        
        // Then
        outputs shouldNotBe null
        outputs!!.keys.any { it.toString().contains("InferenceWorkflowArn") } shouldBe true
        outputs.keys.any { it.toString().contains("FraudScoresTableName") } shouldBe true
        outputs.keys.any { it.toString().contains("AlertTopicArn") } shouldBe true
        outputs.keys.any { it.toString().contains("MonitoringTopicArn") } shouldBe true
    }
})
