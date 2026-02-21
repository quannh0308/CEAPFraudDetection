# S3 ListBucket Permission Fix - Bugfix Design

## Overview

The ScoreHandler Lambda fails at runtime with a 403 AccessDenied error when accessing S3 objects because its IAM role policy is missing the `s3:ListBucket` permission. The existing policy grants `s3:GetObject` and `s3:PutObject` on object-level ARNs (`arn:aws:s3:::bucket/*`) for the workflow, config, and data buckets, but the AWS S3 SDK requires `s3:ListBucket` at the bucket-level ARN (`arn:aws:s3:::bucket`) to properly resolve object requests. The fix adds a new `PolicyStatement` granting `s3:ListBucket` on the three bucket-level ARNs to the ScoreHandler's IAM role.

## Glossary

- **Bug_Condition (C)**: The ScoreHandler Lambda attempts an S3 `GetObject` call against the data or config bucket, and the IAM role lacks `s3:ListBucket` on the bucket-level ARN, causing the SDK to receive a 403 instead of a 404 for missing-object checks.
- **Property (P)**: The ScoreHandler Lambda successfully retrieves objects from S3 without `s3:ListBucket` authorization errors when the permission is present on the bucket-level ARN.
- **Preservation**: All existing permissions (`s3:GetObject`, `s3:PutObject`, `sagemaker:InvokeEndpoint`) and all other Lambda handlers (StoreHandler, AlertHandler, MonitorHandler) remain unchanged.
- **ScoreHandler**: The Lambda function in `fraud-inference-pipeline/src/main/kotlin/com/fraud/inference/ScoreHandler.kt` that loads transaction batches from S3, reads endpoint config from S3, and invokes SageMaker for scoring.
- **InferencePipelineStack**: The CDK stack in `infrastructure/src/main/kotlin/com/frauddetection/infrastructure/InferencePipelineStack.kt` that defines IAM policies for all inference Lambda handlers.

## Bug Details

### Fault Condition

The bug manifests when the ScoreHandler Lambda calls `s3Client.getObject()` on the data bucket or config bucket. The AWS S3 SDK internally requires `s3:ListBucket` at the bucket level to distinguish between "object not found" (404) and "access denied" (403). Without this permission, the SDK returns a 403 AccessDenied error, which the ScoreHandler wraps in an `IllegalStateException`, causing the ScoreStage to fail.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type { lambdaName: String, s3Action: String, bucketArn: String, iamPolicy: IAMPolicy }
  OUTPUT: boolean

  RETURN input.lambdaName == "ScoreHandler"
         AND input.s3Action IN ["GetObject", "PutObject"]
         AND input.bucketArn IN [dataBucketArn, configBucketArn, workflowBucketArn]
         AND NOT iamPolicy.grants("s3:ListBucket", bucketLevelArn(input.bucketArn))
END FUNCTION
```

### Examples

- ScoreHandler calls `s3Client.getObject()` on `s3://fraud-detection-data-<suffix>/daily-batches/2026-02-21.json` → **Expected:** returns object content. **Actual:** throws `IllegalStateException` wrapping 403 AccessDenied.
- ScoreHandler calls `s3Client.getObject()` on config bucket for `current-endpoint.json` → **Expected:** returns endpoint metadata JSON. **Actual:** throws `IllegalStateException` wrapping 403 AccessDenied.
- ScoreHandler calls `s3Client.getObject()` on a genuinely non-existent key → **Expected:** throws `IllegalStateException` with meaningful "not found" message. **Actual:** throws 403 AccessDenied (indistinguishable from missing permission).
- StoreHandler calls `s3Client.getObject()` on workflow bucket → **Expected and Actual:** works correctly (not affected by this bug since StoreHandler only accesses the workflow bucket, which may or may not exhibit the same issue depending on SDK behavior).

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- The ScoreHandler's existing `s3:GetObject` and `s3:PutObject` permissions on object-level ARNs must remain intact
- The ScoreHandler's `sagemaker:InvokeEndpoint` permission must remain unchanged
- StoreHandler, AlertHandler, and MonitorHandler IAM policies must not be modified
- The Step Functions workflow definition, retry logic, and catch states must remain unchanged
- The ScoreHandler Lambda configuration (runtime, memory, timeout, environment variables) must remain unchanged

**Scope:**
All inputs that do NOT involve the ScoreHandler's S3 bucket-level permissions should be completely unaffected by this fix. This includes:
- SageMaker endpoint invocations
- DynamoDB writes by StoreHandler
- SNS publishes by AlertHandler and MonitorHandler
- All other Lambda handler S3 operations (they have their own policies)

## Hypothesized Root Cause

Based on the bug description and code analysis, the root cause is:

1. **Missing `s3:ListBucket` in IAM Policy**: The ScoreHandler's IAM policy (lines ~155-165 of `InferencePipelineStack.kt`) grants `s3:GetObject` and `s3:PutObject` on `arn:aws:s3:::$bucketName/*` (object-level) but does not include `s3:ListBucket` on `arn:aws:s3:::$bucketName` (bucket-level). The AWS S3 SDK uses a HEAD request internally which requires `s3:ListBucket` to properly return 404 vs 403 status codes.

2. **Bucket-level vs Object-level ARN Mismatch**: `s3:ListBucket` is a bucket-level action that requires the ARN without the `/*` suffix. The existing policy only has object-level ARNs (`/*` suffix), so even if `s3:ListBucket` were added to the existing statement's actions list, it would need a separate resource entry or a separate policy statement with bucket-level ARNs.

## Correctness Properties

Property 1: Fault Condition - ScoreHandler S3 ListBucket Permission Granted

_For any_ deployment of the InferencePipelineStack where the ScoreHandler Lambda accesses S3 objects in the data, config, or workflow buckets, the synthesized CloudFormation template SHALL include an IAM policy statement granting `s3:ListBucket` on the bucket-level ARNs (`arn:aws:s3:::dataBucketName`, `arn:aws:s3:::configBucketName`, `arn:aws:s3:::workflowBucketName`) for the ScoreHandler's execution role.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Existing Permissions and Other Handlers Unchanged

_For any_ deployment of the InferencePipelineStack, the fixed code SHALL produce the same IAM policies for StoreHandler, AlertHandler, and MonitorHandler as the original code, and SHALL preserve all existing `s3:GetObject`, `s3:PutObject`, and `sagemaker:InvokeEndpoint` permissions for the ScoreHandler.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `infrastructure/src/main/kotlin/com/frauddetection/infrastructure/InferencePipelineStack.kt`

**Function**: `init` block (ScoreHandler permission section)

**Specific Changes**:
1. **Add a new `PolicyStatement` for `s3:ListBucket`**: Add a third `addToRolePolicy` call on `scoreHandler` that grants `s3:ListBucket` on the bucket-level ARNs (without `/*` suffix) for all three buckets: `workflowBucketName`, `configBucketName`, and `dataBucketName`.

   ```kotlin
   scoreHandler.addToRolePolicy(
       PolicyStatement.Builder.create()
           .effect(Effect.ALLOW)
           .actions(listOf("s3:ListBucket"))
           .resources(listOf(
               "arn:aws:s3:::$workflowBucketName",
               "arn:aws:s3:::$configBucketName",
               "arn:aws:s3:::$dataBucketName"
           ))
           .build()
   )
   ```

2. **Placement**: Insert this new policy statement immediately after the existing `s3:GetObject`/`s3:PutObject` policy statement and before the `sagemaker:InvokeEndpoint` policy statement, keeping the S3 permissions grouped together.

3. **No other changes**: No modifications to any other handler policies, Lambda configurations, or workflow definitions.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code (missing permission in synthesized template), then verify the fix works correctly and preserves existing behavior.

### Exploratory Fault Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis by synthesizing the CDK stack and inspecting the CloudFormation template.

**Test Plan**: Write a CDK synthesis test that instantiates `InferencePipelineStack` and inspects the generated CloudFormation template for the ScoreHandler's IAM role policies. Assert that `s3:ListBucket` is present on bucket-level ARNs. Run on UNFIXED code to observe the assertion failure.

**Test Cases**:
1. **ListBucket on Data Bucket**: Assert ScoreHandler role has `s3:ListBucket` on `arn:aws:s3:::dataBucketName` (will fail on unfixed code)
2. **ListBucket on Config Bucket**: Assert ScoreHandler role has `s3:ListBucket` on `arn:aws:s3:::configBucketName` (will fail on unfixed code)
3. **ListBucket on Workflow Bucket**: Assert ScoreHandler role has `s3:ListBucket` on `arn:aws:s3:::workflowBucketName` (will fail on unfixed code)

**Expected Counterexamples**:
- The synthesized CloudFormation template does not contain any policy statement with `s3:ListBucket` for the ScoreHandler role
- Possible causes: the action was simply never added to the IAM policy during initial implementation

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed stack produces the expected IAM policy.

**Pseudocode:**
```
FOR ALL bucketName IN [workflowBucketName, configBucketName, dataBucketName] DO
  template := synthesize(InferencePipelineStack_fixed)
  scoreHandlerPolicies := extractPolicies(template, "ScoreHandler")
  ASSERT "s3:ListBucket" IN scoreHandlerPolicies.actions
  ASSERT "arn:aws:s3:::${bucketName}" IN scoreHandlerPolicies.resources
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed stack produces the same CloudFormation template as the original for all non-ScoreHandler-ListBucket resources.

**Pseudocode:**
```
FOR ALL resource IN synthesizedTemplate WHERE resource != ScoreHandlerListBucketPolicy DO
  ASSERT originalTemplate[resource] == fixedTemplate[resource]
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It can generate many IAM policy configurations and verify no unintended changes
- It catches edge cases where bucket names might affect ARN construction
- It provides strong guarantees that only the intended permission is added

**Test Plan**: Observe the synthesized CloudFormation template on UNFIXED code for all non-ScoreHandler resources, then write tests to verify these remain identical after the fix.

**Test Cases**:
1. **StoreHandler Policy Preservation**: Verify StoreHandler IAM policies are identical before and after fix
2. **AlertHandler Policy Preservation**: Verify AlertHandler IAM policies are identical before and after fix
3. **MonitorHandler Policy Preservation**: Verify MonitorHandler IAM policies are identical before and after fix
4. **ScoreHandler Existing Permissions Preservation**: Verify `s3:GetObject`, `s3:PutObject`, and `sagemaker:InvokeEndpoint` permissions remain unchanged
5. **Workflow Definition Preservation**: Verify Step Functions state machine definition is unchanged

### Unit Tests

- Synthesize the CDK stack and assert `s3:ListBucket` is present in ScoreHandler's IAM role policies
- Assert the `s3:ListBucket` resources use bucket-level ARNs (no `/*` suffix)
- Assert all three buckets (workflow, config, data) are included in the ListBucket resources
- Assert no other handler roles received `s3:ListBucket`

### Property-Based Tests

- Generate random bucket name suffixes and verify the synthesized template always includes `s3:ListBucket` on correctly constructed bucket-level ARNs for ScoreHandler
- Generate random stack configurations and verify StoreHandler, AlertHandler, and MonitorHandler policies are never modified by the fix
- Verify that the total number of policy statements for non-ScoreHandler roles remains constant across configurations

### Integration Tests

- Deploy the fixed stack to a test environment and invoke the ScoreHandler Lambda with a valid transaction batch path to verify S3 access succeeds
- Verify the full Step Functions workflow completes the ScoreStage without 403 errors
- Verify StoreHandler, AlertHandler, and MonitorHandler continue to function correctly after deployment
