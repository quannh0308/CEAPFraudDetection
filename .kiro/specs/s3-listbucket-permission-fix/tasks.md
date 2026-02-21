# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Fault Condition** - ScoreHandler Missing s3:ListBucket Permission
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists in the synthesized CloudFormation template
  - **Scoped PBT Approach**: Use Kotest property-based testing (`kotest-property`) to generate random bucket suffixes and verify the synthesized template includes `s3:ListBucket` on bucket-level ARNs for the ScoreHandler role
  - Create test file: `infrastructure/src/test/kotlin/com/frauddetection/infrastructure/ScoreHandlerListBucketPropertyTest.kt`
  - Use Kotest `FunSpec` with `checkAll` from `io.kotest.property`
  - For each generated bucket suffix, synthesize `InferencePipelineStack` and extract the CloudFormation template
  - Extract all IAM policy statements attached to the ScoreHandler role
  - Assert that at least one policy statement grants `s3:ListBucket` action
  - Assert that the `s3:ListBucket` statement includes bucket-level ARNs (without `/*` suffix) for all three buckets: `fraud-detection-workflow-<suffix>`, `fraud-detection-config-<suffix>`, `fraud-detection-data-<suffix>`
  - Bug condition from design: `isBugCondition(input)` where `lambdaName == "ScoreHandler" AND s3Action IN ["GetObject", "PutObject"] AND NOT iamPolicy.grants("s3:ListBucket", bucketLevelArn)`
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists because no `s3:ListBucket` statement is present)
  - Document counterexamples found (e.g., "For suffix 'abc123', ScoreHandler role has no s3:ListBucket policy statement")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 2.1, 2.2_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Existing Permissions and Other Handlers Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Create test in the same file: `infrastructure/src/test/kotlin/com/frauddetection/infrastructure/ScoreHandlerListBucketPropertyTest.kt`
  - Observe on UNFIXED code: synthesize the stack and record all IAM policy statements for StoreHandler, AlertHandler, MonitorHandler, and ScoreHandler's existing `s3:GetObject`/`s3:PutObject`/`sagemaker:InvokeEndpoint` permissions
  - Write property-based test using Kotest `checkAll` with random bucket suffixes:
    - For all generated suffixes, ScoreHandler role MUST have `s3:GetObject` and `s3:PutObject` on object-level ARNs (`arn:aws:s3:::bucket/*`) for workflow, config, and data buckets
    - For all generated suffixes, ScoreHandler role MUST have `sagemaker:InvokeEndpoint` on `*`
    - For all generated suffixes, StoreHandler role MUST have `s3:GetObject` and `s3:PutObject` on workflow bucket only (no config or data bucket)
    - For all generated suffixes, AlertHandler role MUST have `s3:GetObject` and `s3:PutObject` on workflow bucket only
    - For all generated suffixes, MonitorHandler role MUST have `s3:GetObject` and `s3:PutObject` on workflow bucket only
    - For all generated suffixes, the Step Functions state machine definition MUST remain unchanged
    - For all generated suffixes, the total number of Lambda functions MUST be exactly 4
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [-] 3. Fix for ScoreHandler missing s3:ListBucket permission

  - [x] 3.1 Implement the fix
    - Edit file: `infrastructure/src/main/kotlin/com/frauddetection/infrastructure/InferencePipelineStack.kt`
    - Add a new `PolicyStatement` granting `s3:ListBucket` on bucket-level ARNs for the ScoreHandler
    - Insert immediately after the existing `s3:GetObject`/`s3:PutObject` policy statement (after ~line 165) and before the `sagemaker:InvokeEndpoint` statement
    - The new statement should be:
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
    - No other files or handler policies should be modified
    - _Bug_Condition: isBugCondition(input) where lambdaName == "ScoreHandler" AND NOT iamPolicy.grants("s3:ListBucket", bucketLevelArn)_
    - _Expected_Behavior: ScoreHandler IAM role includes s3:ListBucket on bucket-level ARNs for all three buckets_
    - _Preservation: StoreHandler, AlertHandler, MonitorHandler policies unchanged; ScoreHandler existing s3:GetObject, s3:PutObject, sagemaker:InvokeEndpoint unchanged_
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_

  - [ ] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - ScoreHandler s3:ListBucket Permission Granted
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior: s3:ListBucket present on bucket-level ARNs
    - When this test passes, it confirms the expected behavior is satisfied
    - Run `ScoreHandlerListBucketPropertyTest` fault condition test
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Existing Permissions and Other Handlers Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run `ScoreHandlerListBucketPropertyTest` preservation tests
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all existing handler permissions and workflow definition are unchanged after fix
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Run the full infrastructure test suite: `cd infrastructure && ../gradlew test`
  - Ensure all tests pass including the existing `InferencePipelineStackTest` and the new `ScoreHandlerListBucketPropertyTest`
  - If all tests pass, the fix is ready for deployment
  - User can deploy with: `./deploy-inference-pipeline.sh`
  - Ask the user if questions arise
