# Bugfix Requirements Document

## Introduction

The ScoreHandler Lambda in the FraudDetectionInference pipeline fails at runtime with a 403 AccessDenied error when attempting to load transaction batches from the S3 data bucket. The root cause is that the ScoreHandler's IAM role policy in `InferencePipelineStack.kt` grants `s3:GetObject` and `s3:PutObject` on the data bucket's objects but is missing the `s3:ListBucket` permission on the bucket resource itself. The AWS S3 SDK requires `s3:ListBucket` at the bucket level to properly distinguish between "object not found" (404) and "access denied" (403) scenarios, causing the Lambda to fail with an `IllegalStateException` wrapping the S3 access denied error.

**Affected file:** `infrastructure/src/main/kotlin/com/frauddetection/infrastructure/InferencePipelineStack.kt`

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the ScoreHandler Lambda attempts to load a transaction batch from the S3 data bucket (e.g., `s3://fraud-detection-data-<suffix>/daily-batches/2026-02-21.json`) THEN the system throws an `IllegalStateException` with a 403 AccessDenied error because the IAM role is not authorized to perform `s3:ListBucket` on the data bucket resource.

1.2 WHEN the ScoreHandler Lambda attempts to read endpoint metadata from the S3 config bucket and the SDK internally requires `s3:ListBucket` THEN the system may also fail with a 403 AccessDenied error for the same missing permission reason on the config bucket.

1.3 WHEN the ScoreHandler Lambda encounters the S3 AccessDenied error THEN the ScoreStage fails, causing the entire FraudDetectionInference Step Functions workflow to enter the `ScoreFailed` catch state after exhausting retries.

### Expected Behavior (Correct)

2.1 WHEN the ScoreHandler Lambda attempts to load a transaction batch from the S3 data bucket THEN the system SHALL successfully retrieve the object without any `s3:ListBucket` authorization errors, because the IAM role policy includes `s3:ListBucket` permission on the data bucket resource ARN (`arn:aws:s3:::<dataBucketName>`).

2.2 WHEN the ScoreHandler Lambda attempts to read endpoint metadata from the S3 config bucket THEN the system SHALL successfully retrieve the object without any `s3:ListBucket` authorization errors, because the IAM role policy includes `s3:ListBucket` permission on the config bucket resource ARN (`arn:aws:s3:::<configBucketName>`).

2.3 WHEN the ScoreHandler Lambda successfully retrieves objects from S3 THEN the ScoreStage SHALL complete normally and the inference workflow SHALL proceed to the StoreStage.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the ScoreHandler Lambda invokes the SageMaker endpoint for scoring THEN the system SHALL CONTINUE TO invoke the endpoint with the correct payload and receive fraud scores between 0.0 and 1.0.

3.2 WHEN the ScoreHandler Lambda writes output data to the workflow S3 bucket THEN the system SHALL CONTINUE TO write successfully using the existing `s3:PutObject` permission.

3.3 WHEN the StoreHandler, AlertHandler, and MonitorHandler Lambdas access their respective S3 buckets and other resources THEN the system SHALL CONTINUE TO operate with their existing permissions unchanged.

3.4 WHEN the ScoreHandler Lambda receives an invalid S3 path or a genuinely missing object THEN the system SHALL CONTINUE TO throw the appropriate `IllegalArgumentException` or `IllegalStateException` with a meaningful error message.
