---
inclusion: auto
---

# Infrastructure Prerequisites

## S3 Buckets Not Managed by CDK

The following S3 buckets are required but NOT created by the CDK stacks. They must be created manually before deployment:

- `fraud-detection-metrics` — Used by MonitorHandler for storing model performance metrics and drift detection baselines. The MonitorHandler reads this from the `METRICS_BUCKET` env var, defaulting to `fraud-detection-metrics` if not set.

Create it with:
```bash
aws s3 mb s3://fraud-detection-metrics --region us-east-1
```

## Environment Variable Mapping

Lambda handlers read specific env var names. The CDK stack must pass the correct names:

| Handler | Env Var | CDK Key |
|---------|---------|---------|
| AlertHandler | `FRAUD_ALERT_TOPIC_ARN` | `FRAUD_ALERT_TOPIC_ARN` |
| MonitorHandler | `MONITORING_ALERT_TOPIC_ARN` | `MONITORING_ALERT_TOPIC_ARN` |
| MonitorHandler | `METRICS_BUCKET` | Not set in CDK (uses default) |
| StoreHandler | `DYNAMODB_TABLE` | `DYNAMODB_TABLE` |

## Known Issues Fixed

1. ScoreHandler was missing `s3:ListBucket` permission on bucket-level ARNs — added in InferencePipelineStack.kt
2. ScoreHandler was sending `application/json` to XGBoost endpoint — changed to `text/csv` format
3. AlertHandler env var was `ALERT_TOPIC_ARN` instead of `FRAUD_ALERT_TOPIC_ARN` — fixed in CDK
4. MonitorHandler env var was `MONITORING_TOPIC_ARN` instead of `MONITORING_ALERT_TOPIC_ARN` — fixed in CDK
5. StoreHandler env var was `FRAUD_SCORES_TABLE` instead of `DYNAMODB_TABLE` — fixed in CDK
