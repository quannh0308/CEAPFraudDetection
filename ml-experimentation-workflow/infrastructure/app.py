#!/usr/bin/env python3
"""CDK app entry point for SageMaker Studio fraud detection infrastructure."""

import aws_cdk as cdk

from sagemaker_studio_stack import SageMakerStudioStack

app = cdk.App()

SageMakerStudioStack(
    app,
    "FraudDetectionSageMakerStudio",
    description="SageMaker Studio environment for fraud detection ML experimentation",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
)

app.synth()
