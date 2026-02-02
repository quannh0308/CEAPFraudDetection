# Fraud Detection ML Pipeline

## Overview

This project implements a fraud detection system using the CEAP (Customer Engagement & Action Platform) workflow orchestration framework.

## Architecture

- **Training Pipeline** (Weekly): Train fraud detection model, deploy to SageMaker
- **Inference Pipeline** (Daily): Score transactions, alert on high-risk cases

## Setup

1. Initialize Git submodule for CEAP platform
2. Build project with Gradle
3. Deploy to AWS

## Documentation

See `.kiro/specs/fraud-detection-ml-pipeline/` for complete specifications.

## Technology Stack

- Kotlin, Python, AWS SageMaker, Step Functions, Lambda, Glue, DynamoDB, SNS
