# CEAP Architecture Overview

## Introduction

CEAP (Customer Engagement & Action Platform) is a state-of-the-art architecture for data processing, ML workflows, and reactive event-driven systems. This document provides a comprehensive overview of the CEAP architecture, its core principles, and how the fraud detection system demonstrates these patterns.

## Table of Contents

1. [CEAP Core Principles](#ceap-core-principles)
2. [Architecture Patterns](#architecture-patterns)
3. [Component Model](#component-model)
4. [Data Flow Patterns](#data-flow-patterns)
5. [Fraud Detection System Architecture](#fraud-detection-system-architecture)
6. [Component Catalog](#component-catalog)

---

## CEAP Core Principles

### 1. S3-Based Orchestration

**Principle**: Use S3 as the primary data exchange mechanism between workflow stages.

**Benefits**:
- **Loose Coupling**: Stages don't directly call each other
- **Debuggability**: All intermediate data is persisted and inspectable
- **Replayability**: Can restart from any stage using S3 data
- **Auditability**: Complete data lineage in S3
- **Scalability**: S3 handles any data volume

**Pattern**:
```
Stage 1 → S3 → Stage 2 → S3 → Stage 3 → S3 → Stage 4
```

### 2. Convention-Based Paths

**Principle**: Use consistent, predictable S3 path conventions for all workflow data.

**Convention**:
```
s3://workflow-bucket/executions/{executionId}/{stageName}/output.json
```

**Benefits**:
- No configuration needed for path resolution
- Easy to locate data for any execution
- Supports parallel executions without conflicts
- Simplifies debugging and troubleshooting

### 3. WorkflowLambdaHandler Pattern

**Principle**: All Lambda handlers extend a base class that provides S3 orchestration.

**Base Class Responsibilities**:
- Read input from S3 (or initialData for first stage)
- Execute stage-specific logic (abstract method)
- Write output to S3
- Handle errors and return StageResult

**Benefits**:
- Consistent error handling across all stages
- Reduced boilerplate code
- Standardized logging and monitoring
- Easy to add new stages


### 4. Workflow Types

**Standard Workflow**: For long-running operations (hours)
- Full execution history
- Supports human approval steps
- Suitable for training pipelines
- Cost: ~$25 per million state transitions

**Express Workflow**: For short-duration operations (minutes)
- High throughput
- Synchronous or asynchronous execution
- Suitable for inference pipelines
- Cost: ~$1 per million state transitions

### 5. Mixed Compute Patterns

**Principle**: Use the right compute for each task.

**Lambda**: For orchestration, lightweight processing, API calls
**Glue**: For large-scale data transformations (PySpark)
**SageMaker**: For ML training and inference
**Step Functions**: For workflow orchestration

---

## Architecture Patterns

### Pattern 1: Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Step Functions Workflow (Standard or Express)              │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ Stage 1  │ →  │ Stage 2  │ →  │ Stage 3  │             │
│  │ (Lambda) │    │ (Glue)   │    │ (Lambda) │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│       ↓               ↓               ↓                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │         S3 Workflow Bucket                         │    │
│  │  executions/{id}/Stage1/output.json               │    │
│  │  executions/{id}/Stage2/output.json               │    │
│  │  executions/{id}/Stage3/output.json               │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Key Characteristics**:
- Each stage reads from previous stage's S3 output
- Each stage writes to its own S3 location
- Stages can be Lambda, Glue, or SageMaker
- Step Functions orchestrates execution order


### Pattern 2: ML Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Training Pipeline (Standard Workflow - Weekly)             │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐ │
│  │ DataPrep │ →  │  Train   │ →  │ Evaluate │ →  │Deploy│ │
│  │  (Glue)  │    │(SageMaker)│    │ (Lambda) │    │(Lmbd)│ │
│  └──────────┘    └──────────┘    └──────────┘    └──────┘ │
│       ↓               ↓               ↓               ↓     │
│  ┌────────────────────────────────────────────────────┐    │
│  │         S3 Orchestration                           │    │
│  │  • Data paths and record counts                    │    │
│  │  • Training job name and model artifact path       │    │
│  │  • Evaluation metrics (accuracy, precision, etc)   │    │
│  │  • Endpoint name and deployment metadata           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  External Services:                                         │
│  • SageMaker: Model training (2-4 hours)                   │
│  • S3: Model artifacts, prepared data                      │
│  • CloudWatch: Logs and metrics                            │
└─────────────────────────────────────────────────────────────┘
```

**Key Characteristics**:
- Long-running (2-4 hours for training)
- Uses Standard Workflow for full history
- Integrates SageMaker for ML operations
- Validation gates (accuracy >= 0.90)

### Pattern 3: ML Inference Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Inference Pipeline (Express Workflow - Daily)              │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐ │
│  │  Score   │ →  │  Store   │ →  │  Alert   │ →  │Monitor│ │
│  │ (Lambda) │    │ (Lambda) │    │ (Lambda) │    │(Lmbd) │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────┘ │
│       ↓               ↓               ↓               ↓     │
│  ┌────────────────────────────────────────────────────┐    │
│  │         S3 Orchestration                           │    │
│  │  • Scored transactions with fraud scores           │    │
│  │  • Storage summary (success/error counts)          │    │
│  │  • Alert summary (high-risk count)                 │    │
│  │  • Monitoring metrics (drift detection)            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  External Services:                                         │
│  • SageMaker Runtime: Endpoint invocation                  │
│  • DynamoDB: Persistent storage                            │
│  • SNS: Alerts and notifications                           │
└─────────────────────────────────────────────────────────────┘
```

**Key Characteristics**:
- Fast execution (5-30 minutes)
- Uses Express Workflow for high throughput
- Processes batches of transactions
- Real-time alerting on high-risk cases


### Pattern 4: Experimentation + Production Integration

```
┌─────────────────────────────────────────────────────────────┐
│  Experimentation Environment (SageMaker Studio)             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Hyperparameter│  │  Algorithm   │  │   Feature    │     │
│  │    Tuning     │  │  Comparison  │  │ Engineering  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                  ↓                  ↓             │
│  ┌────────────────────────────────────────────────────┐    │
│  │      Experiment Tracker (SageMaker Experiments)    │    │
│  │  • Hyperparameters, metrics, artifacts             │    │
│  │  • Code versions, dataset versions                 │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │      Production Integration Layer                  │    │
│  │  • Parameter Store (hyperparameters)               │    │
│  │  • S3 Config Files (production-model-config.yaml)  │    │
│  │  • Step Functions Trigger (retrain pipeline)       │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Production Pipeline (Automated)                            │
│                                                              │
│  Reads hyperparameters from Parameter Store                 │
│  Trains models with experiment-optimized settings           │
│  Deploys to production endpoints                            │
└─────────────────────────────────────────────────────────────┘
```

**Key Characteristics**:
- Experimentation and production are separate but integrated
- Winning configurations promoted via Parameter Store
- Automated production deployment with validated configs
- Full traceability from experiment to production

---

## Component Model

### Component Structure

Every CEAP component follows this structure:

```
┌─────────────────────────────────────────────────────────┐
│  Component Name (e.g., TrainHandler)                    │
├─────────────────────────────────────────────────────────┤
│  INPUT                                                   │
│  • Source: S3 or initialData                            │
│  • Format: JSON                                          │
│  • Schema: Defined by previous stage                    │
│  • Example: { "trainDataPath": "s3://...", ... }        │
├─────────────────────────────────────────────────────────┤
│  PROCESSING                                              │
│  • Business logic (abstract processData method)         │
│  • External service calls (SageMaker, DynamoDB, etc)    │
│  • Error handling and validation                        │
│  • Logging and monitoring                               │
├─────────────────────────────────────────────────────────┤
│  OUTPUT                                                  │
│  • Destination: S3                                       │
│  • Format: JSON                                          │
│  • Schema: Defined for next stage                       │
│  • Example: { "trainingJobName": "...", ... }           │
├─────────────────────────────────────────────────────────┤
│  SIDE EFFECTS                                            │
│  • External service state changes                       │
│  • Example: SageMaker training job created              │
│  • Example: DynamoDB records written                    │
│  • Example: SNS notifications sent                      │
└─────────────────────────────────────────────────────────┘
```

