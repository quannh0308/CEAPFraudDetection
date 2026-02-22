"""
ML Experimentation Workflow — Core modules for fraud detection experimentation.

This package provides tools for running, tracking, and promoting ML experiments
in SageMaker Studio. It integrates with the production fraud-detection-ml-pipeline
via Parameter Store, S3 configuration files, and Step Functions.

Modules:
    experiment_tracking:    ExperimentTracker for SageMaker Experiments integration
    hyperparameter_tuning:  HyperparameterTuner with grid, random, and Bayesian search
    algorithm_comparison:   AlgorithmComparator for side-by-side model evaluation
    feature_engineering:    FeatureEngineer for derived features and feature selection
    model_evaluation:       ModelEvaluator for metrics, visualizations, and threshold checks
    production_integration: ProductionIntegrator for promoting configs to production
    ab_testing:             ABTestingManager for challenger endpoint deployment and comparison

Example:
    from src.experiment_tracking import ExperimentTracker
    from src.hyperparameter_tuning import HyperparameterTuner
    from src.model_evaluation import ModelEvaluator
    from src.production_integration import ProductionIntegrator
"""
