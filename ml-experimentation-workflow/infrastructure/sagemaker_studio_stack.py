"""CDK Stack for SageMaker Studio fraud detection experimentation environment.

Creates:
- SageMaker Studio Domain with IAM authentication
- IAM execution role with permissions for S3, SageMaker, Parameter Store,
  Step Functions, and CloudWatch Logs
- S3 bucket for fraud-detection-config with lifecycle policies
"""

from aws_cdk import (
    Stack,
    CfnOutput,
    RemovalPolicy,
    Duration,
    Fn,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sagemaker as sagemaker,
    aws_ec2 as ec2,
)
from constructs import Construct


class SageMakerStudioStack(Stack):
    """CDK stack for SageMaker Studio fraud detection experimentation."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- IAM Execution Role ---
        self.execution_role = self._create_execution_role()

        # --- S3 Buckets ---
        self.config_bucket = self._create_config_bucket()

        # --- VPC (default) ---
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # --- SageMaker Studio Domain ---
        self.domain = self._create_studio_domain(vpc)

        # --- Outputs ---
        CfnOutput(
            self,
            "SageMakerStudioDomainId",
            value=self.domain.attr_domain_id,
            description="SageMaker Studio Domain ID",
        )

        CfnOutput(
            self,
            "SageMakerStudioUrl",
            value=Fn.sub(
                "https://${AWS::Region}.console.aws.amazon.com/sagemaker/home"
                "?region=${AWS::Region}#/studio/${DomainId}",
                {"DomainId": self.domain.attr_domain_id},
            ),
            description="SageMaker Studio console URL",
        )

        CfnOutput(
            self,
            "ExecutionRoleArn",
            value=self.execution_role.role_arn,
            description="SageMaker execution role ARN",
        )

        CfnOutput(
            self,
            "ConfigBucketName",
            value=self.config_bucket.bucket_name,
            description="Fraud detection config S3 bucket name",
        )

    # ------------------------------------------------------------------
    # IAM
    # ------------------------------------------------------------------
    def _create_execution_role(self) -> iam.Role:
        """Create SageMaker execution role with required permissions."""
        role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for SageMaker Studio fraud detection experimentation",
        )

        # S3 read/write for fraud-detection buckets
        role.add_to_policy(
            iam.PolicyStatement(
                sid="S3FraudDetectionAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                ],
                resources=[
                    "arn:aws:s3:::fraud-detection-data",
                    "arn:aws:s3:::fraud-detection-data/*",
                    "arn:aws:s3:::fraud-detection-config",
                    "arn:aws:s3:::fraud-detection-config/*",
                    "arn:aws:s3:::fraud-detection-models",
                    "arn:aws:s3:::fraud-detection-models/*",
                ],
            )
        )

        # SageMaker training jobs and endpoints
        role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerTrainingAndEndpoints",
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:CreateModel",
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:CreateEndpoint",
                    "sagemaker:DescribeEndpoint",
                    "sagemaker:DeleteEndpoint",
                    "sagemaker:DeleteEndpointConfig",
                    "sagemaker:DeleteModel",
                    "sagemaker:InvokeEndpoint",
                    "sagemaker:CreateHyperParameterTuningJob",
                    "sagemaker:DescribeHyperParameterTuningJob",
                    "sagemaker:StopHyperParameterTuningJob",
                    "sagemaker:ListTrainingJobsForHyperParameterTuningJob",
                    "sagemaker:CreateExperiment",
                    "sagemaker:CreateTrial",
                    "sagemaker:CreateTrialComponent",
                    "sagemaker:UpdateTrialComponent",
                    "sagemaker:DescribeExperiment",
                    "sagemaker:DescribeTrial",
                    "sagemaker:DescribeTrialComponent",
                    "sagemaker:Search",
                    "sagemaker:AddTags",
                ],
                resources=["*"],
            )
        )

        # Parameter Store read/write
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ParameterStoreAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                    "ssm:GetParametersByPath",
                    "ssm:PutParameter",
                    "ssm:DeleteParameter",
                ],
                resources=[
                    Fn.sub(
                        "arn:aws:ssm:${AWS::Region}:${AWS::AccountId}"
                        ":parameter/fraud-detection/*"
                    )
                ],
            )
        )

        # Step Functions execution
        role.add_to_policy(
            iam.PolicyStatement(
                sid="StepFunctionsAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "states:StartExecution",
                    "states:DescribeExecution",
                    "states:ListExecutions",
                ],
                resources=[
                    Fn.sub(
                        "arn:aws:states:${AWS::Region}:${AWS::AccountId}"
                        ":stateMachine:fraud-detection-*"
                    )
                ],
            )
        )

        # CloudWatch Logs
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogsAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                    "logs:GetLogEvents",
                ],
                resources=[
                    Fn.sub(
                        "arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/sagemaker/*"
                    )
                ],
            )
        )

        # ECR access for SageMaker built-in images
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                resources=["*"],
            )
        )

        return role

    # ------------------------------------------------------------------
    # S3
    # ------------------------------------------------------------------
    def _create_config_bucket(self) -> s3.Bucket:
        """Create fraud-detection-config bucket with lifecycle policies."""
        bucket = s3.Bucket(
            self,
            "FraudDetectionConfigBucket",
            bucket_name="fraud-detection-config",
            removal_policy=RemovalPolicy.RETAIN,
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                # Transition old archive backups to Glacier after 30 days
                s3.LifecycleRule(
                    id="ArchiveBackupLifecycle",
                    prefix="archive/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(30),
                        )
                    ],
                    expiration=Duration.days(365),
                ),
                # Expire parameter-store-backups after 90 days
                s3.LifecycleRule(
                    id="ParameterStoreBackupLifecycle",
                    prefix="parameter-store-backups/",
                    expiration=Duration.days(90),
                ),
            ],
        )
        return bucket

    # ------------------------------------------------------------------
    # SageMaker Studio
    # ------------------------------------------------------------------
    def _create_studio_domain(self, vpc: ec2.IVpc) -> sagemaker.CfnDomain:
        """Create SageMaker Studio Domain with IAM authentication."""
        subnet_ids = [subnet.subnet_id for subnet in vpc.public_subnets]

        domain = sagemaker.CfnDomain(
            self,
            "FraudDetectionStudioDomain",
            auth_mode="IAM",
            domain_name="fraud-detection-experimentation",
            vpc_id=vpc.vpc_id,
            subnet_ids=subnet_ids,
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=self.execution_role.role_arn,
                jupyter_server_app_settings=sagemaker.CfnDomain.JupyterServerAppSettingsProperty(
                    default_resource_spec=sagemaker.CfnDomain.ResourceSpecProperty(
                        instance_type="system",
                        sage_maker_image_arn=Fn.sub(
                            "arn:aws:sagemaker:${AWS::Region}:081325390199:image/datascience-1.0"
                        ),
                    )
                ),
                kernel_gateway_app_settings=sagemaker.CfnDomain.KernelGatewayAppSettingsProperty(
                    default_resource_spec=sagemaker.CfnDomain.ResourceSpecProperty(
                        instance_type="ml.t3.medium",
                        sage_maker_image_arn=Fn.sub(
                            "arn:aws:sagemaker:${AWS::Region}:081325390199:image/datascience-1.0"
                        ),
                    )
                ),
            ),
        )
        return domain
