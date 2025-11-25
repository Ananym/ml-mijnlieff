import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as lambda from "aws-cdk-lib/aws-lambda";

// Stack for the Mijnlieff ECR repository (backend container)
export class MijnlieffEcrStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create ECR repository for Mijnlieff backend
    const repository = new ecr.Repository(this, "MijnlieffRepo", {
      repositoryName: "mijnlieff",
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // Output the repository URI
    new cdk.CfnOutput(this, "RepositoryUri", {
      value: repository.repositoryUri,
    });
  }
}

// Stack for the Mijnlieff Lambda backend
export class MijnlieffStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Reference existing ECR repository
    const repository = ecr.Repository.fromRepositoryName(
      this,
      "MijnlieffRepo",
      "mijnlieff"
    );

    // Create Lambda function from container image
    const lambdaFunction = new lambda.DockerImageFunction(
      this,
      "ContainerFunction",
      {
        code: lambda.DockerImageCode.fromEcr(repository, {
          tagOrDigest: "latest",
        }),
        memorySize: 1024,
        timeout: cdk.Duration.seconds(30),
        reservedConcurrentExecutions: 1,
      }
    );

    // Add function URL to Lambda with CORS for GitHub Pages
    const functionUrl = lambdaFunction.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedOrigins: ["*"], // Allow any origin (GitHub Pages, local dev, etc.)
        allowedMethods: [lambda.HttpMethod.POST, lambda.HttpMethod.GET],
        allowedHeaders: ["*"],
      },
    });

    // Output the important values
    new cdk.CfnOutput(this, "EcrRepositoryUri", {
      value: repository.repositoryUri,
    });
    new cdk.CfnOutput(this, "LambdaFunctionUrl", { value: functionUrl.url });
  }
}
