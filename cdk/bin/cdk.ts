#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { MijnlieffStack, MijnlieffEcrStack } from "../lib/cdk-stack";

const app = new cdk.App();

// Use CDK CLI environment (from AWS profile) or explicit env vars
const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT || process.env.AWS_ACCOUNT_ID,
  region: process.env.CDK_DEFAULT_REGION || process.env.AWS_REGION || "eu-west-2",
};

if (!env.account) {
  throw new Error(
    "AWS account not found. Either configure AWS CLI profile or set AWS_ACCOUNT_ID environment variable."
  );
}

// ECR repository for Lambda container images
new MijnlieffEcrStack(app, "MijnlieffEcrStack", { env });

// Lambda backend stack
new MijnlieffStack(app, "MijnlieffStack", { env });

app.synth();
