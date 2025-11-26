#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { MijnlieffStack, MijnlieffEcrStack } from "../lib/cdk-stack";

const app = new cdk.App();

// Let CDK resolve account/region from AWS CLI profile
// Override with env vars if explicitly set
const env = {
  account: process.env.AWS_ACCOUNT_ID || process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.AWS_REGION || process.env.CDK_DEFAULT_REGION || "eu-west-2",
};

// ECR repository for Lambda container images
new MijnlieffEcrStack(app, "MijnlieffEcrStack", { env });

// Lambda backend stack
new MijnlieffStack(app, "MijnlieffStack", { env });

app.synth();
