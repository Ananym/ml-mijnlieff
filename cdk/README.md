# Mijnlieff CDK Infrastructure

AWS CDK infrastructure for the Mijnlieff backend API.

## Architecture

The frontend is hosted on GitHub Pages. This CDK project deploys only the backend:

### MijnlieffEcrStack
- ECR repository: `mijnlieff`
- Stores Docker images for the Lambda function

### MijnlieffStack
- Lambda function (from Docker container in ECR)
- Function URL with CORS enabled for any origin

## Deployment

### Prerequisites
- Node.js and npm
- AWS CLI configured
- Docker (for building container images)

### Setup
```powershell
cd cdk
npm install
npm run build
```

### Deploy
```powershell
npx cdk deploy --all
```

Or individually:
```powershell
npx cdk deploy MijnlieffEcrStack
npx cdk deploy MijnlieffStack
```

### After Deployment
1. Push your Docker image to ECR (see `build_and_deploy.ps1`)
2. The Lambda Function URL will be output - use this as `VITE_API_ENDPOINT` when building the frontend

## Useful Commands

* `npm run build`   - Compile TypeScript
* `npx cdk deploy`  - Deploy stacks
* `npx cdk diff`    - Compare with deployed state
* `npx cdk synth`   - Emit CloudFormation template
* `npx cdk destroy` - Remove stacks
* `npx cdk doctor`  - Check for potential issues
