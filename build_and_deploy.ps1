param(
    [Parameter(Mandatory=$false)]
    [string]$awsRegion = $env:AWS_REGION,

    [Parameter(Mandatory=$false)]
    [string]$awsAccountId = $env:AWS_ACCOUNT_ID,

    [Parameter(Mandatory=$false)]
    [string]$ecrRepoName = "mijnlieff",

    [Parameter(Mandatory=$false)]
    [switch]$noCache = $false
)

# Set defaults if not provided
if (-not $awsRegion) { $awsRegion = "eu-west-2" }

# Validate required parameters
if (-not $awsAccountId) {
    # Try to get from AWS CLI
    $awsAccountId = aws sts get-caller-identity --query Account --output text 2>$null
    if (-not $awsAccountId) {
        Write-Error "AWS Account ID not found. Set AWS_ACCOUNT_ID environment variable or configure AWS CLI."
        exit 1
    }
    Write-Host "Using AWS Account ID from CLI: $awsAccountId" -ForegroundColor Cyan
}

# Build ECR repository URL
$ecrRepository = "${awsAccountId}.dkr.ecr.${awsRegion}.amazonaws.com/${ecrRepoName}"

# Ensure AWS CLI is available
if (!(Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Error "AWS CLI is not installed or not in PATH"
    exit 1
}

# Ensure Docker is running
try {
    docker info > $null 2>&1
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Error "Docker is not running or not accessible"
    exit 1
}

Write-Host "Starting Docker build and push to ECR..." -ForegroundColor Green
Write-Host "  Region: $awsRegion" -ForegroundColor Cyan
Write-Host "  Repository: $ecrRepository" -ForegroundColor Cyan

# Get ECR login token and login to Docker
Write-Host "Logging into ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $awsRegion | docker login --username AWS --password-stdin "${awsAccountId}.dkr.ecr.${awsRegion}.amazonaws.com"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to login to ECR"
    exit 1
}

# Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
if ($noCache) {
    Write-Host "  (with --no-cache flag)"
    docker build --no-cache -t "${ecrRepository}:latest" .
} else {
    docker build -t "${ecrRepository}:latest" .
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed"
    exit 1
}

# Push the image to ECR
Write-Host "Pushing image to ECR..." -ForegroundColor Yellow
docker push "${ecrRepository}:latest"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to push Docker image to ECR"
    exit 1
}

Write-Host ""
Write-Host "Deployment completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  - Frontend is deployed via GitHub Actions to GitHub Pages"
Write-Host "  - If Lambda URL changed, update LAMBDA_FUNCTION_URL in GitHub repo variables"
