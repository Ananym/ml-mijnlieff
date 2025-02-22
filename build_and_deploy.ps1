param(
    [Parameter(Mandatory=$false)]
    [string]$awsRegion = "us-east-1",
    
    [Parameter(Mandatory=$false)]
    [string]$ecrRepository = "123456789012.dkr.ecr.us-east-1.amazonaws.com/mijnlieff",
    
    [Parameter(Mandatory=$false)]
    [string]$s3Bucket = "thestope-public",
    
    [Parameter(Mandatory=$false)]
    [string]$s3Path = "prod",

    [Parameter(Mandatory=$false)]
    [string]$apiEndpoint = "https://api.mijnlieff.com"
)

# Ensure AWS CLI is available
if (!(Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Error "AWS CLI is not installed or not in PATH"
    exit 1
}

# Ensure Docker is running
try {
    docker info > $null
} catch {
    Write-Error "Docker is not running or not accessible"
    exit 1
}

Write-Host "Starting deployment process..." -ForegroundColor Green

# Step 1: Build and push Docker image to ECR
Write-Host "Building and pushing Docker image..." -ForegroundColor Yellow

# Get ECR login token and login to Docker
Write-Host "Logging into ECR..."
aws ecr get-login-password --region $awsRegion | docker login --username AWS --password-stdin "$ecrRepository"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to login to ECR"
    exit 1
}

# Build the Docker image
Write-Host "Building Docker image..."
docker build -t "${ecrRepository}:latest" .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed"
    exit 1
}

# Push the image to ECR
Write-Host "Pushing image to ECR..."
docker push "${ecrRepository}:latest"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to push Docker image to ECR"
    exit 1
}

# Step 2: Build and deploy Vue frontend
Write-Host "Building and deploying Vue frontend..." -ForegroundColor Yellow

# Navigate to Vue frontend directory
Set-Location -Path "vue-frontend"

# Install dependencies
Write-Host "Installing npm dependencies..."
npm install

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install npm dependencies"
    exit 1
}

# Build Vue app with the provided API endpoint
Write-Host "Building Vue app..."
$env:VITE_API_ENDPOINT = $apiEndpoint
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Error "Vue build failed"
    exit 1
}

# Upload to S3 with appropriate cache settings
Write-Host "Uploading to S3..."

# Upload all files except index.html with long cache
Write-Host "Uploading static assets..."
aws s3 sync dist/ "s3://${s3Bucket}/${s3Path}" `
    --exclude "index.html" `
    --cache-control "max-age=31536000" `
    --delete

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upload static assets to S3"
    exit 1
}

# Upload index.html with no-cache setting
Write-Host "Uploading index.html..."
aws s3 cp dist/index.html "s3://${s3Bucket}/${s3Path}/index.html" `
    --cache-control "no-cache"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upload index.html to S3"
    exit 1
}

Write-Host "Deployment completed successfully!" -ForegroundColor Green
Write-Host "Frontend URL: https://${s3Bucket}/${s3Path}/index.html" 