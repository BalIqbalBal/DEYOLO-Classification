# Define variables
$AWS_REGION = "ap-southeast-1"  # E.g., "us-west-2"
$ECR_REPO_NAME = "pain-identification"
$IMAGE_TAG = "latest"  # You can specify a different tag like "v1.0"
$ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
$ECR_URI = "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION

# Step 1: Create ECR repository (if it doesn't exist)
try {
    aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION
} catch {
    aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION
}

# Step 2: Build the Docker image (using .. to refer to the parent directory)
docker build -t ${ECR_REPO_NAME}:${IMAGE_TAG} ..

# Step 3: Tag the Docker image with the ECR URI
docker tag ${ECR_REPO_NAME}:${IMAGE_TAG} ${ECR_URI}

# Step 4: Login to Amazon ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}

# Step 5: Push the Docker image to ECR
docker push ${ECR_URI}

echo "Docker image pushed to ${ECR_URI}"
