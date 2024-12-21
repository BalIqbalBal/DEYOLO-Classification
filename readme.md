
# Docker Image for SageMaker Training

This repository provides a script to build a Docker container, push it to Amazon Elastic Container Registry (ECR), and use it in Amazon SageMaker to run training jobs.

This setup enables you to deploy custom Docker containers for training on Amazon SageMaker, leveraging AWS infrastructure for scalable machine learning training.

## Prerequisites

Before running the script, make sure you have the following installed:

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **AWS CLI**: [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- **SageMaker Python SDK**: Install via pip:
  ```bash
  pip install sagemaker
  ```
- **AWS Credentials**: Set up AWS credentials via `aws configure` or ensure the environment has access to the AWS account.

## Steps

### 1. Build Docker Image and Push to ECR

The script automates the process of:

- Creating an ECR repository if it doesn't already exist.
- Building the Docker image.
- Tagging the Docker image with the ECR URI.
- Logging in to Amazon ECR.
- Pushing the Docker image to ECR.

Follow these steps:

1. **Configure the variables** in the script:
   - Specify the AWS region (e.g., `us-west-2`).
   - Define the ECR repository name and image tag.
   this varaiable is in Define variables section.

2. **Run the script** to build and push the Docker image to your ECR repository.

### 2. Train Model on SageMaker Using the Custom Docker Image

Once the Docker image is pushed to ECR, you can use it in SageMaker for training. The following steps are involved:

1. **Define the Docker image URI** from ECR.
2. **Set up S3 paths** for input data and model output.
3. **Create a SageMaker Estimator** using the custom image URI.
4. **Start the training job** by referencing the S3 dataset.

## Troubleshooting

1. **Access Denied Errors**: Ensure that your IAM role has the correct permissions (`AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, `AmazonECRFullAccess`).
2. **Docker Issues**: Ensure Docker is installed and running correctly.
3. **SageMaker Role**: Ensure the SageMaker role has the necessary permissions to access the S3 input dataset and output bucket.