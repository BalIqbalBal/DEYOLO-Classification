# Base image
FROM 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

# Set the working directory to /opt/ml/code (standard for SageMaker)
WORKDIR /opt/ml/code

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the working directory
COPY src/ .

# Set the entry point for the container to the training script
ENV SAGEMAKER_PROGRAM train_launcher.py


