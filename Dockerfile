# Base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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


