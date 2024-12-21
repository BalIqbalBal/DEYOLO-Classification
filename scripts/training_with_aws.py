from sagemaker.estimator import Estimator

role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"

container_image_uri = '108830828338.dkr.ecr.ap-southeast-1.amazonaws.com/pain-identification:latest'

s3_dataset_path = 's3://pain-identification-datasets/formatted_datasets//'
output_path = 's3://pain-identification-result/'

# Define the SageMaker Estimator
estimator = Estimator(
    image_uri=container_image_uri,
    role=role,
    instance_type='ml.p2.xlarge',  # Change as needed (make sure its support pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime)
    instance_count=1,
    output_path=output_path,  # This is the S3 location for saving model artifacts
    hyperparameters={
        'model': 'deyolo',  # Required argument for train_launcher.py
        'project-name': 'my_project',  # Optional, train_launcher will generate if not provided
        'learning-rate': 0.001,  # Matches '--learning-rate'
        'num-epochs': 100,  # Matches '--num-epochs'
        'batch-size': 32,  # Matches '--batch-size'
        'data-dir': '/opt/ml/input/data/training',  # Default SageMaker location for training data
        'checkpoint': '/opt/ml/model',  # Save checkpoints to model directory (default checkpoint for estimato)
    }
)

# Start the training job with the S3 dataset path
estimator.fit(inputs={'training': s3_dataset_path})
