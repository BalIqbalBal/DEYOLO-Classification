from sagemaker.estimator import Estimator

role = "arn:aws:iam::108830828338:role/SageMakerFullAccess"

container_image_uri = '108830828338.dkr.ecr.ap-southeast-1.amazonaws.com/pain-identification:latest'

s3_dataset_path = 's3://pain-identification-datasets/formatted_datasets/'
output_path = 's3://pain-identification-result/'

# Define the SageMaker Estimator
estimator = Estimator(
    image_uri=container_image_uri,
    role=role,
    instance_type='ml.p3.2xlarge', 
    instance_count=1,
    output_path=output_path,  # This is the S3 location for saving model artifacts
    hyperparameters={
        'model': 'deyolo',  # Required argument for train_launcher.py
        'project-name': 'deyolotiny',  # Optional, train_launcher will generate if not provided
        'learning-rate': 0.000001,  # Matches '--learning-rate'
        'num-epochs': 100,  # Matches '--num-epochs'
        'batch-size': 16,  # Matches '--batch-size'
        'data-dir': '/opt/ml/input/data/training',  # Default SageMaker location for training data
        'checkpoint': output_path,  # Save checkpoints to model directory (default checkpoint for estimato),
        'model-dir': '/opt/ml/model'
    }
)

# Start the training job with the S3 dataset path
estimator.fit(inputs={'training': s3_dataset_path})