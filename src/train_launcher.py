# train_launcher.py
import os
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Training launcher for different models')
    
    parser.add_argument('--model', type=str, required=True, choices=['deyolo', 'vggfacergb', 'vggfacethermal'],
                      help='Model type to train (deyolo or vggface)')
    parser.add_argument('--project-name', type=str, default=None,
                      help='Project name for logging')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=15,
                      help='Batch size')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training',
                      help='Base data directory (default SageMaker location)')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model',
                      help='Directory to save the model (default SageMaker location)')
    parser.add_argument('--checkpoint', type=str)
    return parser.parse_args()

def generate_project_name(base_name):
    """Generate a unique project name with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}"

def main():
    args = parse_args()
    
    # Generate project name if not provided
    if args.project_name is None:
        args.project_name = generate_project_name(args.model)
    
    print(f"Starting training for {args.model} model")
    print(f"Project name: {args.project_name}")
    
    if args.model == 'deyolo':
        from train_DEYOLOCLASS import trainDEYOLOCLASS
        trainDEYOLOCLASS(args)
        
    elif args.model == 'vggfacergb':
        from train_vggface import trainVGGFace      
        trainVGGFace(args, type_model='rgb')
    
    elif args.model == 'vggfacethermal':
        from train_vggface import trainVGGFace      
        trainVGGFace(args, type_model='thermal')
    
    print(f"Training completed for {args.model}")

if __name__ == '__main__':
    main()