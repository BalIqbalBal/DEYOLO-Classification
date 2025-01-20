# train_launcher.py
import os
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Training launcher for different models')
    
    # Training hyperparameters
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deyolorgb', 'deyolothermal', 'vggfacergb', 'vggfacethermal', 
                                 'resnetrgb', 'resnetthermal', 
                                 'shufflenetrgb', 'shufflenetthermal', 
                                 'mobilenetrgb', 'mobilenetthermal',
                                 'vgggfacemm', 'resnetmm', 'deyolomm'],
                        help='Model type to train')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=15,
                        help='Batch size')
    parser.add_argument('--lr-decay-step', type=int, default=10,
                        help="Step size for learning rate decay (in epochs).")
    parser.add_argument('--lr-decay-gamma', type=float, default=0.1,
                        help="Factor by which to decay the learning rate.")
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help="Number of epochs to wait before stopping if validation accuracy doesn't improve.")
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help="Weight decay (L2 regularization) strength.")
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help="Dropout rate for regularization.")
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'focal'], help="Loss function to use.")
    
    # SageMaker parameters
    parser.add_argument('--project-name', type=str, default=None,
                        help='Project name for logging')
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training',
                        help='Base data directory (default SageMaker location)')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model',
                        help='Directory to save the model (default SageMaker location)')
    parser.add_argument('--checkpoint', type=str,
                        help='Checkpoint directory for saving models')
    
    parser.add_argument('--layercam-during-training', action='store_true', help="Compute LayerCAM after each epoch during training.")
    
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
    
    if args.model == 'deyolomm':
        from train_base_modal_fusion import train_multimodal_model
        args.model_type = 'deyolomm'
        train_multimodal_model(args)

    elif args.model == 'resnetmm':
        from train_base_modal_fusion import train_multimodal_model
        args.model_type = 'resnetmm'
        train_multimodal_model(args)

    elif args.model == 'vggfacemm':
        from train_base_modal_fusion import train_multimodal_model
        args.model_type = 'vggfacemm'
        train_multimodal_model(args)
        
    elif args.model == 'vggfacergb':
        from train_base import train_model
        args.model_type = 'vgg'
        train_model(args, type_model='rgb')
    
    elif args.model == 'vggfacethermal':
        from train_base import train_model
        args.model_type = 'vgg'
        train_model(args, type_model='thermal')
    
    elif args.model == 'resnetrgb':
        from train_base import train_model
        args.model_type = 'resnet'
        train_model(args, type_model='rgb')
    
    elif args.model == 'resnetthermal':
        from train_base import train_model
        args.model_type = 'resnet'
        train_model(args, type_model='thermal')
    
    elif args.model == 'shufflenetrgb':
        from train_base import train_model
        args.model_type = 'shufflenet'
        train_model(args, type_model='rgb')
    
    elif args.model == 'shufflenetthermal':
        from train_base import train_model
        args.model_type = 'shufflenet'
        train_model(args, type_model='thermal')
    
    elif args.model == 'mobilenetrgb':
        from train_base import train_model
        args.model_type = 'mobilenet'
        train_model(args, type_model='rgb')
    
    elif args.model == 'mobilenetthermal':
        from train_base import train_model
        args.model_type = 'mobilenet'
        train_model(args, type_model='thermal')
    
    print(f"Training completed for {args.model}")

if __name__ == '__main__':
    main()