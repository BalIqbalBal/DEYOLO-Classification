import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.datasets import getSingleImageDataloader
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

import random

from utils.loss import FocalLoss

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=15)
    parser.add_argument('--model-type', type=str, default='vgg', choices=['vgg', 'resnet', 'shufflenet', 'mobilenet'])
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'focal'], help="Loss function to use.")
    parser.add_argument('--lr-decay-step', type=int, default=10, help="Step size for learning rate decay (in epochs).")
    parser.add_argument('--lr-decay-gamma', type=float, default=0.1, help="Factor by which to decay the learning rate.")
    parser.add_argument('--early-stopping-patience', type=int, default=5, help="Number of epochs to wait before stopping if validation accuracy doesn't improve.")
    parser.add_argument('--weight-decay', type=float, default=1e-5, help="Weight decay (L2 regularization) strength.")
    parser.add_argument('--dropout-rate', type=float, default=0.5, help="Dropout rate for regularization.")
    
    # SageMaker parameters
    parser.add_argument('--project-name', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))

    # LayerCAM parameters
    parser.add_argument('--layercam-during-training', action='store_true', help="Compute LayerCAM after each epoch during training.")

    return parser.parse_args()

# Define model loading functions with dropout
def get_vggface_model(num_classes, pretrained=True, freeze=True, dropout_rate=0.5):
    model = models.vgg16(pretrained=pretrained)
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),  # Add dropout
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),  # Add dropout
        nn.Linear(4096, num_classes),
    )
    return model

def get_resnet_model(num_classes, pretrained=False, dropout_rate=0.5):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),  # Add dropout
        nn.Linear(512, num_classes),
    )
    return model

def get_shufflenet_model(num_classes, pretrained=False, dropout_rate=0.5):
    model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),  # Add dropout
        nn.Linear(512, num_classes),
    )
    return model

def get_mobilenet_model(num_classes, pretrained=False, dropout_rate=0.5):
    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),  # Add dropout
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model

def get_model(model_type, num_classes, pretrained=True, freeze=True, dropout_rate=0.5):
    if model_type == 'vgg':
        return get_vggface_model(num_classes, pretrained, freeze, dropout_rate)
    elif model_type == 'resnet':
        return get_resnet_model(num_classes, pretrained, dropout_rate)
    elif model_type == 'shufflenet':
        return get_shufflenet_model(num_classes, pretrained, dropout_rate)
    elif model_type == 'mobilenet':
        return get_mobilenet_model(num_classes, pretrained, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Log confusion matrix to TensorBoard (raw and normalized)
def log_confusion_matrix(writer, cm, class_names, epoch, stage):
    """
    Logs the confusion matrix (both raw and normalized) to TensorBoard as figures.
    """
    # Plot raw confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{stage} Confusion Matrix - Epoch {epoch}")
    writer.add_figure(f"{stage}/Confusion_Matrix_Raw", fig, epoch)
    plt.close(fig)

    # Plot normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row (true labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{stage} Normalized Confusion Matrix - Epoch {epoch}")
    writer.add_figure(f"{stage}/Confusion_Matrix_Normalized", fig, epoch)
    plt.close(fig)

# Training loop for one epoch
def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device, class_names, model_type, args):
    model.train()  # Ensure the model is in training mode
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_outputs = []

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)

    for step, (images, labels) in enumerate(train_loader_tqdm):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(logits, 1)
        all_labels.extend(labels.cpu().detach().numpy())
        all_preds.extend(preds.cpu().detach().numpy())
        all_outputs.extend(torch.softmax(logits, dim=1).cpu().detach().numpy())

        train_loader_tqdm.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr', average='weighted')

    print(f"Train Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, AUC: {auc:.4f}")
    writer.add_scalar("Train/Loss", total_loss / len(train_loader), epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)
    writer.add_scalar("Train/Precision", precision, epoch)
    writer.add_scalar("Train/Recall", recall, epoch)
    writer.add_scalar("Train/F1", f1, epoch)
    writer.add_scalar("Train/MSE", mse, epoch)
    writer.add_scalar("Train/MAE", mae, epoch)
    writer.add_scalar("Train/AUC", auc, epoch)

    # Log confusion matrix for training
    cm = confusion_matrix(all_labels, all_preds)
    log_confusion_matrix(writer, cm, class_names, epoch, "Train")

    # Compute LayerCAM during training if enabled
    if args.layercam_during_training and epoch % 1 == 0:  # Adjust frequency as needed
        compute_layercam(model, train_loader, device, class_names, writer, model_type, epoch, stage="Train")

    return acc

# Function to compute LayerCAM
def compute_layercam(model, data_loader, device, class_names, writer, model_type, epoch=None, stage="Train"):
    """
    Compute LayerCAM for a given dataset and log the results to TensorBoard.
    """
    model.eval()  # Ensure the model is in evaluation mode

    # Get all unique labels in the dataset
    unique_labels = set()
    for _, label in data_loader.dataset:
        unique_labels.add(label)
    unique_labels = list(unique_labels)

    # Initialize LayerCAM with the correct target layer based on model_type
    if model_type == 'vgg':
        target_layer = model.features[-2]  # Last convolutional layer in VGG
    elif model_type == 'resnet':
        target_layer = model.layer4[-1]  # Last convolutional layer in ResNet
    elif model_type == 'shufflenet':
        target_layer = model.conv5[-1]  # Last convolutional layer in ShuffleNet
    elif model_type == 'mobilenet':
        target_layer = model.features[-1]  # Last convolutional layer in MobileNet
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Use LayerCAM with a context manager
    with LayerCAM(model, target_layer) as cam_extractor:
        for label in unique_labels:
            # Collect all images with the current label
            images_with_label = [(image, image_label) for image, image_label in data_loader.dataset if image_label == label]

            if not images_with_label:
                print(f"No images found for class {label}. Skipping.")
                continue

            # Randomly select one image for the current label
            random_image, random_label = random.choice(images_with_label)

            # Prepare the input tensor
            input_tensor = random_image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            input_tensor.requires_grad_(True)  # Enable gradients for the input tensor

            # Forward pass
            out = model(input_tensor)

            # Retrieve the CAM for the current label
            activation_map = cam_extractor(label, out)

            if activation_map is not None:
                activation_map = activation_map[0]  # Use the first (and only) activation map

                # Overlay the heatmap on the original image
                heatmap = overlay_mask(
                    to_pil_image(input_tensor.squeeze(0).cpu()),  # Convert tensor to PIL image
                    to_pil_image(activation_map, mode='F'),       # Convert activation map to PIL image
                    alpha=0.5                                     # Transparency for the heatmap
                )

                # Convert the heatmap to a numpy array
                heatmap_np = np.array(heatmap)  # Convert PIL image to numpy array

                # Log the heatmap to TensorBoard
                if epoch is not None:
                    writer.add_image(f"{stage}/Heatmap_Class_{label}", heatmap_np, epoch, dataformats='HWC')
                else:
                    writer.add_image(f"{stage}/Heatmap_Class_{label}", heatmap_np, dataformats='HWC')
                print(f"Heatmap for class {label} logged to TensorBoard for {stage}.")
            else:
                print(f"LayerCAM returned None for class {label}. Check the model output.")

# Evaluation function
def evaluate(model, data_loader, criterion, epoch, writer, device, class_names, stage="Val", model_type=None):
    """
    Evaluate the model on a given dataset.
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating ({stage})", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_outputs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr', average='weighted')

    print(f"{stage} Loss: {total_loss / len(data_loader):.4f}, Accuracy: {acc:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, AUC: {auc:.4f}")
    print(f"{stage} Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    if writer is not None:
        writer.add_scalar(f"{stage}/Loss", total_loss / len(data_loader), epoch)
        writer.add_scalar(f"{stage}/Accuracy", acc, epoch)
        writer.add_scalar(f"{stage}/Precision", precision, epoch)
        writer.add_scalar(f"{stage}/Recall", recall, epoch)
        writer.add_scalar(f"{stage}/F1", f1, epoch)
        writer.add_scalar(f"{stage}/MSE", mse, epoch)
        writer.add_scalar(f"{stage}/MAE", mae, epoch)
        writer.add_scalar(f"{stage}/AUC", auc, epoch)

        # Log confusion matrix
        log_confusion_matrix(writer, cm, class_names, epoch, stage)

    return acc, cm

# Main training function
def train_model(args, type_model):
    writer = SummaryWriter(os.path.join(args.checkpoint, args.project_name))
    checkpoint_dir = os.path.join(args.checkpoint, args.project_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    if type_model == 'rgb':
        data_dir = os.path.join(args.data_dir, 'rgb')
    elif type_model == 'thermal':
        data_dir = os.path.join(args.data_dir, 'thermal')
    else:
        raise ValueError("Invalid type_model. Must be 'rgb' or 'thermal'.")
    
    train_loader, val_loader, test_loader = getSingleImageDataloader(
        args.batch_size,
        image_dir=data_dir,
        image_type="rgb"
    )

    num_classes = 5
    class_names = [f"Class {i}" for i in range(num_classes)]  # Replace with actual class names if available
    model = get_model(args.model_type, num_classes, dropout_rate=args.dropout_rate).to(device)

    # Manually set weights for each class
    class_weights = torch.tensor([0.1, 0.225, 0.225, 0.225, 0.225])  # Original weights

    # Make Class 0 weight 5 times smaller
    class_weights[0] = class_weights[0] / 1  # Reduce weight for Class 0

    # Normalize weights so they sum to 1
    class_weights = class_weights / class_weights.sum()
    print("Adjusted class weights:", class_weights)

    class_weights = class_weights.to(device)

    # Initialize the selected loss function
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == 'focal':
        criterion = FocalLoss(gamma=5.0)
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")

    # Optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step,
        gamma=args.lr_decay_gamma
    )

    # Early stopping
    best_accuracy = 0.0
    epochs_without_improvement = 0
    early_stopping_patience = args.early_stopping_patience

    save_interval = 5

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device, class_names, args.model_type, args)
        val_accuracy, cm = evaluate(model, val_loader, criterion, epoch, writer, device, class_names, stage="Val", model_type=args.model_type)

        # Check for improvement in validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best accuracy: {best_accuracy:.4f}. Best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation accuracy for {epochs_without_improvement} epochs.")

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. No improvement for {early_stopping_patience} epochs.")
            break

        # Save checkpoint every `save_interval` epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}.")

        # Step the learning rate scheduler
        scheduler.step()
        print(f"Learning rate updated to: {scheduler.get_last_lr()[0]:.6f}")

    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "model-final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}.")

    # Test the model on the test dataset after training
    print("\nTesting the model on the test dataset...")
    test_accuracy, test_cm = evaluate(model, test_loader, criterion, epoch + 1, writer, device, class_names, stage="Test")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)

    # Compute LayerCAM after training
    print("\nComputing LayerCAM for the test dataset...")
    compute_layercam(model, test_loader, device, class_names, writer, args.model_type, stage="Test")

    writer.close()

# Entry point
if __name__ == "__main__":
    args = parse_args()
    train_model(args, type_model='rgb')