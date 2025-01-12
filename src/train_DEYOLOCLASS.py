import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.datasets import getDualImageDataloader
import argparse
from model.DEYOLO import DEYOLOCLASS, SimpleDEYOLOCLASS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.loss import FocalLoss

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=15)
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
    
    return parser.parse_args()

def log_confusion_matrix(writer, cm, class_names, epoch, stage):
    """
    Logs the confusion matrix to TensorBoard as a figure.
    
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter object.
        cm (np.array): Confusion matrix.
        class_names (list): List of class names.
        epoch (int): Current epoch.
        stage (str): Stage of the confusion matrix (e.g., "Train", "Val", or "Test").
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{stage} Confusion Matrix - Epoch {epoch}")
    writer.add_figure(f"{stage}/Confusion_Matrix", fig, epoch)
    plt.close(fig)

def trainDEYOLOCLASS(args):
    # Initialize TensorBoard with SageMaker output path
    writer = SummaryWriter(os.path.join(args.checkpoint, args.project_name))

    # Create checkpoint directory in SageMaker model directory
    checkpoint_dir = os.path.join(args.checkpoint, args.project_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Setup data directories
    rgb_dir = os.path.join(args.data_dir, 'rgb')
    thermal_dir = os.path.join(args.data_dir, 'thermal')
    print(f"Loading data from:\nRGB: {rgb_dir}\nThermal: {thermal_dir}")

    # Dataset
    train_loader, val_loader, test_loader = getDualImageDataloader(
        args.batch_size,
        rgb_dir=rgb_dir,
        thermal_dir=thermal_dir,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )

    # Define model
    #model = DEYOLOCLASS().to(device)
    model = SimpleDEYOLOCLASS(dropout_rate=args.dropout_rate).to(device)  # Add dropout to the model

    num_classes = 5
    class_names = [f"Label {i}" for i in range(num_classes)]  # Replace with actual class names if available

    # Calculate class weights for Focal Loss
    labels = [label for _, _, label in train_loader.dataset]  # Extract labels from the dataset
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()  # Inverse of class frequency
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)

    # Initialize Focal Loss with class weights
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

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
        
        # Train for one epoch
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device, class_names)
        
        # Evaluate on validation set
        val_accuracy, val_cm = evaluate(model, val_loader, criterion, epoch, writer, device, class_names, stage="Val")
        
        # Save checkpoint if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best validation accuracy: {best_accuracy:.4f}. Best model saved to {model_path}")
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

    # Test the model on the test set
    print("\nTesting the model on the test dataset...")
    test_accuracy, test_cm = evaluate(model, test_loader, criterion, epoch + 1, writer, device, class_names, stage="Test")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)

    writer.close()

# Training function

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device, class_names):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_outputs = []

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)

    for step, (rgb_images, thermal_images, labels) in enumerate(train_loader_tqdm):
        rgb_images, thermal_images, labels = rgb_images.to(device), thermal_images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(rgb_images, thermal_images)  # Forward pass
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(logits, 1)
        all_labels.extend(labels.cpu().detach().numpy())
        all_preds.extend(preds.cpu().detach().numpy())
        all_outputs.extend(torch.softmax(logits, dim=1).cpu().detach().numpy())

        # Set progress bar postfix and print current step info
        train_loader_tqdm.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Metrics
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

    return acc

# Evaluation function
def evaluate(model, data_loader, criterion, epoch, writer, device, class_names, stage="Val"):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_outputs = []

    with torch.no_grad():
        for rgb_images, thermal_images, labels in tqdm(data_loader, desc=f"Evaluating ({stage})", leave=False):
            rgb_images, thermal_images, labels = rgb_images.to(device), thermal_images.to(device), labels.to(device)

            logits = model(rgb_images, thermal_images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_outputs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr', average='weighted')

    print(f"{stage} Loss: {total_loss / len(data_loader):.4f}, Accuracy: {acc:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, AUC: {auc:.4f}")
    writer.add_scalar(f"{stage}/Loss", total_loss / len(data_loader), epoch)
    writer.add_scalar(f"{stage}/Accuracy", acc, epoch)
    writer.add_scalar(f"{stage}/Precision", precision, epoch)
    writer.add_scalar(f"{stage}/Recall", recall, epoch)
    writer.add_scalar(f"{stage}/F1", f1, epoch)
    writer.add_scalar(f"{stage}/MSE", mse, epoch)
    writer.add_scalar(f"{stage}/MAE", mae, epoch)
    writer.add_scalar(f"{stage}/AUC", auc, epoch)

    # Log confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    log_confusion_matrix(writer, cm, class_names, epoch, stage)

    return acc, cm

if __name__ == "__main__":
    args = parse_args()
    trainDEYOLOCLASS(args)