import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.dataset_ay import getDualImageDataloader
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.loss import FocalLoss

from model.DEYOLO import SimpleDEYOLOCLASS
from model.modal_fusion_ay import resnetDecaDepa, vggfaceDecaDepa

# Parse arguments
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
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'focal'], help="Loss function to use.")
    parser.add_argument('--model-type', type=str, default='resnet', choices=['resnetfusion', 'vggfacefusion'], help="Type of backbone model to use.")
    
    # SageMaker parameters
    parser.add_argument('--project-name', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    
    return parser.parse_args()

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
        
        #print(f"Min label: {labels.min().item()}, Max label: {labels.max().item()}")
        #print(f"Logits shape: {logits.shape}")  # Seharusnya [batch_size, n_classes]
        #print(f"Unique labels: {labels.unique()}")  # Pastikan semua dalam rentang 0 hingga n_classes - 1

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

    print(f"Train Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
    writer.add_scalar("Train/Loss", total_loss / len(train_loader), epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)
    writer.add_scalar("Train/Precision", precision, epoch)
    writer.add_scalar("Train/Recall", recall, epoch)
    writer.add_scalar("Train/F1", f1, epoch)
    writer.add_scalar("Train/MSE", mse, epoch)
    writer.add_scalar("Train/MAE", mae, epoch)

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

    print(f"{stage} Loss: {total_loss / len(data_loader):.4f}, Accuracy: {acc:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
    writer.add_scalar(f"{stage}/Loss", total_loss / len(data_loader), epoch)
    writer.add_scalar(f"{stage}/Accuracy", acc, epoch)
    writer.add_scalar(f"{stage}/Precision", precision, epoch)
    writer.add_scalar(f"{stage}/Recall", recall, epoch)
    writer.add_scalar(f"{stage}/F1", f1, epoch)
    writer.add_scalar(f"{stage}/MSE", mse, epoch)
    writer.add_scalar(f"{stage}/MAE", mae, epoch)

    # Log confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    log_confusion_matrix(writer, cm, class_names, epoch, stage)

    return acc, cm

# Main training function
def train_multimodal_model(args):
    # Initialize TensorBoard with SageMaker output path
    writer = SummaryWriter(os.path.join(args.checkpoint, args.project_name))

    # Create checkpoint directory in SageMaker model directory
    checkpoint_dir = os.path.join(args.checkpoint, args.project_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)


    # Dataset
    train_loader, val_loader = getDualImageDataloader(
        batch_size=args.batch_size,
        dataset_dir=args.data_dir,
    )

    # Define model based on the selected type
    if args.model_type == 'resnetmm':
        model = resnetDecaDepa(dropout_rate=args.dropout_rate).to(device)
    elif args.model_type == 'vggfacemm':
        model = vggfaceDecaDepa(dropout_rate=args.dropout_rate).to(device)
    elif args.model_type == 'deyolomm':
        model = SimpleDEYOLOCLASS(dropout_rate=args.dropout_rat).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    num_classes = 49
    class_names = [f"Label {i}" for i in range(num_classes)]  # Replace with actual class names if available


    # Initialize the selected loss function
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
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
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch.pth")
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
    #test_accuracy, test_cm = evaluate(model, test_loader, criterion, epoch + 1, writer, device, class_names, stage="Test")
    print