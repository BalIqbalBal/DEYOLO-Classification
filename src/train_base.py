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

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=15)
    parser.add_argument('--model-type', type=str, default='vgg', choices=['vgg', 'resnet', 'shufflenet', 'mobilenet'])
    
    # SageMaker parameters
    parser.add_argument('--project-name', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))

    return parser.parse_args()

def get_vggface_model(num_classes, pretrained=True, freeze=True):
    model = models.vgg16(pretrained=pretrained)
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model

def get_resnet_model(num_classes, pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_shufflenet_model(num_classes, pretrained=False):
    model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_mobilenet_model(num_classes, pretrained=False):
    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_model(model_type, num_classes, pretrained=True, freeze=True):
    if model_type == 'vgg':
        return get_vggface_model(num_classes, pretrained, freeze)
    elif model_type == 'resnet':
        return get_resnet_model(num_classes, freeze, pretrained=False)
    elif model_type == 'shufflenet':
        return get_shufflenet_model(num_classes, freeze, pretrained=False)
    elif model_type == 'mobilenet':
        return get_mobilenet_model(num_classes, freeze, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device, class_names):
    model.train()
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

def evaluate(model, data_loader, criterion, epoch, writer, device, class_names, stage="Val"):
    """
    Evaluate the model on a given dataset.
    
    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        epoch (int): Current epoch.
        writer (SummaryWriter): TensorBoard SummaryWriter object.
        device (torch.device): Device to run the model on.
        class_names (list): List of class names.
        stage (str): Stage of evaluation ("Val" or "Test").
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
    model = get_model(args.model_type, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    save_interval = 5
    best_accuracy = 0.0

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device, class_names)
        val_accuracy, cm = evaluate(model, val_loader, criterion, epoch, writer, device, class_names, stage="Val")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best accuracy: {best_accuracy:.4f}. Best model saved.")
        
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}.")

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

    writer.close()

if __name__ == "__main__":
    args = parse_args()
    train_model(args, type_model='rgb')