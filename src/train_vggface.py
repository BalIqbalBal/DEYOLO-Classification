import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.datasets import getSingleImageDataloader  #
from torchvision import models

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=15)
    
    # SageMaker parameters
    parser.add_argument('--project-name', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))

def get_vggface_model(num_classes, pretrained=True, freeze=True):
    # Load pre-trained VGG16 model
    model = models.vgg16(pretrained=pretrained)
    
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False  # Freeze the convolutional base
    
    # Modify the classifier
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    return model

def trainVGGFace(args, type_model):
    # Initialize TensorBoard with SageMaker output path
    writer = SummaryWriter(os.path.join(args.checkpoint, args.project_name))

    # Create checkpoint directory in SageMaker model directory
    checkpoint_dir = os.path.join(args.checkpoint, args.project_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Set dataset directory based on model type
    if type_model == 'rgb':
        data_dir = os.path.join(args.data_dir, 'rgb')
    elif type_model == 'thermal':
        data_dir = os.path.join(args.data_dir, 'thermal')
    else:
        raise ValueError("Invalid type_model. Must be 'rgb' or 'thermal'.")
    
    train_loader, test_loader = getSingleImageDataloader(
        args.batch_size,
        image_dir=data_dir,
        image_type="rgb"
    )

    # Define model
    num_classes = 5
    model = get_vggface_model(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    save_interval = 5  # Save model every 5 epochs
    best_accuracy = 0.0

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device=device)
        test_accuracy = evaluate(model, test_loader, criterion, epoch, writer, device=device)

        # Save checkpoint if test accuracy improves
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best accuracy: {best_accuracy:.4f}. Best model saved.")
        
        # Save checkpoint every `save_interval` epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}.")

    # Close TensorBoard writer
    # Save final model
    final_model_path = os.path.join(args.model_dir, "model-final.pth")
    torch.save(model.state_dict(), final_model_path)

    writer.close()

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer, device):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_outputs = []

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)
    for step, (images, labels) in enumerate(train_loader_tqdm):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)  # Forward pass
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

    return acc

# Evaluation function
def evaluate(model, test_loader, criterion, epoch, writer, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
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

    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {acc:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, AUC: {auc:.4f}")
    writer.add_scalar("Test/Loss", total_loss / len(test_loader), epoch)
    writer.add_scalar("Test/Accuracy", acc, epoch)
    writer.add_scalar("Test/Precision", precision, epoch)
    writer.add_scalar("Test/Recall", recall, epoch)
    writer.add_scalar("Test/F1", f1, epoch)
    writer.add_scalar("Test/MSE", mse, epoch)
    writer.add_scalar("Test/MAE", mae, epoch)
    writer.add_scalar("Test/AUC", auc, epoch)

    return acc

if __name__ == "__main__":
    args = parse_args()
    trainVGGFace(args)
