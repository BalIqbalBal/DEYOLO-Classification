import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.datasets import getDualImageDataloader


from model.TEDLM import TEDLMStackFusion, StackRGBThermalVGGFaceDNN1


def trainTEDLMStackFusion(project_name):
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=f"runs/TEDLMStackFusion/{project_name}")

    # Create checkpoint directory
    checkpoint_dir = f"runs/TEDLMStackFusion/{project_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Dataset 
    batch_size = 2
    train_loader, test_loader = getDualImageDataloader(batch_size)

    # Define model
    model = StackRGBThermalVGGFaceDNN1().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Use suitable loss for your problem
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 50
    save_interval = 5  # Default to save model every 5 epochs
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_accuracy = train_one_epoch(epoch, model, optimizer, criterion, train_loader, writer, device)
        test_accuracy = evaluate(epoch, model, criterion, test_loader, writer, device)

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
    writer.close()


# Training and evaluation function
def train_one_epoch(epoch, model, optimizer, criterion, train_loader, writer, device):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False)
    for rgb_images, thermal_images, labels in train_loader_tqdm:
        rgb_images, thermal_images, labels = rgb_images.to(device), thermal_images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(rgb_images, thermal_images)  # Forward pass
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(logits, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(torch.softmax(logits, dim=1).detach().numpy())

        train_loader_tqdm.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')

    print(f"Train Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.4f}")
    writer.add_scalar("Train/Loss", total_loss / len(train_loader), epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)
    writer.add_scalar("Train/Precision", precision, epoch)
    writer.add_scalar("Train/Recall", recall, epoch)
    writer.add_scalar("Train/F1", f1, epoch)
    writer.add_scalar("Train/MSE", mse, epoch)
    writer.add_scalar("Train/MAE", mae, epoch)
    writer.add_scalar("Train/AUC", auc, epoch)

    return acc

def evaluate(epoch, model, criterion, test_loader, writer, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for rgb_images, thermal_images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            rgb_images, thermal_images, labels = rgb_images.to(device), thermal_images.to(device), labels.to(device)

            logits = model(rgb_images, thermal_images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')

    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {acc:.4f}")
    writer.add_scalar("Test/Loss", total_loss / len(test_loader), epoch)
    writer.add_scalar("Test/Accuracy", acc, epoch)
    writer.add_scalar("Test/Precision", precision, epoch)
    writer.add_scalar("Test/Recall", recall, epoch)
    writer.add_scalar("Test/F1", f1, epoch)
    writer.add_scalar("Test/MSE", mse, epoch)
    writer.add_scalar("Test/MAE", mae, epoch)
    writer.add_scalar("Test/AUC", auc, epoch)

    return acc

trainTEDLMStackFusion('test')