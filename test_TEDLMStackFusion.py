import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score
from model.TEDLM import TEDLMStackFusion

def test_model(project_name, test_loader):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Define model
    model = TEDLMStackFusion(n_components=128).to(device)

    # Load the best model checkpoint
    checkpoint_path = f"runs/TEDLMStackFusion/{project_name}/best_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Please check the path.")
        return

    model.eval()  # Set the model to evaluation mode

    # Evaluation
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    criterion = torch.nn.CrossEntropyLoss()  # Use the same loss function as during training

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for rgb_images, thermal_images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            rgb_images, thermal_images, labels = rgb_images.to(device), thermal_images.to(device), labels.to(device)

            # Forward pass
            logits = model(rgb_images, thermal_images)

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Predictions
            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # For AUC, get the softmax probabilities
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')

    # Print metrics
    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {acc:.4f}")
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    test_model("project_name")  # Replace "project_name" with the actual project name
