import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score
from model.TEDLM import TEDLMFeatureFusion

def test_model(project_name, test_loader):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Define model
    model = TEDLMFeatureFusion(n_components=128).to(device)

    # Load the best model
    checkpoint_path = f"runs/TEDLMFeatureFusion/{project_name}/best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Evaluation
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_outputs = []

    criterion = torch.nn.CrossEntropyLoss()  # Use the same loss as during training

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images, images)  # Forward pass
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_outputs.extend(logits.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr', average='weighted')

    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {acc:.4f}")
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    test_model("project_name")
