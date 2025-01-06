# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss implementation.

        Args:
            alpha (float or list): Weighting factor for class balancing. Can be a float or a list of weights for each class.
            gamma (float): Focusing parameter. Higher values reduce the loss contribution from easy examples.
            reduction (str): Reduction method for the loss ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the predicted probabilities for the true class
        pt = torch.exp(-ce_loss)  # pt = p if target == 1 else 1-p

        # Compute Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss