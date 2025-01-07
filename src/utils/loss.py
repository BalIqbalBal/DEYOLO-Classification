import torch
import torch.nn as nn


# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor): Class weights. If None, no class weights are applied.
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
            reduction (str): How to reduce the loss ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the predicted probabilities for the true class
        p_t = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Reduce the loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss