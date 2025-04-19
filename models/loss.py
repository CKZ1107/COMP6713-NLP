import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss functions
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # weight for each class
        self.gamma = gamma  # focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        # standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # focal loss component
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Combined Weighted Focal Loss.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', class_weights=None):
        super(WeightedFocalLoss, self).__init__()
        # alpha is for sample weighting, class_weights is for label weighting
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        # label weighted CE loss
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,
            reduction='none'
        )

        # sample weighting (alpha)
        if self.alpha is not None:
            alpha_weights = self.alpha[targets]
            ce_loss = alpha_weights * ce_loss

        # focal component
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
