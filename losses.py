"""
Loss functions for cancer instance segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, CLASS_WEIGHTS, DEVICE


def iou_metric(pred, gt, num_classes, weights=None, softmax=False):
    """
    Calculate the weighted mean Intersection over Union (IoU).

    Args:
        pred: Predicted logits or probabilities
        gt: Ground truth masks
        num_classes: Number of classes
        weights: Class weights
        softmax: Whether to apply softmax to predictions

    Returns:
        Mean IoU score
    """
    # Run softmax if input is logits
    if softmax:
        pred = nn.Softmax(dim=1)(pred)

    # One-hot encoding of ground truth
    gt = F.one_hot(gt.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Computation of intersection and union
    intersection = torch.sum(gt * pred, dim=(2, 3))
    union = torch.sum(pred + gt, dim=(2, 3)) - intersection

    # Apply weights
    if weights is not None:
        intersection = weights * intersection
        union = weights * union

    # Compute IoU avoiding division by zero
    valid = union > 0
    iou = torch.zeros_like(union)
    iou[valid] = intersection[valid] / union[valid]

    # Mean IoU across all valid entries
    return iou[valid].mean()


class IoULoss(nn.Module):
    """IoU-based loss function."""

    def __init__(self, num_classes, weights=None, softmax=False):
        super().__init__()
        self.softmax = softmax
        self.num_classes = num_classes
        self.weights = weights

    def forward(self, pred, gt):
        """Convert IoU score to a loss value."""
        return -torch.log(iou_metric(pred, gt, self.num_classes, self.weights, self.softmax))


def get_loss_function(use_cross_entropy=False):
    """
    Get the appropriate loss function.

    Args:
        use_cross_entropy: If True, use CrossEntropyLoss; otherwise use IoULoss

    Returns:
        Loss function
    """
    if use_cross_entropy:
        return nn.CrossEntropyLoss(reduction='mean')
    else:
        weights = torch.tensor(CLASS_WEIGHTS).to(DEVICE)
        return IoULoss(softmax=True, num_classes=NUM_CLASSES, weights=weights)
