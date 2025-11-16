"""
Metrics and evaluation utilities for cancer instance segmentation.
"""
import torch
import torch.nn as nn
import torchmetrics as TM
from losses import iou_metric, get_loss_function
from config import NUM_CLASSES, CLASS_WEIGHTS, DEVICE


def calculate_metrics(model, loader, use_cross_entropy=False):
    """
    Calculate metrics for model evaluation.

    Args:
        model: The segmentation model
        loader: DataLoader for evaluation
        use_cross_entropy: Whether to use cross-entropy loss

    Returns:
        avg_loss: Average loss
        avg_accuracy: Average pixel accuracy
        avg_iou: Average IoU
        avg_f1: Average F1 score
    """
    model.eval()
    criterion = get_loss_function(use_cross_entropy)

    f1score = TM.classification.MulticlassF1Score(NUM_CLASSES, average='micro').to(DEVICE)
    pixel_metric = TM.classification.MulticlassAccuracy(NUM_CLASSES, average='micro').to(DEVICE)

    f1_scores = []
    pixel_accuracies = []
    iou_accuracies = []
    running_loss = 0.0
    running_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            predictions = model(inputs)

            # Calculate loss
            if use_cross_entropy:
                targets_for_loss = targets.squeeze(dim=1)
            else:
                targets_for_loss = targets

            loss = criterion(predictions, targets_for_loss)
            running_samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

            # Get predictions
            pred_probabilities = nn.Softmax(dim=1)(predictions)
            pred_labels = predictions.argmax(dim=1).unsqueeze(1)
            pred_mask = pred_labels.to(torch.float)

            # Calculate metrics
            f1_score = f1score(pred_mask, targets)
            pixel_accuracy = pixel_metric(pred_labels, targets)
            weights = torch.tensor(CLASS_WEIGHTS).to(DEVICE)
            iou = iou_metric(pred_probabilities, targets, num_classes=NUM_CLASSES, weights=weights)

            f1_scores.append(f1_score.item())
            pixel_accuracies.append(pixel_accuracy.item())
            iou_accuracies.append(iou.item())

    # Compute averages
    avg_loss = running_loss / running_samples
    avg_accuracy = torch.tensor(pixel_accuracies).mean().item()
    avg_iou = torch.tensor(iou_accuracies).mean().item()
    avg_f1 = torch.tensor(f1_scores).mean().item()

    return avg_loss, avg_accuracy, avg_iou, avg_f1
