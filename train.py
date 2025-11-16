"""
Main training script for cancer instance segmentation.

Usage:
    python train.py --data_dir /path/to/data --save_dir ./outputs
"""
import os
import argparse
import time
import torch
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, SEED, FOLD_NUM, MAX_FOLD,
    SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA, USE_CROSS_ENTROPY, DEVICE
)
from dataset import prepare_dataloaders
from model import create_model
from losses import get_loss_function
from metrics import calculate_metrics


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, optimizer, criterion, use_cross_entropy):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)

        if use_cross_entropy:
            targets = targets.squeeze(dim=1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_samples += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)

    return running_loss / running_samples


def plot_metrics(train_data, val_data, ylabel, title, save_path, filename):
    """Helper function to plot and save metrics."""
    plt.figure(figsize=(10, 7), facecolor='white')
    plt.plot(train_data, color='tab:blue', linestyle='-', label=f'train {ylabel}')
    plt.plot(val_data, color='tab:red', linestyle='-', label=f'validation {ylabel}')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_path, filename))
    plt.close()


def save_plots(train_acc, val_acc, train_loss, val_loss, train_iou, val_iou, save_dir):
    """Save training plots."""
    plot_metrics(train_acc, val_acc, 'accuracy', 'Training vs Validation Accuracy', save_dir, 'accuracy.png')
    plot_metrics(train_loss, val_loss, 'loss', 'Training vs Validation Loss', save_dir, 'loss.png')
    plot_metrics(train_iou, val_iou, 'IoU', 'Training vs Validation mIoU', save_dir, 'iou.png')


def train(args):
    """Main training function."""
    # Set seed for reproducibility
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving results to: {args.save_dir}")

    # Prepare data
    print("\n" + "="*50)
    print("Preparing data...")
    print("="*50)
    train_loader, valid_loader = prepare_dataloaders(
        args.data_dir,
        args.batch_size,
        args.fold_num,
        args.max_fold,
        args.limit_per_part
    )

    # Create model
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    model = create_model()
    print(f"Model created and moved to {DEVICE}")

    # Setup optimizer, scheduler, and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )
    criterion = get_loss_function(args.use_cross_entropy)

    # Training loop
    print("\n" + "="*50)
    print(f"Starting training for {args.epochs} epochs...")
    print("="*50)

    train_loss_list, train_acc_list, train_iou_list, train_f1_list = [], [], [], []
    val_loss_list, val_acc_list, val_iou_list, val_f1_list = [], [], [], []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.use_cross_entropy)

        # Evaluate
        train_loss, train_acc, train_iou, train_f1 = calculate_metrics(
            model, train_loader, args.use_cross_entropy
        )
        val_loss, val_acc, val_iou, val_f1 = calculate_metrics(
            model, valid_loader, args.use_cross_entropy
        )

        # Store metrics
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_iou_list.append(train_iou)
        train_f1_list.append(train_f1)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_iou_list.append(val_iou)
        val_f1_list.append(val_f1)

        # Print progress
        elapsed_time = int(time.time() - start_time)
        print(f"Epoch: {epoch:04d}, Time: {elapsed_time:04d}s, LR: {optimizer.param_groups[0]['lr']:.5f}, "
              f"Training: (Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, IoU: {train_iou:.4f}), "
              f"Validation: (Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, IoU: {val_iou:.4f})")

        start_time = time.time()

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(args.save_dir, "trained_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete! Model saved to: {final_model_path}")

    # Save plots
    save_plots(train_acc_list, val_acc_list, train_loss_list, val_loss_list,
               train_iou_list, val_iou_list, args.save_dir)
    print(f"Training plots saved to: {args.save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train cancer instance segmentation model')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing Part 1, Part 2, Part 3 folders')
    parser.add_argument('--save_dir', type=str, default='./outputs',
                        help='Directory to save model and results (default: ./outputs)')
    parser.add_argument('--limit_per_part', type=int, default=1000,
                        help='Number of samples to load per part (default: 1000)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')

    # Cross-validation arguments
    parser.add_argument('--fold_num', type=int, default=FOLD_NUM,
                        help=f'Current fold number (default: {FOLD_NUM})')
    parser.add_argument('--max_fold', type=int, default=MAX_FOLD,
                        help=f'Maximum number of folds (default: {MAX_FOLD})')

    # Scheduler arguments
    parser.add_argument('--scheduler_step', type=int, default=SCHEDULER_STEP_SIZE,
                        help=f'Scheduler step size (default: {SCHEDULER_STEP_SIZE})')
    parser.add_argument('--scheduler_gamma', type=float, default=SCHEDULER_GAMMA,
                        help=f'Scheduler gamma (default: {SCHEDULER_GAMMA})')

    # Loss function
    parser.add_argument('--use_cross_entropy', action='store_true',
                        help='Use CrossEntropyLoss instead of IoULoss')

    args = parser.parse_args()

    # Run training
    train(args)


if __name__ == '__main__':
    main()
