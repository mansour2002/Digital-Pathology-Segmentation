"""
Configuration file for cancer instance segmentation training.
"""
import torch

# Classes and their properties
CLASSES = ['Background', 'Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial']
NUM_CLASSES = len(CLASSES)
CLASS_COLORS = [(0, 0, 0), (255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
CLASS_WEIGHTS = [0.1, 1, 1, 1, 1, 1]

# Image settings
IMAGE_SIZE = 256

# Training settings
BATCH_SIZE = 15
LEARNING_RATE = 0.001
EPOCHS = 300
SEED = 42

# Cross-validation settings
FOLD_NUM = 1
MAX_FOLD = 5

# Model settings
MODEL_ENCODER = "efficientnet-b7"
ENCODER_WEIGHTS = "imagenet"

# Scheduler settings
SCHEDULER_STEP_SIZE = 25
SCHEDULER_GAMMA = 0.7

# Loss function
USE_CROSS_ENTROPY = False  # If False, uses IoU Loss

# Device configuration
def get_device():
    """Return the appropriate device (CUDA or CPU) based on availability."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()
