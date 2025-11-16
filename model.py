"""
Model architecture for cancer instance segmentation.
"""
import segmentation_models_pytorch as smp
from config import MODEL_ENCODER, ENCODER_WEIGHTS, NUM_CLASSES, DEVICE


def create_model():
    """
    Create and initialize the segmentation model.

    Returns:
        model: UNet model with EfficientNet-B7 encoder
    """
    model = smp.Unet(
        MODEL_ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=NUM_CLASSES
    )
    model = model.to(DEVICE)
    return model
