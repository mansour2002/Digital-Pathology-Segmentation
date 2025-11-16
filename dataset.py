"""
Dataset and data loading utilities for cancer instance segmentation.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from config import IMAGE_SIZE


def load_data(data_dir, limit_per_part=1000):
    """
    Load images and masks from the data directory.

    Args:
        data_dir: Root directory containing Part 1, Part 2, Part 3 folders
        limit_per_part: Number of samples to load per part (default: 1000)

    Returns:
        images: Concatenated numpy array of images
        masks: Concatenated numpy array of masks
    """
    parts = ['Part 1', 'Part 2', 'Part 3']
    images_list = []
    masks_list = []

    for part in parts:
        img_path = f"{data_dir}/{part}/Images/images.npy"
        mask_path = f"{data_dir}/{part}/Masks/masks.npy"

        images_list.append(np.load(img_path, mmap_mode='r')[:limit_per_part])
        masks_list.append(np.load(mask_path, mmap_mode='r')[:limit_per_part])

    images = np.concatenate(images_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)

    return images, masks


def delete_empty_tiles(images, masks):
    """Remove tiles that don't contain any cells."""
    del_ind = []
    for i in range(masks.shape[0]):
        if np.max(masks[i, :, :, :5]) == 0:
            del_ind.append(i)

    print(f"Removing {len(del_ind)} empty tiles: {del_ind}")
    images = np.delete(images, del_ind, 0)
    masks = np.delete(masks, del_ind, 0)

    return images, masks


def vectorize_3d_mask(masks):
    """
    Convert 3D multi-channel masks to 2D class indices.

    Args:
        masks: numpy array of shape (N, H, W, 6)

    Returns:
        mask_2d: numpy array of shape (N, H, W, 1) with class indices
    """
    if masks.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 6):
        raise ValueError(f"Expected shape for each mask: ({IMAGE_SIZE}, {IMAGE_SIZE}, 6)")

    # Initialize the 2D mask array
    mask_2d = np.zeros(masks.shape[:3], dtype=int)

    # Apply mask channel indices, preferring lower index channels where overlaps occur
    for channel in range(5):  # 6 classes, but first is background
        channel_mask = (masks[..., channel] != 0) & (mask_2d == 0)
        mask_2d[channel_mask] = channel + 1

    return np.expand_dims(mask_2d, axis=-1)


def split_data(images, masks, fold_num=1, max_fold=5):
    """
    Split images and masks into train and validation sets using k-fold strategy.

    Args:
        images: numpy array of images
        masks: numpy array of masks
        fold_num: Current fold number (1-indexed)
        max_fold: Total number of folds

    Returns:
        train_images, train_masks, valid_images, valid_masks
    """
    num_samples = images.shape[0]
    start_val = int((fold_num - 1) / max_fold * num_samples)
    end_val = int(fold_num / max_fold * num_samples)

    train_images = np.concatenate((images[:start_val], images[end_val:]), axis=0)
    train_masks = np.concatenate((masks[:start_val], masks[end_val:]), axis=0)
    valid_images = images[start_val:end_val]
    valid_masks = masks[start_val:end_val]

    return train_images, train_masks, valid_images, valid_masks


def get_train_transforms(img_size):
    """Augmentations for training images and masks."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(shear=0.4, mode=4, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.5, p=1.0),
        A.ShiftScaleRotate(scale_limit=0.4, rotate_limit=180, shift_limit=0.5, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True),
        A.Blur(blur_limit=3, p=0.2),
        A.Sharpen(alpha=0.1, p=0.2)
    ])


def get_valid_transforms(img_size):
    """Augmentations for validation images and masks."""
    return A.Compose([A.Resize(img_size, img_size, always_apply=True)])


class SegmentationDataset(Dataset):
    """Dataset class for cancer instance segmentation."""

    def __init__(self, images, masks, transforms):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Apply transformations
        transformed = self.transforms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Convert to tensors
        image = torch.from_numpy(np.transpose(image / 255, (2, 0, 1))).float()
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).long()

        return image, mask


def prepare_dataloaders(data_dir, batch_size, fold_num=1, max_fold=5, limit_per_part=1000):
    """
    Complete data preparation pipeline.

    Args:
        data_dir: Root directory containing the data
        batch_size: Batch size for training
        fold_num: Current fold number
        max_fold: Total number of folds
        limit_per_part: Number of samples to load per part

    Returns:
        train_loader, valid_loader
    """
    # Load data
    print("Loading data...")
    images, masks = load_data(data_dir, limit_per_part)
    print(f"Loaded images: {images.shape}, masks: {masks.shape}")

    # Remove empty tiles
    images, masks = delete_empty_tiles(images, masks)
    print(f"After removing empty tiles - images: {images.shape}, masks: {masks.shape}")

    # Vectorize masks
    masks = vectorize_3d_mask(masks)
    images = np.uint8(images)
    print(f"Vectorized masks: {masks.shape}")

    # Split data
    train_images, train_masks, valid_images, valid_masks = split_data(
        images, masks, fold_num, max_fold
    )
    print(f"Train: {train_images.shape}, Valid: {valid_images.shape}")

    # Create datasets
    train_dataset = SegmentationDataset(
        train_images, train_masks, get_train_transforms(IMAGE_SIZE)
    )
    valid_dataset = SegmentationDataset(
        valid_images, valid_masks, get_valid_transforms(IMAGE_SIZE)
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, valid_loader
