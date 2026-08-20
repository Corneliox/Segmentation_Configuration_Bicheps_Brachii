import torch
import torch.nn as nn
import numpy as np
import cv2


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combines Binary Cross-Entropy (BCE) and Dice Loss."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


def compute_metrics(pred, target, threshold=0.5, eps=1e-7):
    """
    Computes Dice Coefficient and IoU (Intersection over Union).
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum().item()
    union = pred_bin.sum().item() + target_bin.sum().item() - intersection
    
    dice = (2.0 * intersection + eps) / (pred_bin.sum().item() + target_bin.sum().item() + eps)
    iou = (intersection + eps) / (union + eps)

    return {"dice": dice, "iou": iou}


def postprocess_mask(mask, min_size=500, kernel_size=5):
    """
    Cleans up binary segmentation mask using morphological operations
    and keeps largest connected component (representing Biceps Brachii).
    """
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Morphological Closing & Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        # Index 0 is background; find largest foreground
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = (labels == largest_label).astype(np.uint8) * 255
        return (largest_mask > 127).astype(np.float32)

    return (cleaned > 127).astype(np.float32)


def calculate_muscle_thickness(binary_mask, pixel_spacing_mm=0.1):
    """
    Calculates muscle thickness from the binary mask:
    For each column, finds the topmost active pixel (superficial fascia)
    and bottommost active pixel (deep fascia).
    Returns mean thickness, max thickness, and thickness profile.
    """
    h, w = binary_mask.shape
    thickness_profile = []

    for col in range(w):
        column_pixels = np.where(binary_mask[:, col] > 0.5)[0]
        if len(column_pixels) > 0:
            top_y = column_pixels[0]
            bottom_y = column_pixels[-1]
            thickness_px = bottom_y - top_y
            thickness_profile.append(thickness_px * pixel_spacing_mm)

    if len(thickness_profile) == 0:
        return {"mean_thickness_mm": 0.0, "max_thickness_mm": 0.0, "profile": []}

    return {
        "mean_thickness_mm": float(np.mean(thickness_profile)),
        "std_thickness_mm": float(np.std(thickness_profile)),
        "max_thickness_mm": float(np.max(thickness_profile)),
        "profile": thickness_profile
    }
