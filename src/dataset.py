import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UltrasoundDataset(Dataset):
    """
    Dataset loader for B-mode ultrasound images and corresponding segmentation masks.
    Supports single-channel grayscale and multi-channel input.
    """
    def __init__(self, image_paths, mask_paths=None, transform=None, in_channels=1):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Read image
        if self.in_channels == 1:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image at path: {img_path}")
            image = np.expand_dims(image, axis=-1)  # Shape: (H, W, 1)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image at path: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask if available
        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask at path: {mask_path}")
            # Binarize mask (0 or 1)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = None

        # Apply Albumentations augmentations
        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]
        else:
            # Basic tensor conversion if no transform provided
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)
            if mask is not None:
                mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Ensure mask is shape (1, H, W)
        if mask is not None:
            if isinstance(mask, torch.Tensor) and mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif isinstance(mask, np.ndarray) and mask.ndim == 2:
                mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if mask is not None:
            return image, mask, os.path.basename(img_path)
        return image, os.path.basename(img_path)


def get_transforms(cfg, split="train"):
    """
    Build Albumentations pipeline for train / validation / test.
    """
    height = cfg["data"]["image_height"]
    width = cfg["data"]["image_width"]

    if split == "train":
        aug_cfg = cfg.get("augmentation", {})
        return A.Compose([
            A.Resize(height, width),
            A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_prob", 0.5)),
            A.RandomBrightnessContrast(p=aug_cfg.get("random_brightness_contrast_prob", 0.3)),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=aug_cfg.get("shift_scale_rotate_prob", 0.4)),
            A.GaussNoise(p=aug_cfg.get("gaussian_noise_prob", 0.2)),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height, width),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
            ToTensorV2()
        ])


def build_dataloaders(cfg):
    """
    Match image and mask files from dataset directories, split into train/val/test,
    and create PyTorch DataLoaders.
    """
    image_dir = cfg["data"]["image_dir"]
    mask_dir = cfg["data"]["mask_dir"]

    # Support relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(image_dir):
        image_dir = os.path.normpath(os.path.join(base_dir, image_dir))
    if not os.path.isabs(mask_dir):
        mask_dir = os.path.normpath(os.path.join(base_dir, mask_dir))

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found at: {image_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found at: {mask_dir}")

    valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".bmp")
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])
    
    matched_images = []
    matched_masks = []

    for img_name in image_files:
        base_name, _ = os.path.splitext(img_name)
        # Search for corresponding mask with same basename
        found = False
        for ext in valid_extensions:
            mask_candidate = os.path.join(mask_dir, f"{base_name}{ext}")
            if os.path.exists(mask_candidate):
                matched_images.append(os.path.join(image_dir, img_name))
                matched_masks.append(mask_candidate)
                found = True
                break
            # Try appending '_mask' suffix
            mask_candidate_suffix = os.path.join(mask_dir, f"{base_name}_mask{ext}")
            if os.path.exists(mask_candidate_suffix):
                matched_images.append(os.path.join(image_dir, img_name))
                matched_masks.append(mask_candidate_suffix)
                found = True
                break

    if len(matched_images) == 0:
        raise ValueError(f"No matching image-mask pairs found between '{image_dir}' and '{mask_dir}'.")

    print(f"Total matched pairs found: {len(matched_images)}")

    train_ratio = cfg["data"]["train_split"]
    val_ratio = cfg["data"]["val_split"]
    test_ratio = cfg["data"].get("test_split", 0.1)
    seed = cfg["data"].get("random_seed", 42)

    # Train / Temp split
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        matched_images, matched_masks,
        train_size=train_ratio,
        random_state=seed
    )

    # Val / Test split
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks,
        train_size=relative_val_ratio,
        random_state=seed
    )

    print(f"Train samples: {len(train_imgs)}, Val samples: {len(val_imgs)}, Test samples: {len(test_imgs)}")

    in_ch = cfg["data"]["in_channels"]
    train_ds = UltrasoundDataset(train_imgs, train_masks, transform=get_transforms(cfg, "train"), in_channels=in_ch)
    val_ds = UltrasoundDataset(val_imgs, val_masks, transform=get_transforms(cfg, "val"), in_channels=in_ch)
    test_ds = UltrasoundDataset(test_imgs, test_masks, transform=get_transforms(cfg, "test"), in_channels=in_ch)

    bs = cfg["training"]["batch_size"]
    workers = cfg["training"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader
