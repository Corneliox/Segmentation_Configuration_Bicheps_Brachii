import os
import sys
import yaml
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from utils import compute_metrics, postprocess_mask, calculate_muscle_thickness


def evaluate_model(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device_name = cfg["project"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Read physical scale calibration if available
    calib_file = cfg["data"].get("calibration_file", "../dataset/spatial_calibration.csv")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(calib_file):
        calib_file = os.path.normpath(os.path.join(base_dir, calib_file))

    pixel_spacing_mm = 0.1113  # Default fallback
    if os.path.exists(calib_file):
        try:
            calib_df = pd.read_csv(calib_file)
            pixel_spacing_mm = float(calib_df["MM_per_pixel"].iloc[0])
            print(f"Loaded spatial calibration: 1 pixel = {pixel_spacing_mm:.4f} mm")
        except Exception as e:
            print(f"Warning reading calibration file: {e}")

    # Build Dataloaders
    _, _, test_loader = build_dataloaders(cfg)

    # Load Model Checkpoint
    checkpoint_path = cfg["inference"].get("checkpoint_path", "./weights/best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = cfg["inference"].get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    threshold = cfg["inference"].get("threshold", 0.5)
    total_dice = []
    total_iou = []

    print("\n--- Evaluating on Test Set ---")
    with torch.no_grad():
        for images, masks, filenames in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            metrics = compute_metrics(preds, masks, threshold=threshold)
            total_dice.append(metrics["dice"])
            total_iou.append(metrics["iou"])

            # Save Visualizations
            if cfg["inference"].get("save_visualizations", True):
                img_np = images[0, 0].cpu().numpy()
                img_np = ((img_np * 0.5 + 0.5) * 255).astype(np.uint8)  # Denormalize

                gt_mask = masks[0, 0].cpu().numpy()
                raw_pred_mask = (preds[0, 0].cpu().numpy() > threshold).astype(np.float32)
                cleaned_pred = postprocess_mask(raw_pred_mask)

                # Calculate muscle thickness in millimeters
                thickness_info = calculate_muscle_thickness(cleaned_pred, pixel_spacing_mm=pixel_spacing_mm)

                # Create 4-panel figure
                fig, axes = plt.subplots(1, 4, figsize=(18, 5))
                
                axes[0].imshow(img_np, cmap="gray")
                axes[0].set_title("Input B-Mode US")
                axes[0].axis("off")

                axes[1].imshow(img_np, cmap="gray")
                axes[1].imshow(gt_mask, cmap="jet", alpha=0.4)
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                axes[2].imshow(img_np, cmap="gray")
                axes[2].imshow(cleaned_pred, cmap="autumn", alpha=0.4)
                axes[2].set_title(f"Predicted Biceps Mask\nDice: {metrics['dice']:.3f}, IoU: {metrics['iou']:.3f}")
                axes[2].axis("off")

                # Boundary Overlay (Fascia Tracking)
                contours, _ = cv2.findContours((cleaned_pred * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

                axes[3].imshow(overlay)
                axes[3].set_title(f"Fascia Boundary\nMean Thickness: {thickness_info['mean_thickness_mm']:.2f} mm")
                axes[3].axis("off")

                save_path = os.path.join(output_dir, f"eval_{filenames[0]}")
                plt.tight_layout()
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

    mean_dice = np.mean(total_dice)
    mean_iou = np.mean(total_iou)
    print(f"\n================ Test Results ================")
    print(f"Mean Dice Score : {mean_dice:.4f}")
    print(f"Mean IoU        : {mean_iou:.4f}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"==============================================")


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    evaluate_model(cfg_file)
