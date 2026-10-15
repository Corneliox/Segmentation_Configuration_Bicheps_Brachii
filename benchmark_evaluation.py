"""
=====================================================================
COMPREHENSIVE BENCHMARK EVALUATION SCRIPT FOR BICEPS BRACHII U-NET
Computes:
1. Overlap Metrics: Dice Coefficient (F1), IoU (Jaccard)
2. Classification Metrics: Pixel Accuracy, Precision, Recall, Specificity
3. Error Metrics: MAE, MSE, BCE Loss, Relative Area Error (RAE)
4. Clinical Biomechanics: Muscle Thickness Error (mm), Pearson r, R^2
5. Temporal Visualizations & Metric Curves across Video Frames
=====================================================================
"""

import os
import sys
import glob
import time
import yaml
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure Windows terminal supports UTF-8 characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.model import build_model
from src.utils import calculate_muscle_thickness


def run_benchmark():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    cfg_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"================================================================")
    print(f"🔬 BICEPS BRACHII U-NET COMPREHENSIVE BENCHMARK EVALUATION")
    print(f"================================================================")
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    img_base = os.path.join(parent_dir, "dataset", "images")
    mask_base = os.path.join(parent_dir, "dataset", "masks")
    weights_path = os.path.join(base_dir, "weights", "interactive_unet.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(base_dir, "weights", "best_model.pth")

    print(f"Model Checkpoint Path: {weights_path}")
    print(f"Image Directory      : {img_base}")
    print(f"Mask Directory       : {mask_base}")

    # Load Model
    model = build_model(cfg).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Model successfully loaded and set to eval mode.\n")

    # Collect matched image-mask pairs
    mask_files = sorted(glob.glob(os.path.join(mask_base, "**", "frame_*.png"), recursive=True))
    pairs = []
    for mf in mask_files:
        rel = os.path.relpath(mf, mask_base)
        img_f = os.path.join(img_base, rel)
        if os.path.exists(img_f):
            pairs.append((img_f, mf, rel))

    total_pairs = len(pairs)
    print(f"Found {total_pairs} valid Ground Truth mask pairs on disk.")
    if total_pairs == 0:
        print("[ERROR] No matched image-mask pairs found. Exiting.")
        return

    # Physical calibration: 1 pixel = 0.1757 mm (Depth = 89.96 mm / 512 px)
    pixel_spacing_mm = cfg["inference"].get("pixel_spacing_mm", 0.1757)

    results = []
    bce_fn = nn.BCELoss()

    print(f"Evaluating all {total_pairs} frames with PyTorch CUDA...")

    for idx, (img_path, mask_path, rel_path) in enumerate(pairs):
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None or gt_mask is None:
            continue

        h, w = img_gray.shape
        tw = int(round(w / 32.0)) * 32
        th = int(round(h / 32.0)) * 32

        # Preprocess
        resized_img = cv2.resize(img_gray, (tw, th), interpolation=cv2.INTER_LINEAR)
        x = (resized_img.astype(np.float32) / 255.0 - 0.5) / 0.5
        t_in = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out_tensor = model(t_in)
            prob_map = out_tensor.squeeze().cpu().numpy()

        # Resize probability back to original resolution
        prob_orig = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_bin = (prob_orig > 0.5).astype(np.uint8)
        gt_bin = (gt_mask > 127).astype(np.uint8)

        # Confusion matrix elements
        tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
        fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
        fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
        tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))
        total_pixels = tp + fp + fn + tn

        # Overlap & classification metrics
        eps = 1e-7
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        pixel_acc = (tp + tn) / total_pixels
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)

        # Error metrics
        mae = float(np.mean(np.abs(prob_orig - gt_bin)))
        mse = float(np.mean((prob_orig - gt_bin) ** 2))
        
        # BCE Loss
        prob_clamped = np.clip(prob_orig, 1e-7, 1.0 - 1e-7)
        bce_loss = float(-np.mean(gt_bin * np.log(prob_clamped) + (1.0 - gt_bin) * np.log(1.0 - prob_clamped)))

        # Relative Area Error
        gt_area = float(max(1, np.sum(gt_bin)))
        pred_area = float(np.sum(pred_bin))
        rae = abs(pred_area - gt_area) / gt_area

        # Clinical muscle thickness
        gt_thick = calculate_muscle_thickness(gt_bin, pixel_spacing_mm=pixel_spacing_mm)["mean_thickness_mm"]
        pred_thick = calculate_muscle_thickness(pred_bin, pixel_spacing_mm=pixel_spacing_mm)["mean_thickness_mm"]
        thick_err_mm = abs(pred_thick - gt_thick)

        results.append({
            "frame_idx": idx + 1,
            "filename": os.path.basename(img_path),
            "relative_path": rel_path,
            "dice": dice,
            "iou": iou,
            "pixel_accuracy": pixel_acc,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "mae": mae,
            "mse": mse,
            "bce_loss": bce_loss,
            "relative_area_error": rae,
            "gt_thickness_mm": gt_thick,
            "pred_thickness_mm": pred_thick,
            "thickness_error_mm": thick_err_mm,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })

    df = pd.DataFrame(results)

    # Save CSV output
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "evaluation_benchmark_all_frames.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved detailed per-frame metrics to: {csv_path}")

    # Compute Global & Summary Statistics
    mean_dice = df["dice"].mean()
    std_dice = df["dice"].std()
    median_dice = df["dice"].median()

    mean_iou = df["iou"].mean()
    std_iou = df["iou"].std()
    median_iou = df["iou"].median()

    mean_acc = df["pixel_accuracy"].mean()
    mean_prec = df["precision"].mean()
    mean_rec = df["recall"].mean()
    mean_spec = df["specificity"].mean()

    mean_mae = df["mae"].mean()
    mean_bce = df["bce_loss"].mean()
    mean_rae = df["relative_area_error"].mean()

    mean_thick_err = df["thickness_error_mm"].mean()
    std_thick_err = df["thickness_error_mm"].std()

    # Pearson correlation for muscle thickness
    pearson_r = float(df["gt_thickness_mm"].corr(df["pred_thickness_mm"]))
    r2_score = float(pearson_r ** 2)

    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print(" 📊 HASIL EVALUASI AKURASI & ERROR MATRIX MODEL U-NET")
    print("=" * 65)
    print(f" Jumlah Frame Diuji           : {len(df)} frame teranotasi")
    print(f" Waktu Komputasi Evaluasi     : {elapsed:.2f} detik (~{len(df)/elapsed:.1f} FPS)")
    print("-" * 65)
    print(f" [OVERLAP METRICS]")
    print(f" • Dice Coefficient (F1-Score): {mean_dice*100:.2f}% ± {std_dice*100:.2f}% (Median: {median_dice*100:.2f}%)")
    print(f" • IoU (Jaccard Index)        : {mean_iou*100:.2f}% ± {std_iou*100:.2f}% (Median: {median_iou*100:.2f}%)")
    print("-" * 65)
    print(f" [CLASSIFICATION METRICS]")
    print(f" • Pixel Accuracy             : {mean_acc*100:.2f}%")
    print(f" • Precision (Pos Pred Value) : {mean_prec*100:.2f}%")
    print(f" • Recall / Sensitivity       : {mean_rec*100:.2f}%")
    print(f" • Specificity (True Neg Rate): {mean_spec*100:.2f}%")
    print("-" * 65)
    print(f" [ERROR MATRIX & LOSS]")
    print(f" • Mean Absolute Error (MAE)  : {mean_mae:.4f}")
    print(f" • Mean Binary Cross-Entropy  : {mean_bce:.4f}")
    print(f" • Mean Dice Loss (1 - Dice)  : {1.0 - mean_dice:.4f}")
    print(f" • Relative Area Error (RAE)  : {mean_rae*100:.2f}%")
    print("-" * 65)
    print(f" [CLINICAL BIOMECHANICS METRICS (BICEPS THICKNESS)]")
    print(f" • Mean Muscle Thickness Error: {mean_thick_err:.2f} mm ± {std_thick_err:.2f} mm")
    print(f" • Pearson Correlation (r)    : {pearson_r:.4f}")
    print(f" • Coefficient of Det (R²)    : {r2_score:.4f}")
    print("=" * 65)

    # Generate Visualization Figure
    fig = plt.figure(figsize=(18, 12), facecolor="#f8f9fa")
    
    # Subplot 1: Dice & IoU Distribution (Boxplot + KDE)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(df["dice"] * 100, bins=30, color="#007acc", alpha=0.7, edgecolor="black", label=f"Dice (Mean: {mean_dice*100:.1f}%)")
    ax1.hist(df["iou"] * 100, bins=30, color="#ff9800", alpha=0.6, edgecolor="black", label=f"IoU (Mean: {mean_iou*100:.1f}%)")
    ax1.set_title("Distribusi Dice Score & IoU Overlap", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Score (%)", fontsize=10)
    ax1.set_ylabel("Frekuensi Frame", fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Temporal Tracking of Dice & IoU across Frame Index
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(df["frame_idx"], df["dice"] * 100, color="#007acc", label="Dice (%)", alpha=0.8, linewidth=1.2)
    ax2.plot(df["frame_idx"], df["iou"] * 100, color="#ff9800", label="IoU (%)", alpha=0.8, linewidth=1.2)
    ax2.axhline(mean_dice * 100, color="blue", linestyle="--", label=f"Mean Dice ({mean_dice*100:.1f}%)")
    ax2.set_title("Stabilitas Akurasi Sepanjang Video Trial (Temporal Tracking)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frame Index (Waktu Berjalan)", fontsize=10)
    ax2.set_ylabel("Overlap Score (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Clinical Muscle Thickness Tracking (GT vs Prediction)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(df["frame_idx"], df["gt_thickness_mm"], color="#2e7d32", label="Ground Truth Thickness (mm)", linewidth=1.8)
    ax3.plot(df["frame_idx"], df["pred_thickness_mm"], color="#d32f2f", linestyle="--", label="U-Net Predicted Thickness (mm)", linewidth=1.5)
    ax3.fill_between(df["frame_idx"], df["gt_thickness_mm"], df["pred_thickness_mm"], color="red", alpha=0.15, label="Thickness Absolute Error")
    ax3.set_title(f"Estimasi Ketebalan Otot Biceps (r = {pearson_r:.3f}, R² = {r2_score:.3f})", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Frame Index", fontsize=10)
    ax3.set_ylabel("Muscle Thickness (mm)", fontsize=10)
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Summary Error Matrix Bar Chart
    ax4 = fig.add_subplot(2, 2, 4)
    metric_names = ["Dice\n(F1)", "IoU\n(Jaccard)", "Pixel\nAcc", "Precision\n(PPV)", "Recall\n(Sens)", "Specificity\n(TNR)"]
    metric_values = [mean_dice * 100, mean_iou * 100, mean_acc * 100, mean_prec * 100, mean_rec * 100, mean_spec * 100]
    colors = ["#007acc", "#ff9800", "#4caf50", "#9c27b0", "#00bcd4", "#607d8b"]
    bars = ax4.bar(metric_names, metric_values, color=colors, edgecolor="black", width=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax4.set_title("Ringkasan Metrik Akurasi Segmentasi U-Net", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Persentase (%)", fontsize=10)
    ax4.set_ylim(0, 115)
    ax4.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "benchmark_metrics_summary.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved benchmark charts to: {chart_path}")

    # Also save visual comparison of Best, Median, and Worst frame
    best_idx = df["dice"].idxmax()
    median_idx = (df["dice"] - median_dice).abs().idxmin()
    worst_idx = df["dice"].idxmin()

    sample_indices = [
        ("Best Frame", best_idx),
        ("Median Frame", median_idx),
        ("Worst Frame", worst_idx)
    ]

    fig_samples, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor="#1a1a1a")
    for r, (label, s_idx) in enumerate(sample_indices):
        row_data = df.iloc[s_idx]
        img_p, msk_p, _ = pairs[s_idx]
        img_raw = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        gt_raw = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        
        # Predict
        tw = int(round(img_raw.shape[1] / 32.0)) * 32
        th = int(round(img_raw.shape[0] / 32.0)) * 32
        x = (cv2.resize(img_raw, (tw, th)).astype(np.float32) / 255.0 - 0.5) / 0.5
        t_in = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pr = model(t_in).squeeze().cpu().numpy()
        pred_raw = (cv2.resize(pr, (img_raw.shape[1], img_raw.shape[0])) > 0.5).astype(np.uint8)

        # Draw overlays
        gt_overlay = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        pred_overlay = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        diff_overlay = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)

        gt_overlay[gt_raw > 127] = [0, 255, 0]
        pred_overlay[pred_raw > 0] = [255, 140, 0]

        # Difference: Green = TP, Red = FP, Blue = FN
        diff_overlay[(pred_raw == 1) & (gt_raw > 127)] = [0, 255, 0]      # TP
        diff_overlay[(pred_raw == 1) & (gt_raw <= 127)] = [255, 0, 0]     # FP
        diff_overlay[(pred_raw == 0) & (gt_raw > 127)] = [0, 100, 255]    # FN

        axes[r, 0].imshow(img_raw, cmap="gray")
        axes[r, 0].set_title(f"{label}: {row_data['filename']}\nInput US B-Mode", color="white", fontsize=10)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(gt_overlay)
        axes[r, 1].set_title(f"Ground Truth Mask\nThick: {row_data['gt_thickness_mm']:.2f}mm", color="#00e676", fontsize=10)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(pred_overlay)
        axes[r, 2].set_title(f"U-Net Prediction\nDice: {row_data['dice']*100:.1f}%, IoU: {row_data['iou']*100:.1f}%", color="#ff9800", fontsize=10)
        axes[r, 2].axis("off")

        axes[r, 3].imshow(diff_overlay)
        axes[r, 3].set_title(f"Error Map\nGreen=TP, Red=FP, Blue=FN", color="#00e5ff", fontsize=10)
        axes[r, 3].axis("off")

    plt.tight_layout()
    samples_chart_path = os.path.join(output_dir, "benchmark_sample_comparisons.png")
    plt.savefig(samples_chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig_samples)
    print(f"✓ Saved sample qualitative comparisons to: {samples_chart_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_benchmark()
