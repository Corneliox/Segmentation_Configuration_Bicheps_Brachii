import os
import sys
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from model import build_model
from utils import postprocess_mask, calculate_muscle_thickness


def analyze_all_trials(config_path="configs/config.yaml"):
    """
    Runs segmentation inference on all extracted ultrasound frames across all trials,
    grouping and exporting results strictly separated by:
    1. Condition (5rm vs 10rm)
    2. Acquisition Takes (Trial 1, Trial 2, ..., Trial N)
    3. Time-Series Muscle Thickness Curves (Tracking thickness over time in seconds)
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device_name = cfg["project"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = os.path.dirname(base_dir)

    # Read manifest if available
    manifest_path = os.path.join(parent_dir, "dataset", "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Please run convert_bin2dataset.m first.")

    manifest_df = pd.read_csv(manifest_path)
    group_col = "Subject" if "Subject" in manifest_df.columns else "Condition"
    print(f"Loaded manifest with {len(manifest_df)} total frames across groups: {manifest_df[group_col].unique()}")

    # Read spatial calibration (mm per pixel)
    calib_file = os.path.join(parent_dir, "dataset", "spatial_calibration.csv")
    pixel_spacing_mm = 0.1757
    if os.path.exists(calib_file):
        try:
            calib_df = pd.read_csv(calib_file)
            pixel_spacing_mm = float(calib_df["MM_per_pixel"].iloc[0])
            print(f"Physical scale: 1 pixel = {pixel_spacing_mm:.4f} mm")
        except Exception as e:
            print(f"Warning reading calibration: {e}")

    # Load Model
    checkpoint_path = cfg["inference"].get("checkpoint_path", "./weights/best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_base_dir = os.path.join(base_dir, "outputs", "trial_analysis")
    os.makedirs(output_base_dir, exist_ok=True)

    results_records = []

    # Group by Subject/Condition and Trial
    grouped = manifest_df.groupby([group_col, "Trial_Name"])

    print(f"\nAnalyzing {len(grouped)} unique acquisition trials...")

    for (cond, trial_name), group_df in grouped:
        print(f"\n--> Processing [{cond}] Take: {trial_name} ({len(group_df)} frames) ...")
        
        trial_thickness_series = []
        trial_timestamps = []

        for _, row in tqdm(group_df.iterrows(), total=len(group_df), desc=f"{cond}-{trial_name}"):
            img_rel_path = row["Image_Rel_Path"]
            img_abs_path = os.path.join(parent_dir, "dataset", img_rel_path)
            
            if not os.path.exists(img_abs_path):
                continue

            img = cv2.imread(img_abs_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Normalize & prepare tensor
            h, w = img.shape
            tensor_in = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
            tensor_in = torch.tensor(tensor_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tensor_in)
                pred_mask = (pred[0, 0].cpu().numpy() > 0.5).astype(np.float32)
                cleaned_mask = postprocess_mask(pred_mask)

            # Compute thickness metrics
            thick_data = calculate_muscle_thickness(cleaned_mask, pixel_spacing_mm=pixel_spacing_mm)
            
            mean_th = thick_data["mean_thickness_mm"]
            max_th = thick_data["max_thickness_mm"]
            t_sec = row["Timestamp_Sec"]

            trial_thickness_series.append(mean_th)
            trial_timestamps.append(t_sec)

            results_records.append({
                "Condition": cond,
                "Trial_Name": trial_name,
                "Source_Bin": row["Source_Bin"],
                "Slice_Index": row["Slice_Index"],
                "Timestamp_Sec": t_sec,
                "Mean_Thickness_mm": mean_th,
                "Max_Thickness_mm": max_th
            })

        # Plot time series curve for this acquisition take
        if len(trial_timestamps) > 0:
            cond_out_dir = os.path.join(output_base_dir, cond)
            os.makedirs(cond_out_dir, exist_ok=True)

            plt.figure(figsize=(10, 4))
            plt.plot(trial_timestamps, trial_thickness_series, 'b-', linewidth=1.5, label=f'{trial_name}')
            plt.title(f"Biceps Brachii Thickness Over Time | {cond} - {trial_name}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Muscle Thickness (mm)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            
            plot_path = os.path.join(cond_out_dir, f"thickness_curve_{trial_name}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()

    # Save complete detailed CSV
    results_df = pd.DataFrame(results_records)
    results_csv_path = os.path.join(output_base_dir, "muscle_thickness_all_trials.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved complete trial metrics to: {results_csv_path}")

    # Generate Comparative Condition Plot (5rm vs 10rm summary)
    plt.figure(figsize=(12, 5))
    for cond in manifest_df["Condition"].unique():
        cond_data = results_df[results_df["Condition"] == cond]
        if not cond_data.empty:
            plt.plot(cond_data["Timestamp_Sec"], cond_data["Mean_Thickness_mm"], '.', alpha=0.3, label=f"Condition: {cond}")
    
    plt.title("Biceps Brachii Muscle Dynamics Comparison: 5RM vs 10RM")
    plt.xlabel("Timestamp (sec)")
    plt.ylabel("Mean Muscle Thickness (mm)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    comp_plot_path = os.path.join(output_base_dir, "comparison_5rm_vs_10rm.png")
    plt.tight_layout()
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()
    print(f"Saved comparative 5RM vs 10RM graph to: {comp_plot_path}")


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    analyze_all_trials(cfg_file)
