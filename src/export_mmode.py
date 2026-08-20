import os
import glob
import struct
import numpy as np
import cv2
import pandas as pd


def extract_mmode_from_bin(bin_path, dr_dB=78.0, center_col=16):
    """
    Reads .bin RF file and creates a continuous M-Mode (Time-Motion) stitched image
    by horizontally concatenating the center scanline across all time frames.
    Applies medical B-mode Logarithmic Dynamic Range Compression (DR = 78 dB).
    """
    with open(bin_path, 'rb') as f:
        hdr_id = f.read(6).decode('ascii', errors='ignore')
        if not hdr_id.startswith('RF'):
            return None, None
        
        data = struct.unpack('<11i', f.read(44))
        num_samples = data[6]  # e.g. 2959
        num_lines = data[7]    # e.g. 33
        sampling_ns = data[8]
        fps = data[5] / 100.0 if data[5] > 0 else 120.0
        
        frame_bytes = num_samples * num_lines * 2
        f.seek(0, 2)
        total_size = f.tell()
        offset = 6 + 44 + (3 * num_lines * 4)
        f.seek(offset)
        
        m_mode_columns = []
        
        while f.tell() + frame_bytes <= total_size:
            raw_bytes = f.read(frame_bytes)
            arr = np.frombuffer(raw_bytes, dtype=np.int16).reshape((num_lines, num_samples)).T
            # Extract center scanline
            col_data = np.abs(arr[:, center_col]).astype(np.float32)
            m_mode_columns.append(col_data)
            # Skip timestamps if present
            f.seek(f.tell() + num_lines * 4)

    if not m_mode_columns:
        return None, None

    # Stack horizontally: Shape = (num_samples, num_frames)
    m_mode_matrix = np.column_stack(m_mode_columns)

    # Dynamic Range Logarithmic Compression (EchoWave standard)
    env_max = np.max(m_mode_matrix)
    if env_max == 0:
        env_max = 1.0
    norm_m = m_mode_matrix / env_max
    m_mode_db = (20.0 * np.log10(norm_m + 1e-9) + dr_dB) / dr_dB
    m_mode_db = np.clip(m_mode_db, 0.0, 1.0)
    m_mode_uint8 = (m_mode_db * 255.0).astype(np.uint8)

    # Physical scaling metadata
    dz_mm = (1540.0 * (sampling_ns * 1e-9) / 2.0) * 1000.0
    total_depth_mm = num_samples * dz_mm
    duration_sec = len(m_mode_columns) / fps

    meta = {
        "num_frames": len(m_mode_columns),
        "fps": fps,
        "duration_sec": duration_sec,
        "num_samples": num_samples,
        "dz_mm": dz_mm,
        "total_depth_mm": total_depth_mm
    }

    return m_mode_uint8, meta


def generate_all_mmode(base_dir="e:/Probefix"):
    britty_dir = os.path.join(base_dir, "britty")
    out_base_dir = os.path.join(base_dir, "dataset", "mmode_stitched")
    
    conditions = ["5rm", "10rm"]
    manifest_records = []

    print("==========================================================")
    print("GENERATING HORIZONTALLY STITCHED M-MODE (TIME-MOTION) IMAGES")
    print(f"Source: {britty_dir}")
    print(f"Destination: {out_base_dir}")
    print("==========================================================")

    for cond in conditions:
        cond_dir = os.path.join(britty_dir, cond)
        if not os.path.exists(cond_dir):
            continue

        cond_out_dir = os.path.join(out_base_dir, cond)
        os.makedirs(cond_out_dir, exist_ok=True)

        bin_files = sorted(glob.glob(os.path.join(cond_dir, "**", "*.bin"), recursive=True))
        print(f"\nProcessing Condition [{cond}]: Found {len(bin_files)} .bin files")

        for idx, bin_p in enumerate(bin_files, 1):
            subfolder = os.path.basename(os.path.dirname(bin_p))
            fname = os.path.basename(bin_p)
            raw_name = os.path.splitext(fname)[0]

            trial_tag = f"take_{idx:02d}_{subfolder}_{raw_name}"
            out_img_name = f"mmode_{cond}_{trial_tag}.png"
            out_img_path = os.path.join(cond_out_dir, out_img_name)

            print(f"  [{idx}/{len(bin_files)}] Stitching: {fname} ({subfolder}) ...")
            mmode_img, meta = extract_mmode_from_bin(bin_p)

            if mmode_img is not None:
                cv2.imwrite(out_img_path, mmode_img)
                print(f"      -> Saved {out_img_name} [Size: {mmode_img.shape[0]} depth x {mmode_img.shape[1]} time | {meta['duration_sec']:.2f} sec]")

                manifest_records.append({
                    "Condition": cond,
                    "Take_ID": idx,
                    "Subfolder": subfolder,
                    "Source_File": fname,
                    "Output_File": out_img_name,
                    "Total_Frames": meta["num_frames"],
                    "Duration_Sec": round(meta["duration_sec"], 2),
                    "FPS": round(meta["fps"], 1),
                    "Total_Depth_mm": round(meta["total_depth_mm"], 2),
                    "Relative_Path": os.path.relpath(out_img_path, base_dir)
                })

    if manifest_records:
        df = pd.DataFrame(manifest_records)
        manifest_csv = os.path.join(out_base_dir, "mmode_manifest.csv")
        df.to_csv(manifest_csv, index=False)
        print(f"\nSaved M-Mode Manifest to: {manifest_csv}")

    print("\nALL M-MODE HORIZONTAL STITCHING COMPLETE!")


if __name__ == "__main__":
    generate_all_mmode()
