"""
=====================================================================
convert_all.py (Exact Python port of import_RFdata2MATLAB.m)
=====================================================================
Reads binary ArtUs files without any frame offset drift:
1. HASIL SATUAN: Isotropic 2D Slices (512x89 px, DR 78 dB) in dataset/images/
2. HASIL AKHIR BERDERET: Raw 33-scanline slices concatenated horizontally
   (2959 depth x [N * 33] width) in dataset/horizontal_stitched_filmstrip/
3. GILANG CSV & M-MODE: dataset/stitched_gilang/
=====================================================================
"""

import os
import glob
import struct
import numpy as np
import cv2
import pandas as pd


def read_bin_exact(bin_path):
    """
    Exact implementation of import_RFdata2MATLAB.m in Python.
    Reads per-frame header (44 bytes), beam positions (396 bytes),
    and timestamps (132 bytes) on EVERY frame to prevent any offset drift.
    """
    with open(bin_path, 'rb') as f:
        hdr_id = f.read(6).decode('ascii', errors='ignore')
        num_header_els = 11 if hdr_id.startswith(('RF0001', 'RF0002', 'RF0003', 'RF0004')) else 7
        
        f.seek(0, 2)
        total_size = f.tell()
        f.seek(6)
        
        frames = []
        fps = 120.0
        
        while f.tell() < total_size:
            # 1. Read header elements (11 int32 = 44 bytes)
            hdr_bytes = f.read(num_header_els * 4)
            if len(hdr_bytes) < num_header_els * 4:
                break
            hdr = struct.unpack(f'<{num_header_els}i', hdr_bytes)
            num_samples = hdr[6]
            num_lines = hdr[7]
            source_id = hdr[3]
            if hdr[5] > 0:
                fps = hdr[5] / 100.0
            
            # 2. Read beam coordinates (3 * num_lines int32 = 396 bytes)
            beam_bytes = f.read(3 * num_lines * 4)
            if len(beam_bytes) < 3 * num_lines * 4:
                break
            
            # 3. Read timestamps (num_lines uint32 = 132 bytes)
            if hdr_id.startswith(('RF0003', 'RF0004')):
                ts_bytes = f.read(num_lines * 4)
                if len(ts_bytes) < num_lines * 4:
                    break
            
            # 4. Read frame RF data
            size_of_frame = num_samples * num_lines * 2 if source_id == 4 else num_samples * num_lines
            raw_bytes = f.read(size_of_frame * 2)
            if len(raw_bytes) < size_of_frame * 2:
                break
            
            arr = np.frombuffer(raw_bytes, dtype=np.int16)
            I_comp = arr[:num_samples * num_lines] if source_id == 4 else arr
            mat = np.reshape(I_comp, (num_samples, num_lines), order='F')
            frames.append(mat)
            
        return frames, (num_samples, num_lines), fps


def process_bin(bin_path, cond_name, raw_name, trial_tag, output_img_base, output_pano_base, output_gilang_base):
    frames, (num_samples, num_lines), fps = read_bin_exact(bin_path)

    if not frames:
        return 0, None, None

    num_slices = len(frames)
    target_h, target_w = 512, 89
    dr_dB = 78.0

    trial_img_dir = os.path.join(output_img_base, cond_name, trial_tag)
    pano_dir = os.path.join(output_pano_base, cond_name)
    gilang_dir = os.path.join(output_gilang_base, cond_name)
    os.makedirs(trial_img_dir, exist_ok=True)
    os.makedirs(pano_dir, exist_ok=True)
    os.makedirs(gilang_dir, exist_ok=True)

    raw33_slices = []
    records = []
    a2_gilang = np.zeros((num_samples, num_slices), dtype=np.float32)

    for slice_idx, fr in enumerate(frames, 1):
        a2_gilang[:, slice_idx - 1] = fr[:, min(17, fr.shape[1] - 1)]

        env = np.abs(fr).astype(np.float32)
        norm = env / (np.max(env) + 1e-9)
        b_db = (20.0 * np.log10(norm + 1e-9) + dr_dB) / dr_dB
        b_db = np.clip(b_db, 0.0, 1.0)
        b_raw_uint8 = (b_db * 255).astype(np.uint8)

        # 1. HASIL SATUAN (Isotropic 512 x 89 px)
        b_iso = cv2.resize(b_db, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        img_fname = f"frame_{slice_idx:04d}.png"
        cv2.imwrite(os.path.join(trial_img_dir, img_fname), (b_iso * 255).astype(np.uint8))

        # 2. Tampung potongan mentah 33 scanlines untuk deretan horizontal
        raw33_slices.append(b_raw_uint8)

        records.append({
            "Condition": cond_name,
            "Trial_Name": trial_tag,
            "Source_Bin": os.path.basename(bin_path),
            "Slice_Index": slice_idx,
            "Timestamp_Sec": round((slice_idx - 1) / fps, 3),
            "Image_Rel_Path": os.path.join("images", cond_name, trial_tag, img_fname),
            "Raw33_Panorama_Rel_Path": os.path.join("horizontal_stitched_filmstrip", cond_name, f"panorama_raw33_{trial_tag}.png")
        })

    # 3. HASIL AKHIR BERDERET (Deretan 33 Scanlines per Slice)
    raw33_panorama = np.hstack(raw33_slices)
    pano_fname = f"panorama_raw33_{trial_tag}.png"
    cv2.imwrite(os.path.join(pano_dir, pano_fname), raw33_panorama)

    # 4. Save Gilang CSV & Stitched Preview
    csv_fname = f"{cond_name}_{raw_name}.csv"
    np.savetxt(os.path.join(gilang_dir, csv_fname), a2_gilang, delimiter=",", fmt="%.4f")
    
    a3_gilang = np.abs(a2_gilang) ** 0.5
    p_high = np.percentile(a3_gilang, 99.5)
    p_low = np.percentile(a3_gilang, 1.0)
    a3_norm = np.clip((a3_gilang - p_low) / (p_high - p_low + 1e-9), 0, 1)
    cv2.imwrite(os.path.join(gilang_dir, f"stitched_gilang_{cond_name}_{raw_name}.png"), (a3_norm * 255).astype(np.uint8))

    return num_slices, records, (pano_fname, raw33_panorama.shape)


def run_full_conversion(base_dir="e:/Probefix"):
    britty_dir = os.path.join(base_dir, "britty")
    output_img_base = os.path.join(base_dir, "dataset", "images")
    output_pano_base = os.path.join(base_dir, "dataset", "horizontal_stitched_filmstrip")
    output_gilang_base = os.path.join(base_dir, "dataset", "stitched_gilang")
    
    conditions = ["5rm", "10rm"]
    all_manifest = []

    print("====================================================================")
    print("STARTING HYBRID EXTRACTION (SATUAN 512x89 + RAW 33-LINE FILMSTRIP)")
    print("====================================================================")

    for cond in conditions:
        cond_dir = os.path.join(britty_dir, cond)
        if not os.path.exists(cond_dir): continue

        bin_files = sorted(glob.glob(os.path.join(cond_dir, "**", "*.bin"), recursive=True))
        print(f"\nCondition [{cond}]: Found {len(bin_files)} .bin files")

        for idx, bin_p in enumerate(bin_files, 1):
            subfolder = os.path.basename(os.path.dirname(bin_p))
            fname = os.path.basename(bin_p)
            raw_name = os.path.splitext(fname)[0]
            trial_tag = f"trial_{idx:02d}_{subfolder}_{raw_name}"

            print(f"  [{idx}/{len(bin_files)}] Processing: {fname} ...")
            n_frames, records, pano_info = process_bin(bin_p, cond, raw_name, trial_tag, output_img_base, output_pano_base, output_gilang_base)
            if records:
                all_manifest.extend(records)
                print(f"      -> Satuan: {n_frames} frames (512x89 px) di {trial_tag}")
                print(f"      -> Raw 33-Line Panorama: {pano_info[0]} (Ukuran: {pano_info[1][1]}px x {pano_info[1][0]}px)")

    if all_manifest:
        df = pd.DataFrame(all_manifest)
        manifest_path = os.path.join(base_dir, "dataset", "dataset_manifest.csv")
        df.to_csv(manifest_path, index=False)
        print(f"\nDataset manifest saved to: {manifest_path}")

    print("\nALL CONVERSIONS COMPLETE WITHOUT ANY DRIFT!")


if __name__ == "__main__":
    run_full_conversion()
