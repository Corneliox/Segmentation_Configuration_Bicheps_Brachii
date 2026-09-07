"""
=====================================================================
convert_all.py (Ultrasound Data Processor - Clean MATLAB Gilang Method)
=====================================================================
Uses Gilang's native clean power compression:
   a3 = abs(RF).^0.25;
   a4 = imgaussfilt(a3, 1.0);
Produces ultra-clean, noise-free, smooth B-mode slices and panoramas.
=====================================================================
"""

import os
import glob
import struct
import numpy as np
import cv2
import pandas as pd


def read_bin_exact(bin_path):
    with open(bin_path, 'rb') as f:
        hdr_id = f.read(6).decode('ascii', errors='ignore')
        num_header_els = 11 if hdr_id.startswith(('RF0001', 'RF0002', 'RF0003', 'RF0004')) else 7
        
        f.seek(6)
        hdr = struct.unpack(f'<{num_header_els}i', f.read(num_header_els * 4))
        num_samples = hdr[6]
        num_lines = hdr[7]
        source_id = hdr[3]
        sampling_ns = hdr[8]
        fps = hdr[5] / 100.0 if hdr[5] > 0 else 120.0
        
        beam_bytes = f.read(3 * num_lines * 4)
        if len(beam_bytes) == 3 * num_lines * 4:
            beams = struct.unpack(f'<{3 * num_lines}i', beam_bytes)
            xs = [beams[3*i] / 10000.0 for i in range(num_lines)]
            width_mm = (max(xs) - min(xs)) * 10.0
        else:
            width_mm = 38.4 if num_lines >= 64 else 9.9

        c_sound = 1540.0
        dz_mm = (c_sound * (sampling_ns * 1e-9) / 2.0) * 1000.0
        depth_mm = num_samples * dz_mm

        frame_header_bytes = (num_header_els * 4) + (3 * num_lines * 4) + (num_lines * 4 if hdr_id.startswith(('RF0003', 'RF0004')) else 0)
        data_elements = num_samples * num_lines * 2 if source_id == 4 else num_samples * num_lines
        frame_data_bytes = data_elements * 2
        frame_stride = frame_header_bytes + frame_data_bytes

        f.seek(0, 2)
        total_size = f.tell()
        num_frames = (total_size - 6) // frame_stride

        frames = []
        f.seek(6)
        while f.tell() < total_size:
            hdr_b = f.read(num_header_els * 4)
            if len(hdr_b) < num_header_els * 4: break
            
            b_b = f.read(3 * num_lines * 4)
            if len(b_b) < 3 * num_lines * 4: break
            
            if hdr_id.startswith(('RF0003', 'RF0004')):
                ts_b = f.read(num_lines * 4)
                if len(ts_b) < num_lines * 4: break
                
            raw_b = f.read(frame_data_bytes)
            if len(raw_b) < frame_data_bytes: break
            
            arr = np.frombuffer(raw_b, dtype=np.int16)
            I_comp = arr[:num_samples * num_lines] if source_id == 4 else arr
            mat = np.reshape(I_comp, (num_samples, num_lines), order='F')
            frames.append(mat)

        meta = {
            "num_samples": num_samples,
            "num_lines": num_lines,
            "depth_mm": depth_mm,
            "width_mm": width_mm,
            "fps": fps,
            "dz_mm": dz_mm,
            "num_frames": len(frames)
        }
        return frames, meta


def process_trial_clean_gilang(bin_path, subject_name, trial_tag, output_base):
    print(f"\n--> Memproses Subject [{subject_name}] File: {os.path.basename(bin_path)} (Clean Gilang Method) ...")
    frames, meta = read_bin_exact(bin_path)
    
    if not frames:
        print(f"    [Error] Tidak ada frame yang terbaca dari {bin_path}")
        return []

    num_samples = meta["num_samples"]
    num_lines = meta["num_lines"]
    depth_mm = meta["depth_mm"]
    width_mm = meta["width_mm"]
    fps = meta["fps"]
    num_slices = len(frames)

    aspect_ratio = width_mm / depth_mm
    target_h = 512
    target_w = int(round(target_h * aspect_ratio))
    center_col = num_lines // 2

    print(f"    Dimensi Akustik : {num_samples} sampel x {num_lines} scanlines")
    print(f"    Dimensi Fisik   : Kedalaman {depth_mm:.1f} mm x Lebar {width_mm:.1f} mm (Aspect Ratio = {aspect_ratio:.3f})")
    print(f"    Total Frame     : {num_slices} frame ({num_slices / fps:.1f} detik @ {fps:.1f} FPS)")
    print(f"    Ukuran Satuan   : {target_w} px lebar x {target_h} px tinggi (Clean Noise-Free)")

    img_dir = os.path.join(output_base, "images", subject_name, trial_tag)
    pano_dir = os.path.join(output_base, "horizontal_stitched_filmstrip", subject_name)
    gilang_dir = os.path.join(output_base, "stitched_gilang", subject_name)
    mask_dir = os.path.join(output_base, "masks", subject_name, trial_tag)

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(pano_dir, exist_ok=True)
    os.makedirs(gilang_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    records = []
    raw_scanlines = []
    a2_gilang = np.zeros((num_samples, num_slices), dtype=np.float32)

    for slice_idx, fr in enumerate(frames, 1):
        a2_gilang[:, slice_idx - 1] = fr[:, center_col]

        # -------------------------------------------------------------
        # METODE BERSIH ASLI GILANG (Power Compression .^0.25 + Gaussian Smoothing)
        # -------------------------------------------------------------
        a3 = np.abs(fr).astype(np.float32) ** 0.25
        a3_min = np.min(a3)
        a3_max = np.max(a3)
        a3_norm = (a3 - a3_min) / (a3_max - a3_min + 1e-9)
        
        # Filter Gaussian 1.0 (Membersihkan noise dan memperhalus gradien jaringan)
        a4 = cv2.GaussianBlur(a3_norm, (0, 0), 1.0)
        
        # Isotropic Scan Conversion (512 x 219 px)
        b_iso = cv2.resize(a4, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        b_uint8 = (np.clip(b_iso, 0, 1) * 255.0).astype(np.uint8)

        img_fname = f"frame_{slice_idx:04d}.png"
        cv2.imwrite(os.path.join(img_dir, img_fname), b_uint8)
        
        # Raw uint8 for filmstrip
        b_raw = (np.clip(a4, 0, 1) * 255.0).astype(np.uint8)
        raw_scanlines.append(b_raw)

        records.append({
            "Subject": subject_name,
            "Trial_Name": trial_tag,
            "Source_Bin": os.path.basename(bin_path),
            "Slice_Index": slice_idx,
            "Timestamp_Sec": round((slice_idx - 1) / fps, 3),
            "Image_Rel_Path": os.path.join("images", subject_name, trial_tag, img_fname),
            "Panorama_Rel_Path": os.path.join("horizontal_stitched_filmstrip", subject_name, f"panorama_raw_{trial_tag}.png"),
            "Gilang_CSV_Rel_Path": os.path.join("stitched_gilang", subject_name, f"{subject_name}_{trial_tag}.csv")
        })

    # Save Merged Raw Scanlines Filmstrip
    raw_filmstrip = np.hstack(raw_scanlines)
    pano_fname = f"panorama_raw_{trial_tag}.png"
    cv2.imwrite(os.path.join(pano_dir, pano_fname), raw_filmstrip)
    print(f"    [OK] Panorama Filmstrip : {pano_fname} ({raw_filmstrip.shape[1]} px x {raw_filmstrip.shape[0]} px)")

    # Save Gilang M-Mode & CSV
    csv_fname = f"{subject_name}_{trial_tag}.csv"
    np.savetxt(os.path.join(gilang_dir, csv_fname), a2_gilang, delimiter=",", fmt="%.4f")
    
    a3_gilang = np.abs(a2_gilang) ** 0.5
    p_high = np.percentile(a3_gilang, 99.5)
    p_low = np.percentile(a3_gilang, 1.0)
    a3_norm = np.clip((a3_gilang - p_low) / (p_high - p_low + 1e-9), 0, 1)
    cv2.imwrite(os.path.join(gilang_dir, f"stitched_gilang_{subject_name}_{trial_tag}.png"), (a3_norm * 255).astype(np.uint8))
    print(f"    [OK] Metode Gilang CSV  : {csv_fname} & stitched preview tersimpan")

    return records, meta


def run_full_conversion():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.dirname(script_dir)
    parent_dir = os.path.dirname(proj_root)
    workspace_root = os.path.dirname(parent_dir)

    possible_roots = [
        os.path.join(parent_dir, "Probefix - Sebastian"),
        os.path.join(workspace_root, "Probefix - Sebastian"),
        r"D:\~Ideas n Innovation\~~Taiwan\AU\Probefix Segmentation\Probefix - Sebastian"
    ]
    
    sebastian_dir = None
    for p in possible_roots:
        if os.path.exists(p):
            sebastian_dir = p
            break

    if not sebastian_dir:
        print(f"[Error] Folder 'Probefix - Sebastian' tidak ditemukan!")
        return

    output_base = os.path.join(parent_dir, "dataset")
    os.makedirs(output_base, exist_ok=True)

    print("====================================================================")
    print("   CLEAN ULTRASOUND DATASET CONVERTER (METODE MATLAB GILANG)        ")
    print("====================================================================")

    bin_files = sorted(glob.glob(os.path.join(sebastian_dir, "**", "*.bin"), recursive=True))
    all_records = []
    last_meta = None

    for bin_p in bin_files:
        rel_dir = os.path.relpath(os.path.dirname(bin_p), sebastian_dir)
        subject_name = rel_dir.replace("\\", "_").replace("/", "_").replace(" ", "_")
        if subject_name in ("", "."):
            subject_name = "Sebastian"
        
        fname = os.path.basename(bin_p)
        raw_name = os.path.splitext(fname)[0].replace(" ", "_")
        trial_tag = f"trial_{raw_name}"

        rec, meta = process_trial_clean_gilang(bin_p, subject_name, trial_tag, output_base)
        if rec:
            all_records.extend(rec)
            last_meta = meta

    if all_records:
        df = pd.DataFrame(all_records)
        manifest_path = os.path.join(output_base, "dataset_manifest.csv")
        df.to_csv(manifest_path, index=False)
        print(f"\n[OK] Master Manifest tersimpan di: {manifest_path} ({len(all_records)} total frames)")

    if last_meta:
        mm_per_pixel = last_meta["depth_mm"] / 512.0
        calib_df = pd.DataFrame([{
            "Depth_mm": last_meta["depth_mm"],
            "Width_mm": last_meta["width_mm"],
            "Height_px": 512,
            "Width_px": int(round(512 * (last_meta["width_mm"] / last_meta["depth_mm"]))),
            "MM_per_pixel": mm_per_pixel,
            "Probe": "L12-5N40-A4",
            "Num_Scanlines": last_meta["num_lines"]
        }])
        calib_path = os.path.join(output_base, "spatial_calibration.csv")
        calib_df.to_csv(calib_path, index=False)
        print(f"[OK] Kalibrasi Skala Fisik: 1 px = {mm_per_pixel:.4f} mm ({calib_path})")

    print("\n====================================================================")
    print("      SELURUH KONVERSI BERSIH (METODE GILANG) SELESAI!              ")
    print("====================================================================")


if __name__ == "__main__":
    run_full_conversion()
