"""
Roboflow Annotation Restorer & Credential Reconciler
===================================================
Script untuk:
1. Membaca database kredensial lokal (local_credentials_mapping.json).
2. Memindai folder hasil unduhan dari Roboflow (YOLO segmentation txt, COCO json, atau mask).
3. Menghapus string mutasi/hash bawaan Roboflow (misal .rf.3ab4f...).
4. Menyatukan kembali label/anotasi ke identitas subjek dan file gambar aslinya.
"""

import os
import re
import sys
import json
import shutil
import argparse
from pathlib import Path

BASE_DIR = Path(r"D:\~Ideas n Innovation\~~Taiwan\AU\Probefix Segmentation\Probefix")
MAPPING_JSON = BASE_DIR / "Segmentation_Configuration" / "local_credentials_mapping.json"
DEFAULT_DOWNLOAD_DIR = BASE_DIR / "dataset" / "roboflow_downloaded"
RESTORE_OUTPUT_DIR = BASE_DIR / "dataset" / "restored_annotations"

def parse_args():
    parser = argparse.ArgumentParser(description="Restore downloaded Roboflow annotations to original filenames.")
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR), help="Path ke folder dataset hasil unduhan Roboflow")
    parser.add_argument("--output-dir", default=str(RESTORE_OUTPUT_DIR), help="Path folder output hasil restorasi")
    return parser.parse_args()

def extract_base_alias(filename: str):
    """
    Ekstrak ID alias asli dari nama file mutasi Roboflow.
    Contoh:
    'sub_A_seq1_0001_png.rf.3a8b4c7d.txt' -> 'sub_A_seq1_0001.png'
    'sub_B_seq1_0025_jpg.rf.9f8e7d.png'  -> 'sub_B_seq1_0025.png'
    'sub_A_seq1_0001.txt'                 -> 'sub_A_seq1_0001.png'
    """
    pattern = r"(sub_[A-Z]_seq\d+_\d{4})"
    match = re.search(pattern, filename)
    if match:
        return match.group(1) + ".png"
    return None

def main():
    args = parse_args()
    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)

    print("=" * 65)
    print(" REKONSILIASI KREDENSIAL & RESTORASI ANOTASI ROBOFLOW")
    print("=" * 65)

    if not MAPPING_JSON.exists():
        print(f"[ERROR] Database kredensial tidak ditemukan: {MAPPING_JSON}")
        print("Pastikan Anda sudah menjalankan 'roboflow_anonymize_and_upload.py' sebelumnya.")
        sys.exit(1)

    with open(MAPPING_JSON, "r", encoding="utf-8") as f:
        mapping_db = json.load(f)

    print(f"[INFO] Database kredensial dimuat: {len(mapping_db)} entri terdaftar.")

    if not download_dir.exists():
        print(f"[PERINGATAN] Folder download belum ada: {download_dir}")
        print(f"Silakan ekstrak zip dataset hasil unduhan Roboflow ke folder tersebut:")
        print(f"  {download_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    restored_count = 0

    # 1. Cari file anotasi (txt, json, xml, png masks)
    for root, _, files in os.walk(download_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".txt", ".json", ".xml", ".png", ".jpg"]:
                alias_key = extract_base_alias(fname)
                if alias_key and alias_key in mapping_db:
                    info = mapping_db[alias_key]
                    subject = info["subject"]
                    orig_filename = info["original_filename"]
                    orig_stem = Path(orig_filename).stem

                    # Tentukan target output folder per subjek
                    target_subject_dir = output_dir / subject
                    target_subject_dir.mkdir(parents=True, exist_ok=True)

                    # Simpan nama file baru sesuai nama frame asli (contoh: frame_0001.txt)
                    restored_fname = f"{orig_stem}{ext}"
                    src_file = Path(root) / fname
                    dst_file = target_subject_dir / restored_fname

                    shutil.copy2(src_file, dst_file)
                    restored_count += 1
                    print(f"[RESTORE] {fname} -> {subject}/{restored_fname}")

    print("\n" + "=" * 65)
    print(f"[SELESAI] Total {restored_count} file berhasil direkonsiliasi dan disimpan di:")
    print(f"  {output_dir}")
    print("=" * 65)

if __name__ == "__main__":
    main()
