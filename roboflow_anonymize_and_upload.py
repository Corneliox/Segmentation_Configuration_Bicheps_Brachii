"""
Roboflow Sequential Pseudonymization & Batch Uploader
=====================================================
Pipeline untuk:
1. Menyamarkan identitas subjek dan parameter probe (Pseudonymization).
2. Mempertahankan urutan frame (sub_A_seq1_0001, sub_B_seq1_0001) agar annotator tetap nyaman.
3. Menyimpan database kredensial lokal (JSON & CSV) untuk rekonsiliasi setelah download.
4. Mengunggah gambar ke Roboflow via Terminal API.
"""

import os
import sys
import json
import csv
import glob
import shutil
import argparse
from pathlib import Path

# ==================== DEFAULT CONFIGURATION ====================
DEFAULT_API_KEY = ""        # Isi API key Roboflow Anda di sini (atau masukkan saat diminta)
DEFAULT_WORKSPACE = ""      # ID Workspace Roboflow
DEFAULT_PROJECT = ""        # ID Project Roboflow (Instance Segmentation)
DEFAULT_STRIDE = 1          # 1 = Semua frame, 5 = 1 dari 5 frame, 10 = 1 dari 10 frame

BASE_DIR = Path(r"D:\~Ideas n Innovation\~~Taiwan\AU\Probefix Segmentation\Probefix")
TARGET_DATASETS = [
    {
        "alias": "sub_A_seq1",
        "subject": "Tissa_-_10RM",
        "path": BASE_DIR / "dataset" / "images" / "Tissa_-_10RM" / "trial_14_23_12_04-09-2026_L12-5N40-A4",
        "batch_name": "Sequence_A"
    },
    {
        "alias": "sub_B_seq1",
        "subject": "Azmi_-_RM_10",
        "path": BASE_DIR / "dataset" / "images" / "Azmi_-_RM_10" / "trial_16_06_40_03-09-2026_L12-5N40-A4",
        "batch_name": "Sequence_B"
    }
]

MAPPING_JSON = BASE_DIR / "Segmentation_Configuration" / "local_credentials_mapping.json"
MAPPING_CSV = BASE_DIR / "Segmentation_Configuration" / "local_credentials_mapping.csv"
TEMP_UPLOAD_DIR = BASE_DIR / "dataset" / "temp_anonymized_upload"
# ===============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Upload pseudonimized ultrasound frames to Roboflow.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Roboflow Private API Key")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Roboflow Workspace ID")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Roboflow Project ID")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Frame stride (1=all, 5=every 5th frame)")
    parser.add_argument("--prepare-only", action="store_true", help="Hanya siapkan file & mapping tanpa upload ke Roboflow")
    return parser.parse_args()

def prepare_pseudonym_dataset(stride: int):
    print("=" * 65)
    print(" 1. PERSIAPAN ANOMISASI & DATABASE KREDENSIAL LOKAL")
    print("=" * 65)

    if TEMP_UPLOAD_DIR.exists():
        shutil.rmtree(TEMP_UPLOAD_DIR)
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    mapping_db = {}
    csv_rows = []
    total_prepared = 0

    for ds in TARGET_DATASETS:
        alias = ds["alias"]
        subject = ds["subject"]
        folder_path = ds["path"]
        batch_name = ds["batch_name"]

        if not folder_path.exists():
            print(f"[ERROR] Folder tidak ditemukan: {folder_path}")
            continue

        files = sorted(list(folder_path.glob("*.png")))
        selected_files = files[::stride]
        print(f"\n[Folder] {subject}")
        print(f"  Total file asli: {len(files)} | Terpilih (stride={stride}): {len(selected_files)}")

        for idx, orig_path in enumerate(selected_files, start=1):
            anon_name = f"{alias}_{idx:04d}.png"
            dest_path = TEMP_UPLOAD_DIR / anon_name

            # Gunakan hard link jika didukung (instan & hemat disk), fallback ke copy
            try:
                os.link(orig_path, dest_path)
            except Exception:
                shutil.copyfile(orig_path, dest_path)

            record = {
                "anonymized_name": anon_name,
                "original_filename": orig_path.name,
                "original_path": str(orig_path),
                "subject": subject,
                "batch_name": batch_name,
                "frame_index": idx,
                "stride_used": stride
            }
            mapping_db[anon_name] = record
            csv_rows.append(record)
            total_prepared += 1

    # Simpan JSON Mapping
    with open(MAPPING_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping_db, f, indent=4)

    # Simpan CSV Mapping
    with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "anonymized_name", "original_filename", "subject", "batch_name", "frame_index", "stride_used", "original_path"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[BERHASIL] Total {total_prepared} gambar telah disiapkan di:\n  {TEMP_UPLOAD_DIR}")
    print(f"[BERHASIL] Kredensial lokal tersimpan di:\n  {MAPPING_JSON}\n  {MAPPING_CSV}")
    return total_prepared

def upload_to_roboflow(api_key: str, workspace_id: str, project_id: str):
    print("\n" + "=" * 65)
    print(" 2. MENGUNGGAH KE ROBOFLOW")
    print("=" * 65)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] Library 'roboflow' belum terpasang. Jalankan: pip install roboflow")
        sys.exit(1)

    print(f"[INFO] Menghubungkan ke Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_id).project(project_id)

    files_to_upload = sorted(list(TEMP_UPLOAD_DIR.glob("*.png")))
    total = len(files_to_upload)

    if total == 0:
        print("[PERINGATAN] Tidak ada file di folder temporary untuk diunggah.")
        return

    # Baca mapping untuk mengambil batch name
    with open(MAPPING_JSON, "r", encoding="utf-8") as f:
        mapping_db = json.load(f)

    print(f"[INFO] Mulai upload {total} gambar ke project '{project_id}'...")
    success_count = 0
    fail_count = 0

    for idx, fpath in enumerate(files_to_upload, start=1):
        meta = mapping_db.get(fpath.name, {})
        batch_name = meta.get("batch_name", "Probefix_Batch")
        subject_alias = meta.get("anonymized_name", "").split("_")[0] + "_" + meta.get("anonymized_name", "").split("_")[1]

        try:
            project.upload(
                image_path=str(fpath),
                batch_name=batch_name,
                tag_names=[subject_alias, "probefix_biceps"],
                num_retry_uploads=3
            )
            success_count += 1
            print(f"[{idx}/{total}] SUKSES: {fpath.name}")
        except Exception as e:
            fail_count += 1
            print(f"[{idx}/{total}] GAGAL ({fpath.name}): {e}")

    print("\n" + "=" * 65)
    print(f"[SELESAI] Hasil Upload: {success_count} Berhasil, {fail_count} Gagal.")
    print("=" * 65)

def main():
    args = parse_args()

    api_key = args.api_key.strip()
    workspace = args.workspace.strip()
    project = args.project.strip()
    stride = args.stride

    # Jalankan persiapan
    total = prepare_pseudonym_dataset(stride)
    if total == 0:
        print("[ERROR] Tidak ada file yang dipersiapkan. Periksa path folder.")
        sys.exit(1)

    if args.prepare_only:
        print("\n[INFO] Mode --prepare-only aktif. Upload dilewati.")
        return

    # Jika API key kosong, tanyakan ke pengguna di terminal
    if not api_key:
        print("\n--- KREDENSIAL ROBOFLOW DIPERLUKAN ---")
        api_key = input("Masukkan Roboflow API Key: ").strip()
    if not workspace:
        workspace = input("Masukkan Roboflow Workspace ID: ").strip()
    if not project:
        project = input("Masukkan Roboflow Project ID: ").strip()

    if not (api_key and workspace and project):
        print("\n[PERINGATAN] Kredensial Roboflow belum lengkap. Proses upload dibatalkan.")
        print(f"Catatan: File lokal yang telah disamarkan tetap tersimpan di:\n  {TEMP_UPLOAD_DIR}")
        print("Anda dapat menjalankan upload kapan saja setelah mengisi kredensial.")
        return

    upload_to_roboflow(api_key, workspace, project)

if __name__ == "__main__":
    main()
