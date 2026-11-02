"""
=====================================================================
biceps_pipeline_studio.py
BICEPS BRACHII ULTRASOUND SEGMENTATION & BIOMECHANICS PIPELINE STUDIO
=====================================================================
UI/UX Architecture & Palette (D:\\Download Move1\\Main.png):
- #2E3532 : Deep Charcoal Slate (Calming eye-care dark canvas & structure)
- #7E9181 : Muted Sage Green (Secondary panels, fascia guides, rest marker)
- #DBC2CF : Soft Blush / Lilac (Gentle highlights, secondary labels)
- #80475E : Muted Plum / Wine (Primary contraction alerts, AI action)
- #C7EF00 : Electric Lime (Active scrub needle, peak metric, play button)

Key Capabilities:
1. End-to-End Pipeline: Raw .BIN Ingestion via MATLAB HD or Pre-extracted PNGs
2. AI Segmentation Engine: U-Net Biceps Brachii & Aponeurosis tracking
3. Kinematics Telemetry: Live synchronized muscle thickness waveforms & metrics
4. Sample Datasets & Presets Hub:
   - One-click loading of bundled/installed subjects (Azmi, Tissa)
   - Cloud downloading capability for remote collaborators
   - Ultrasound transducer specifications & custom preset importer (.json/.yaml)
5. Rounded, ergonomic medical styling & Dual Themes (Dark & Light)
6. Bilingual Localization (English default & 繁體中文 Mandarin)
=====================================================================
"""

import os
import sys
import glob
import time
import struct
import json
import shutil
import zipfile
import urllib.request
import subprocess
import yaml
import cv2
import numpy as np
import pandas as pd
import torch

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QRectF
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont,
    QCursor, QKeySequence, QIcon
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QStatusBar,
    QMessageBox, QScrollArea, QFrame, QSplitter, QShortcut,
    QFileDialog, QProgressBar, QCheckBox, QButtonGroup, QRadioButton,
    QDialog, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QGroupBox
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# =====================================================================
# PATH CONFIGURATION & PYINSTALLER PORTABLE RESOURCE MANAGEMENT
# =====================================================================
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle
    BUNDLE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    APP_DIR = os.path.dirname(sys.executable)
    CURRENT_DIR = os.path.join(BUNDLE_DIR, "src") if os.path.exists(os.path.join(BUNDLE_DIR, "src")) else BUNDLE_DIR
    PROJECT_ROOT = APP_DIR
else:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    APP_DIR = PROJECT_ROOT
    BUNDLE_DIR = PROJECT_ROOT

PARENT_ROOT = os.path.dirname(PROJECT_ROOT)

def get_resource_path(relative_path: str) -> str:
    """Retrieve absolute path to a resource, checking local app dir first then internal bundle."""
    # 1. Check in external app directory next to .exe (allows user-added presets/weights)
    app_path = os.path.join(APP_DIR, relative_path)
    if os.path.exists(app_path):
        return app_path
    # 2. Check in PyInstaller internal unpacked bundle directory (_MEIPASS)
    bundle_path = os.path.join(BUNDLE_DIR, relative_path)
    if os.path.exists(bundle_path):
        return bundle_path
    return app_path

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import build_model
from utils import postprocess_mask, calculate_muscle_thickness


# =====================================================================
# BILINGUAL LOCALIZATION (ENGLISH DEFAULT & 繁體中文 MANDARIN)
# =====================================================================
TRANSLATIONS = {
    "en": {
        "app_title": "BICEPS BRACHII ULTRASOUND STUDIO",
        "app_subtitle": "End-to-End Pipeline: Raw .BIN (MATLAB HD) -> AI Segmentation -> Real-Time Kinematics",
        "preset_label": "Trial Preset:",
        "preset_prompt": "-- Select Biceps Trial Preset --",
        "btn_sample_hub": "📥 Sample Data & Presets Hub",
        "btn_select_bin": "📁 Select .BIN File",
        "btn_run_matlab": "⚡ Extract via MATLAB",
        "btn_select_folder": "📂 Browse Image Folder",
        "btn_run_ai": "🧠 Run AI Segmentation",
        "btn_running_ai": "Processing AI...",
        "cb_mask": "Biceps Mask (Sage)",
        "cb_sup": "Superficial Fascia (Lime)",
        "cb_deep": "Deep Fascia (Plum)",
        "cb_grid": "Depth Grid (mm)",
        "view_overlay": "View: Soft Overlay Mask",
        "view_side_by_side": "View: Side-by-Side (Raw & Mask)",
        "view_raw": "View: Raw B-Mode Only",
        "btn_play": "▶ Play (Space)",
        "btn_pause": "⏸ Pause (Space)",
        "speed_30": "30 FPS",
        "speed_60": "60 FPS (Normal)",
        "speed_112": "112 FPS (Native Cine)",
        "speed_15": "15 FPS (Slow)",
        "metrics_title": "MUSCLE BIOMECHANICS TELEMETRY",
        "card_current": "Current Thickness",
        "card_rest": "Rest Baseline Thickness",
        "card_peak": "Peak Active Thickness",
        "card_delta": "Contraction Delta & Strain",
        "status_active": "STATUS: ACTIVE CONTRACTION",
        "status_relax": "STATUS: RESTING (RELAX)",
        "btn_export_csv": "💾 Export Kinematics CSV",
        "btn_save_fig": "📊 Save HD Summary Figure",
        "btn_open_folder": "📁 Open Output Folder",
        "chart_title": "Dynamic Biceps Brachii Thickness Waveform (Kinematics)",
        "chart_y": "Thickness (mm)",
        "chart_x": "Time (Seconds)",
        "chart_curve_label": "Biceps Thickness (mm)",
        "chart_rest": "Rest",
        "chart_peak": "Peak",
        "theme_dark": "🌙 Dark Mode",
        "theme_light": "☀️ Light Mode",
        "lang_en": "🌐 English",
        "lang_zh": "🌐 繁體中文 (Mandarin)",
        "ready_status": "Studio ready. Select a trial preset or .BIN file to begin.",
        "extracting_status": "Extracting frames via MATLAB (Gilang HD Acoustic Method)...",
        "folder_loaded": "Folder loaded: {folder} ({count} frames detected).",
        "seg_complete_title": "Segmentation Complete",
        "seg_complete_body": "AI Segmentation successfully completed for {count} frames!\n\n- Rest Baseline : {rest:.2f} mm\n- Peak Active   : {peak:.2f} mm\n- Delta         : +{delta:.2f} mm (+{pct:.1f}%)\n- Frame Rate    : {fps:.1f} FPS",
        "export_csv_success": "Kinematics data successfully saved to:\n{path}",
        "export_fig_success": "Visual summary successfully saved to:\n{path}",
        "warning_no_bin": "Please select a .BIN file first!",
        "warning_no_images": "Please load an ultrasound frame folder first!",
        "warning_empty_folder": "No PNG frames found in: {folder}",
        "warning_no_data": "No kinematics data available to export!",
        # Hub Dialog Keys
        "hub_title": "Sample Datasets & Ultrasound Presets Hub",
        "hub_tab_datasets": "Trial Datasets (Azmi & Tissa)",
        "hub_tab_presets": "Ultrasound Presets & Specifications",
        "hub_azmi_title": "Subject Azmi (10-RM Dynamic Flexion)",
        "hub_azmi_desc": "858 frames cine loop | High frame rate RF acquisition | Clear fascia boundaries",
        "hub_tissa_title": "Subject Tissa (10-RM Dynamic Flexion)",
        "hub_tissa_desc": "918 frames cine loop | Dynamic concentric/eccentric phases | Pronounced aponeurosis motion",
        "hub_status_installed": "INSTALLED (Local Ready)",
        "hub_status_remote": "AVAILABLE VIA CLOUD",
        "hub_btn_load": "Load into Studio",
        "hub_btn_download": "Download Dataset",
        "hub_remote_url_label": "Cloud Asset Download URL (.zip):",
        "hub_apply_preset": "Apply Selected Preset to Studio",
        "hub_import_preset": "Import Custom Preset (.json/.yaml)",
        "hub_export_preset": "Export Preset Template",
        "hub_preset_applied_msg": "Ultrasound Preset '{name}' Applied!\n- Pixel Spacing: {spacing:.4f} mm/px\n- Frame Rate: {fps:.1f} FPS\n- Central Freq: {freq} MHz",
        "hub_hosting_advice": "Recommended Distribution: Host trial bundles (.zip) on GitHub Releases Asset or Hugging Face Datasets for high-speed, direct CDN distribution without API friction."
    },
    "zh": {
        "app_title": "肱二頭肌超音波分割與生物力學分析工作站",
        "app_subtitle": "完整處理流程：原始 .BIN (MATLAB 高清) -> U-Net AI 自動分割 -> 即時肌肉動力學",
        "preset_label": "預設試驗：",
        "preset_prompt": "-- 請選擇肱二頭肌試驗預設 --",
        "btn_sample_hub": "📥 範例數據與超音波規格中心",
        "btn_select_bin": "📁 選擇 .BIN 檔案",
        "btn_run_matlab": "⚡ MATLAB 高清解析",
        "btn_select_folder": "📂 瀏覽影像資料夾",
        "btn_run_ai": "🧠 執行 AI 肌肉分割",
        "btn_running_ai": "AI 分割處理中...",
        "cb_mask": "二頭肌遮罩 (鼠尾草綠)",
        "cb_sup": "淺層筋膜 (亮萊姆黃)",
        "cb_deep": "深層筋膜 (深酒紅)",
        "cb_grid": "深度標尺 (毫米)",
        "view_overlay": "視圖：柔和半透明疊加遮罩",
        "view_side_by_side": "視圖：並排對比 (原始與遮罩)",
        "view_raw": "視圖：僅顯示原始 B-Mode",
        "btn_play": "▶ 播放 (空白鍵)",
        "btn_pause": "⏸ 暫停 (空白鍵)",
        "speed_30": "30 FPS",
        "speed_60": "60 FPS (標準)",
        "speed_112": "112 FPS (原生超音波)",
        "speed_15": "15 FPS (慢速)",
        "metrics_title": "肌肉生物力學評估指標",
        "card_current": "當前肌肉厚度",
        "card_rest": "靜息基線厚度 (Rest)",
        "card_peak": "最大收縮厚度 (Peak)",
        "card_delta": "收縮增量與應變量",
        "status_active": "狀態：主動收縮中 (ACTIVE)",
        "status_relax": "狀態：靜息放鬆 (RELAX)",
        "btn_export_csv": "💾 匯出動力學 CSV 數據",
        "btn_save_fig": "📊 儲存高清視覺化圖表",
        "btn_open_folder": "📁 開啟輸出資料夾",
        "chart_title": "肱二頭肌即時厚度動力學波形圖 (毫米/秒)",
        "chart_y": "肌肉厚度 (毫米)",
        "chart_x": "時間 (秒)",
        "chart_curve_label": "二頭肌厚度 (毫米)",
        "chart_rest": "靜息基線",
        "chart_peak": "收縮峰值",
        "theme_dark": "🌙 深色護眼模式",
        "theme_light": "☀️ 淺色醫療模式",
        "lang_en": "🌐 English (英文)",
        "lang_zh": "🌐 繁體中文 (Mandarin)",
        "ready_status": "系統就緒。請選取試驗預設或 .BIN 檔案以開始分析。",
        "extracting_status": "正在使用 MATLAB 進行高清影像切片提取 (Gilang 方法)...",
        "folder_loaded": "已載入資料夾：{folder} (偵測到 {count} 幀影像)。",
        "seg_complete_title": "分割分析完成",
        "seg_complete_body": "肱二頭肌 AI 分割已順利完成共 {count} 幀！\n\n- 靜息厚度 (Rest)   : {rest:.2f} mm\n- 峰值厚度 (Peak)   : {peak:.2f} mm\n- 收縮增量 (Delta)  : +{delta:.2f} mm (+{pct:.1f}%)\n- 取樣幀率 (FPS)    : {fps:.1f} FPS",
        "export_csv_success": "動力學數據已成功儲存至：\n{path}",
        "export_fig_success": "高清視覺化圖表已成功儲存至：\n{path}",
        "warning_no_bin": "請先選擇 .BIN 檔案！",
        "warning_no_images": "請先載入超音波影像資料夾！",
        "warning_empty_folder": "未在資料夾中找到 PNG 影像：{folder}",
        "warning_no_data": "無可供匯出的動力學數據！",
        # Hub Dialog Keys
        "hub_title": "範例受試者數據與超音波規格設定中心",
        "hub_tab_datasets": "受試者試驗數據 (Azmi 與 Tissa)",
        "hub_tab_presets": "超音波預設與探頭規格",
        "hub_azmi_title": "受試者 Azmi (10-RM 動態手臂屈曲)",
        "hub_azmi_desc": "858 幀連續超音波 | 高幀率動態射頻取樣 | 筋膜層邊界極為清晰",
        "hub_tissa_title": "受試者 Tissa (10-RM 動態手臂屈曲)",
        "hub_tissa_desc": "918 幀連續超音波 | 完整向心與離心收縮階段 | 腱膜活動動態明顯",
        "hub_status_installed": "本機已安裝 (可直接載入)",
        "hub_status_remote": "雲端套件待下載",
        "hub_btn_load": "載入至分析工作站",
        "hub_btn_download": "下載數據集套件",
        "hub_remote_url_label": "雲端數據套件下載網址 (.zip)：",
        "hub_apply_preset": "套用所選預設至工作站",
        "hub_import_preset": "匯入自訂超音波規格檔 (.json/.yaml)",
        "hub_export_preset": "匯出預設規格範本",
        "hub_preset_applied_msg": "已成功套用超音波預設規格「{name}」！\n- 標尺校正比例: {spacing:.4f} mm/px\n- 取樣幀率: {fps:.1f} FPS\n- 中心頻率: {freq} MHz",
        "hub_hosting_advice": "最佳發布建議：將試驗數據壓縮包 (.zip) 上傳至 GitHub Releases Asset 或 Hugging Face Datasets，可享受全球高速 CDN 且無需額外身份驗證。"
    }
}


# =====================================================================
# THREAD: DOWNLOAD & EXTRACT SAMPLE DATASETS
# =====================================================================
class DownloadExtractWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, download_url, dest_dir):
        super().__init__()
        self.download_url = download_url
        self.dest_dir = dest_dir
        self.is_cancelled = False

    def run(self):
        try:
            os.makedirs(self.dest_dir, exist_ok=True)
            zip_dest = os.path.join(self.dest_dir, "temp_sample_data.zip")

            self.progress_signal.emit(5, f"Connecting to {self.download_url}...")

            req = urllib.request.Request(
                self.download_url,
                headers={"User-Agent": "Probefix-Ultrasound-Studio/1.0"}
            )

            with urllib.request.urlopen(req, timeout=30) as response, open(zip_dest, "wb") as out_file:
                total_length = response.getheader("content-length")
                total_bytes = int(total_length) if total_length else None
                dl_bytes = 0
                block_sz = 65536

                while True:
                    if self.is_cancelled:
                        self.finished_signal.emit(False, "Download cancelled by user.")
                        return
                    buf = response.read(block_sz)
                    if not buf:
                        break
                    dl_bytes += len(buf)
                    out_file.write(buf)

                    if total_bytes:
                        pct = int((dl_bytes / total_bytes) * 75)
                        mb_done = dl_bytes / (1024 * 1024)
                        mb_total = total_bytes / (1024 * 1024)
                        self.progress_signal.emit(
                            min(75, 5 + pct),
                            f"Downloading: {mb_done:.1f} MB / {mb_total:.1f} MB ({int(dl_bytes*100/total_bytes)}%)"
                        )
                    else:
                        mb_done = dl_bytes / (1024 * 1024)
                        self.progress_signal.emit(40, f"Downloaded {mb_done:.1f} MB...")

            self.progress_signal.emit(80, "Unpacking dataset archive...")
            if zipfile.is_zipfile(zip_dest):
                with zipfile.ZipFile(zip_dest, "r") as zip_ref:
                    zip_ref.extractall(self.dest_dir)
                try:
                    os.remove(zip_dest)
                except Exception:
                    pass
                self.progress_signal.emit(100, "Dataset extraction completed successfully!")
                self.finished_signal.emit(True, f"Extracted to: {self.dest_dir}")
            else:
                self.finished_signal.emit(False, "Downloaded file is not a valid zip archive.")

        except Exception as e:
            self.finished_signal.emit(False, str(e))


# =====================================================================
# THREAD 1: EKSTRAKSI MATLAB (.BIN -> SLICE PNG)
# =====================================================================
class MatlabExtractionThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str, str)

    def __init__(self, bin_path, output_dir=None, use_python_engine=False):
        super().__init__()
        self.bin_path = bin_path
        self.output_dir = output_dir
        self.use_python_engine = use_python_engine

    def run(self):
        try:
            bin_name = os.path.splitext(os.path.basename(self.bin_path))[0]
            clean_bin_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in bin_name)

            if not self.output_dir:
                self.output_dir = os.path.join(
                    PARENT_ROOT, "dataset", "images", "Extracted_Bin", f"trial_{clean_bin_name}"
                )
            os.makedirs(self.output_dir, exist_ok=True)

            self.progress_signal.emit(10, f"Processing: {os.path.basename(self.bin_path)}...")

            if not self.use_python_engine:
                self.progress_signal.emit(25, "Invoking MATLAB Engine (Gilang HD Method)...")
                converter_dir = get_resource_path("matlab_converter")
                bin_escaped = self.bin_path.replace("\\", "/")
                out_escaped = self.output_dir.replace("\\", "/")
                conv_escaped = converter_dir.replace("\\", "/")

                matlab_cmd = (
                    f"try, addpath('{conv_escaped}'); "
                    f"matlab_process_single_bin('{bin_escaped}', '{out_escaped}'); "
                    f"catch e, disp(getReport(e)); end; exit;"
                )

                cmd = ["matlab", "-nosplash", "-nodesktop", "-wait", "-r", matlab_cmd]
                try:
                    subprocess.run(cmd, capture_output=True, text=True)
                except (FileNotFoundError, Exception) as ex_matlab:
                    self.progress_signal.emit(35, f"MATLAB executable not available on host ({str(ex_matlab)}).")

                pngs = glob.glob(os.path.join(self.output_dir, "frame_*.png"))
                if len(pngs) > 0:
                    self.progress_signal.emit(100, f"Success! {len(pngs)} frames extracted via MATLAB.")
                    self.finished_signal.emit(True, self.output_dir, f"MATLAB extracted {len(pngs)} frames.")
                    return
                else:
                    self.progress_signal.emit(50, "MATLAB headless finished, falling back to Python Native Engine...")

            # Fallback Python Native
            self.progress_signal.emit(40, "Extracting RF Data with Python Native Gilang Engine...")
            self.extract_with_python(self.bin_path, self.output_dir)

        except Exception as e:
            self.finished_signal.emit(False, "", f"Extraction failed: {str(e)}")

    def extract_with_python(self, bin_path, out_dir):
        with open(bin_path, 'rb') as f:
            hdr_id = f.read(6).decode('ascii', errors='ignore')
            num_header_els = 11 if hdr_id.startswith(('RF0001', 'RF0002', 'RF0003', 'RF0004')) else 7
            f.seek(6)
            hdr = struct.unpack(f'<{num_header_els}i', f.read(num_header_els * 4))
            num_samples = hdr[6]
            num_lines = hdr[7]
            source_id = hdr[3]
            sampling_ns = hdr[8]
            fps = hdr[5] / 100.0 if hdr[5] > 0 else 60.0

            beam_bytes = f.read(3 * num_lines * 4)
            if len(beam_bytes) == 3 * num_lines * 4:
                beams = struct.unpack(f'<{3 * num_lines}i', beam_bytes)
                xs = [beams[3 * i] / 10000.0 for i in range(num_lines)]
                width_mm = (max(xs) - min(xs)) * 10.0
            else:
                width_mm = 38.4

            c_sound = 1540.0
            dz_mm = (c_sound * (sampling_ns * 1e-9) / 2.0) * 1000.0
            depth_mm = num_samples * dz_mm

            frame_header_bytes = (num_header_els * 4) + (3 * num_lines * 4) + (num_lines * 4 if hdr_id.startswith(('RF0003', 'RF0004')) else 0)
            data_elements = num_samples * num_lines * 2 if source_id == 4 else num_samples * num_lines
            frame_data_bytes = data_elements * 2

            f.seek(0, 2)
            total_size = f.tell()

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

        num_slices = len(frames)
        aspect = width_mm / depth_mm
        target_h = 512
        target_w = int(round(target_h * aspect))

        for idx, fr in enumerate(frames, 1):
            a3 = np.abs(fr).astype(np.float32) ** 0.25
            a3_min, a3_max = np.min(a3), np.max(a3)
            a3_norm = (a3 - a3_min) / (a3_max - a3_min + 1e-9)
            a4 = cv2.GaussianBlur(a3_norm, (0, 0), 1.0)
            b_iso = cv2.resize(a4, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            b_uint8 = (np.clip(b_iso, 0, 1) * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"frame_{idx:04d}.png"), b_uint8)

            if idx % 50 == 0 or idx == num_slices:
                pct = 50 + int((idx / num_slices) * 48)
                self.progress_signal.emit(pct, f"Saving frame {idx}/{num_slices}...")

        with open(os.path.join(out_dir, "metadata.txt"), "w") as f_meta:
            f_meta.write(f"Source_Bin: {os.path.basename(bin_path)}\n")
            f_meta.write(f"Num_Frames: {num_slices}\n")
            f_meta.write(f"FPS: {fps:.2f}\n")
            f_meta.write(f"Depth_mm: {depth_mm:.2f}\n")
            f_meta.write(f"Width_mm: {width_mm:.2f}\n")
            f_meta.write(f"Pixel_Spacing_mm: {depth_mm / target_h:.4f}\n")

        self.progress_signal.emit(100, f"Extraction complete! {num_slices} frames extracted.")
        self.finished_signal.emit(True, out_dir, f"Successfully extracted {num_slices} frames.")


# =====================================================================
# THREAD 2: INFERENSI SEGMENTASI U-NET AI (WARNING-SUPPRESSED)
# =====================================================================
class SegmentationInferenceThread(QThread):
    progress_signal = pyqtSignal(int, str)
    frame_done_signal = pyqtSignal(int, float, float, float)
    finished_signal = pyqtSignal(bool, dict)

    def __init__(self, image_paths, model_path=None, pixel_spacing_mm=0.1757, fps=60.0):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        self.pixel_spacing_mm = pixel_spacing_mm
        self.fps = fps
        self.is_cancelled = False

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.progress_signal.emit(5, f"Loading U-Net on {device}...")

            cfg_path = get_resource_path(os.path.join("configs", "config.yaml"))
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg = yaml.safe_load(f)
            else:
                cfg = {"model": {"name": "unet", "encoder_name": "resnet34", "in_channels": 1, "classes": 1}}

            model = build_model(cfg).to(device)

            if not self.model_path or not os.path.exists(self.model_path):
                self.model_path = get_resource_path(os.path.join("weights", "best_model.pth"))
                if not os.path.exists(self.model_path):
                    self.model_path = get_resource_path(os.path.join("weights", "interactive_unet.pth"))

            # Clean PyTorch Checkpoint Loading (Zero FutureWarning)
            try:
                ckpt = torch.load(self.model_path, map_location=device, weights_only=True)
            except Exception:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    ckpt = torch.load(self.model_path, map_location=device, weights_only=False)

            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()

            n_frames = len(self.image_paths)
            target_h, target_w = 512, 224

            superficial_mm = []
            deep_mm = []
            thickness_mm = []
            timestamps = []
            masks_list = []

            self.progress_signal.emit(10, f"Segmenting {n_frames} frames...")

            with torch.no_grad():
                for idx, p in enumerate(self.image_paths):
                    if self.is_cancelled:
                        break

                    raw_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if raw_img is None:
                        continue

                    orig_h, orig_w = raw_img.shape
                    resized = cv2.resize(raw_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
                    t_in = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                    pred = model(t_in)[0, 0].cpu().numpy()
                    raw_bin = (pred >= 0.5).astype(np.float32)
                    clean_mask = postprocess_mask(raw_bin, min_size=400)

                    mask_orig = cv2.resize(clean_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    masks_list.append(mask_orig)

                    t_info = calculate_muscle_thickness(mask_orig, pixel_spacing_mm=self.pixel_spacing_mm)
                    th = t_info["mean_thickness_mm"]

                    col_tops = []
                    col_bottoms = []
                    for c in range(orig_w):
                        pts = np.where(mask_orig[:, c] > 0.5)[0]
                        if len(pts) > 0:
                            col_tops.append(pts[0] * self.pixel_spacing_mm)
                            col_bottoms.append(pts[-1] * self.pixel_spacing_mm)

                    sup_d = float(np.mean(col_tops)) if len(col_tops) > 0 else 14.0
                    deep_d = float(np.mean(col_bottoms)) if len(col_bottoms) > 0 else sup_d + th
                    t_sec = idx / self.fps

                    superficial_mm.append(sup_d)
                    deep_mm.append(deep_d)
                    thickness_mm.append(th)
                    timestamps.append(t_sec)

                    self.frame_done_signal.emit(idx, sup_d, deep_d, th)

                    if idx % 15 == 0 or idx == n_frames - 1:
                        pct = 10 + int((idx / n_frames) * 88)
                        self.progress_signal.emit(pct, f"Segmenting frame {idx + 1}/{n_frames} (Thick: {th:.2f} mm)...")

            self.progress_signal.emit(100, "AI segmentation complete!")

            results = {
                "superficial_mm": np.array(superficial_mm),
                "deep_mm": np.array(deep_mm),
                "thickness_mm": np.array(thickness_mm),
                "timestamps": np.array(timestamps),
                "masks": masks_list
            }
            self.finished_signal.emit(True, results)

        except Exception as e:
            self.finished_signal.emit(False, {"error": str(e)})


# =====================================================================
# WIDGET GRAFIK KINEMATIKA MATPLOTLIB (PALETTE HARMONIZED)
# =====================================================================
class KinematicsGraphWidget(QWidget):
    seek_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme = "dark"
        self.lang = "en"
        # Facecolor #232A27 for dark mode, harmonized with #2E3532
        self.fig = Figure(figsize=(8, 2.3), facecolor="#232A27")
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.needle = None
        self.timestamps = []
        self.thickness_mm = []
        self.rest_baseline_mm = None
        self.peak_mm = None
        self.total_frames = 0

        self.canvas.mpl_connect("button_press_event", self.on_click)

    def set_theme_and_lang(self, theme, lang):
        self.theme = theme
        self.lang = lang
        self.update_plot()

    def update_data(self, timestamps, thickness_mm, rest_baseline_mm=None, peak_mm=None):
        self.timestamps = timestamps
        self.thickness_mm = thickness_mm
        self.rest_baseline_mm = rest_baseline_mm
        self.peak_mm = peak_mm
        self.total_frames = len(timestamps)
        self.update_plot()

    def update_plot(self):
        t_dict = TRANSLATIONS[self.lang]
        self.ax.clear()

        # Google Gemini Preset Mapping
        if self.theme == "dark":
            bg_fig = "#131314"
            bg_ax = "#1e1f20"
            text_color = "#c4c7c5"       # Gemini muted text
            grid_color = "#37393b"       # Subtle divider
            line_color = "#a8c7fa"       # Gemini Blue
            fill_color = "#a8c7fa"
            rest_color = "#fde293"       # Gemini Amber
            peak_color = "#f28b82"       # Gemini Coral
            needle_color = "#ffffff"     # Sharp White Needle
            legend_bg = "#1e1f20"
            legend_edge = "#37393b"
            legend_text = "#e3e3e3"
        else:
            bg_fig = "#f0f4f9"
            bg_ax = "#ffffff"
            text_color = "#444746"
            grid_color = "#e0e3e7"
            line_color = "#0b57d0"       # Gemini Light Blue
            fill_color = "#a8c7fa"
            rest_color = "#e37400"       # Light Amber
            peak_color = "#d93025"       # Light Coral
            needle_color = "#1f1f1f"
            legend_bg = "#ffffff"
            legend_edge = "#e0e3e7"
            legend_text = "#1f1f1f"

        self.fig.patch.set_facecolor(bg_fig)
        self.ax.set_facecolor(bg_ax)

        if len(self.timestamps) > 1:
            total_time = self.timestamps[-1]
            self.ax.plot(self.timestamps, self.thickness_mm, color=line_color, linewidth=2.3, label=t_dict["chart_curve_label"])
            self.ax.fill_between(self.timestamps, 0, self.thickness_mm, color=fill_color, alpha=0.18)

            if self.rest_baseline_mm is not None:
                self.ax.axhline(self.rest_baseline_mm, color=rest_color, linestyle="--", linewidth=1.6,
                                label=f"{t_dict['chart_rest']}: {self.rest_baseline_mm:.1f}mm")
            if self.peak_mm is not None:
                self.ax.axhline(self.peak_mm, color=peak_color, linestyle=":", linewidth=1.8,
                                label=f"{t_dict['chart_peak']}: {self.peak_mm:.1f}mm")

            self.needle = self.ax.axvline(self.timestamps[0], color=needle_color, linewidth=2.0)
            self.ax.set_xlim(0, total_time)
            y_max = max(self.thickness_mm) * 1.25 if len(self.thickness_mm) > 0 else 40
            self.ax.set_ylim(0, max(30, y_max))

            self.ax.set_ylabel(t_dict["chart_y"], color=text_color, fontsize=9, fontweight="bold")
            self.ax.set_xlabel(t_dict["chart_x"], color=text_color, fontsize=9)
            self.ax.set_title(t_dict["chart_title"], color=text_color, fontsize=10, fontweight="bold", pad=6)
            self.ax.tick_params(colors=text_color, labelsize=8)
            self.ax.grid(True, linestyle="--", alpha=0.45, color=grid_color)
            self.ax.legend(loc="upper right", facecolor=legend_bg, edgecolor=legend_edge, labelcolor=legend_text, fontsize=8)

            for spine in self.ax.spines.values():
                spine.set_color(grid_color)

        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    def set_cursor_frame(self, frame_idx):
        if self.needle and 0 <= frame_idx < len(self.timestamps):
            t = self.timestamps[frame_idx]
            self.needle.set_xdata([t, t])
            self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes == self.ax and event.xdata is not None and len(self.timestamps) > 0:
            click_t = event.xdata
            closest_idx = int(np.argmin(np.abs(self.timestamps - click_t)))
            self.seek_signal.emit(closest_idx)


# =====================================================================
# MODAL DIALOG: SAMPLE DATASETS & ULTRASOUND PRESETS HUB
# =====================================================================
class SampleDataAndPresetDialog(QDialog):
    load_folder_signal = pyqtSignal(str)
    preset_applied_signal = pyqtSignal(dict)

    def __init__(self, parent=None, current_lang="en", current_theme="dark"):
        super().__init__(parent)
        self.parent = parent
        self.lang = current_lang
        self.theme = current_theme
        self.download_worker = None

        t = TRANSLATIONS[self.lang]
        self.setWindowTitle(t["hub_title"])
        self.resize(780, 560)
        self.init_ui()

    def init_ui(self):
        t = TRANSLATIONS[self.lang]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Datasets
        tab_datasets = QWidget()
        vbox_d = QVBoxLayout(tab_datasets)
        vbox_d.setContentsMargins(12, 12, 12, 12)
        vbox_d.setSpacing(12)

        # Card 1: Azmi 10-RM
        self.card_azmi = self.create_dataset_card(
            title=t["hub_azmi_title"],
            desc=t["hub_azmi_desc"],
            subj_dir_name="Azmi_-_RM_10"
        )
        vbox_d.addWidget(self.card_azmi)

        # Card 2: Tissa 10-RM
        self.card_tissa = self.create_dataset_card(
            title=t["hub_tissa_title"],
            desc=t["hub_tissa_desc"],
            subj_dir_name="Tissa_-_10RM"
        )
        vbox_d.addWidget(self.card_tissa)

        # Remote URL config & Downloader Box
        group_dl = QGroupBox(t["hub_remote_url_label"])
        group_dl.setObjectName("panelCard")
        vbox_dl = QVBoxLayout(group_dl)
        vbox_dl.setSpacing(8)

        self.edit_remote_url = QLineEdit()
        self.edit_remote_url.setText("https://github.com/Antigravity/Probefix/releases/download/v1.0.0/biceps_sample_trials.zip")
        self.edit_remote_url.setPlaceholderText("Enter direct .zip download link...")
        vbox_dl.addWidget(self.edit_remote_url)

        dl_btn_row = QHBoxLayout()
        self.btn_download_bundle = QPushButton(t["hub_btn_download"])
        self.btn_download_bundle.setObjectName("btnPlum")
        self.btn_download_bundle.clicked.connect(self.start_bundle_download)
        dl_btn_row.addWidget(self.btn_download_bundle)

        self.lbl_dl_status = QLabel("")
        self.lbl_dl_status.setObjectName("lblMonoInfo")
        dl_btn_row.addWidget(self.lbl_dl_status, stretch=1)
        vbox_dl.addLayout(dl_btn_row)

        self.dl_progress = QProgressBar()
        self.dl_progress.setFixedHeight(12)
        self.dl_progress.setValue(0)
        self.dl_progress.setVisible(False)
        vbox_dl.addWidget(self.dl_progress)

        vbox_d.addWidget(group_dl)

        # Architectural Recommendation Notice
        lbl_notice = QLabel(t["hub_hosting_advice"])
        lbl_notice.setWordWrap(True)
        lbl_notice.setStyleSheet("font-size: 11px; color: #7E9181; font-style: italic; padding: 4px;")
        vbox_d.addWidget(lbl_notice)
        vbox_d.addStretch()

        self.tabs.addTab(tab_datasets, t["hub_tab_datasets"])

        # Tab 2: Ultrasound Presets & Probe Specifications
        tab_presets = QWidget()
        vbox_p = QVBoxLayout(tab_presets)
        vbox_p.setContentsMargins(12, 12, 12, 12)
        vbox_p.setSpacing(12)

        p_row1 = QHBoxLayout()
        lbl_select = QLabel("Active Ultrasound Transducer Preset:")
        p_row1.addWidget(lbl_select)

        self.combo_us_presets = QComboBox()
        self.combo_us_presets.currentIndexChanged.connect(self.on_preset_dropdown_changed)
        p_row1.addWidget(self.combo_us_presets, stretch=1)
        vbox_p.addLayout(p_row1)

        # Presets Specs Table
        self.table_specs = QTableWidget(7, 2)
        self.table_specs.setHorizontalHeaderLabels(["Parameter", "Specification Value"])
        self.table_specs.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_specs.verticalHeader().setVisible(False)
        self.table_specs.setEditTriggers(QTableWidget.NoEditTriggers)
        vbox_p.addWidget(self.table_specs)

        # Notes label
        self.lbl_preset_notes = QLabel("")
        self.lbl_preset_notes.setWordWrap(True)
        self.lbl_preset_notes.setStyleSheet("font-size: 11px; color: #DBC2CF; padding: 4px;")
        vbox_p.addWidget(self.lbl_preset_notes)

        # Buttons Row
        p_btn_row = QHBoxLayout()
        self.btn_apply_preset = QPushButton(t["hub_apply_preset"])
        self.btn_apply_preset.setObjectName("btnLime")
        self.btn_apply_preset.clicked.connect(self.on_apply_preset_clicked)
        p_btn_row.addWidget(self.btn_apply_preset)

        self.btn_import_preset = QPushButton(t["hub_import_preset"])
        self.btn_import_preset.setObjectName("btnSage")
        self.btn_import_preset.clicked.connect(self.on_import_preset_clicked)
        p_btn_row.addWidget(self.btn_import_preset)

        self.btn_export_template = QPushButton(t["hub_export_preset"])
        self.btn_export_template.clicked.connect(self.on_export_template_clicked)
        p_btn_row.addWidget(self.btn_export_template)

        vbox_p.addLayout(p_btn_row)
        vbox_p.addStretch()

        self.tabs.addTab(tab_presets, t["hub_tab_presets"])

        # Load presets data from disk
        self.load_presets_from_file()

        # Bottom Close Button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignRight)

    def create_dataset_card(self, title, desc, subj_dir_name):
        t = TRANSLATIONS[self.lang]
        card = QFrame()
        card.setObjectName("metricCard")
        hbox = QHBoxLayout(card)
        hbox.setContentsMargins(14, 10, 14, 10)
        hbox.setSpacing(14)

        vbox = QVBoxLayout()
        vbox.setSpacing(3)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("font-size: 13px; font-weight: bold; color: #FFFFFF;")
        lbl_d = QLabel(desc)
        lbl_d.setStyleSheet("font-size: 11px; color: #7E9181;")
        vbox.addWidget(lbl_t)
        vbox.addWidget(lbl_d)
        hbox.addLayout(vbox, stretch=1)

        # Check local existence
        target_path = None
        images_base = os.path.join(PARENT_ROOT, "dataset", "images")
        subj_dir = os.path.join(images_base, subj_dir_name)

        frame_count = 0
        if os.path.exists(subj_dir):
            trials = [d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))]
            if trials:
                t_dir = os.path.join(subj_dir, trials[0])
                pngs = glob.glob(os.path.join(t_dir, "frame_*.png"))
                if pngs:
                    target_path = t_dir
                    frame_count = len(pngs)
            else:
                pngs = glob.glob(os.path.join(subj_dir, "frame_*.png"))
                if pngs:
                    target_path = subj_dir
                    frame_count = len(pngs)

        if target_path and frame_count > 0:
            lbl_badge = QLabel(f"✓ {t['hub_status_installed']} ({frame_count} f)")
            lbl_badge.setStyleSheet(
                "background-color: #2E3532; color: #C7EF00; border: 1px solid #7E9181; "
                "border-radius: 6px; padding: 6px 10px; font-weight: bold; font-size: 10px;"
            )
            hbox.addWidget(lbl_badge)

            btn_load = QPushButton(t["hub_btn_load"])
            btn_load.setObjectName("btnLime")
            btn_load.clicked.connect(lambda: self.load_dataset(target_path))
            hbox.addWidget(btn_load)
        else:
            lbl_badge = QLabel(t["hub_status_remote"])
            lbl_badge.setStyleSheet(
                "background-color: #2E3532; color: #DBC2CF; border: 1px solid #80475E; "
                "border-radius: 6px; padding: 6px 10px; font-size: 10px;"
            )
            hbox.addWidget(lbl_badge)

            btn_dl = QPushButton(t["hub_btn_download"])
            btn_dl.setObjectName("btnPlum")
            btn_dl.clicked.connect(self.start_bundle_download)
            hbox.addWidget(btn_dl)

        return card

    def load_dataset(self, folder_path):
        self.load_folder_signal.emit(folder_path)
        self.accept()

    def start_bundle_download(self):
        url = self.edit_remote_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid dataset .zip URL.")
            return

        dest_dir = os.path.join(PARENT_ROOT, "dataset", "images")
        self.dl_progress.setVisible(True)
        self.dl_progress.setValue(5)
        self.lbl_dl_status.setText("Initiating download...")
        self.btn_download_bundle.setEnabled(False)

        self.download_worker = DownloadExtractWorker(url, dest_dir)
        self.download_worker.progress_signal.connect(self.on_download_progress)
        self.download_worker.finished_signal.connect(self.on_download_finished)
        self.download_worker.start()

    def on_download_progress(self, pct, msg):
        self.dl_progress.setValue(pct)
        self.lbl_dl_status.setText(msg)

    def on_download_finished(self, success, msg):
        self.btn_download_bundle.setEnabled(True)
        t = TRANSLATIONS[self.lang]
        if success:
            QMessageBox.information(self, "Download Successful", f"{t['dlg_download_success'] if 'dlg_download_success' in t else 'Download completed!'}\n{msg}")
            if self.parent:
                self.parent.discover_existing_trials()
            self.accept()
        else:
            QMessageBox.warning(self, "Download Failed", f"Could not download dataset:\n{msg}")
            self.lbl_dl_status.setText("Failed.")

    def load_presets_from_file(self):
        presets_path = get_resource_path(os.path.join("presets", "ultrasound_presets.json"))
        if not os.path.exists(presets_path):
            return

        with open(presets_path, "r", encoding="utf-8") as f:
            self.presets_data = json.load(f)

        self.combo_us_presets.clear()
        for p in self.presets_data.get("presets", []):
            self.combo_us_presets.addItem(p["name"], p)

        if self.combo_us_presets.count() > 0:
            self.on_preset_dropdown_changed(0)

    def on_preset_dropdown_changed(self, idx):
        preset = self.combo_us_presets.currentData()
        if not preset:
            return

        params = [
            ("Transducer Model", preset.get("probe_model", "Telemed Linear")),
            ("Transducer Type", preset.get("probe_type", "Linear Array 40 mm")),
            ("Center Frequency", f"{preset.get('frequency_mhz', 10.0)} MHz ({preset.get('frequency_range_mhz', '')})"),
            ("Acoustic Sound Velocity", f"{preset.get('sound_velocity_m_s', 1540)} m/s"),
            ("Default Acquisition Depth", f"{preset.get('max_depth_mm', 45.0)} mm"),
            ("Cine Acquisition Frame Rate", f"{preset.get('native_fps', 112.0)} FPS"),
            ("Pixel Spacing Calibration", f"{preset.get('pixel_spacing_mm', 0.05)} mm / pixel"),
        ]

        for row, (k, v) in enumerate(params):
            item_k = QTableWidgetItem(k)
            item_k.setForeground(QColor("#DBC2CF"))
            item_v = QTableWidgetItem(str(v))
            item_v.setForeground(QColor("#C7EF00"))
            self.table_specs.setItem(row, 0, item_k)
            self.table_specs.setItem(row, 1, item_v)

        notes = preset.get("notes", "")
        self.lbl_preset_notes.setText(f"Preset Notes: {notes}")

    def on_apply_preset_clicked(self):
        preset = self.combo_us_presets.currentData()
        if preset:
            self.preset_applied_signal.emit(preset)
            t = TRANSLATIONS[self.lang]
            msg = t["hub_preset_applied_msg"].format(
                name=preset.get("name", "Preset"),
                spacing=preset.get("pixel_spacing_mm", 0.05),
                fps=preset.get("native_fps", 112.0),
                freq=preset.get("frequency_mhz", 10.0)
            )
            QMessageBox.information(self, "Preset Applied", msg)

    def on_import_preset_clicked(self):
        f_path, _ = QFileDialog.getOpenFileName(
            self, "Import Ultrasound Specification Preset", PROJECT_ROOT,
            "Ultrasound Presets (*.json *.yaml *.yml)"
        )
        if f_path:
            try:
                if f_path.endswith((".yaml", ".yml")):
                    with open(f_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                else:
                    with open(f_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                if "presets" in data:
                    preset_list = data["presets"]
                elif isinstance(data, list):
                    preset_list = data
                else:
                    preset_list = [data]

                for p in preset_list:
                    name = p.get("name", os.path.basename(f_path))
                    self.combo_us_presets.addItem(f"[Imported] {name}", p)

                self.combo_us_presets.setCurrentIndex(self.combo_us_presets.count() - 1)
                QMessageBox.information(self, "Import Success", f"Successfully imported {len(preset_list)} preset(s)!")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to parse preset file:\n{str(e)}")

    def on_export_template_clicked(self):
        dest_path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset Template", os.path.join(PROJECT_ROOT, "presets", "custom_ultrasound_spec.json"),
            "JSON (*.json)"
        )
        if dest_path:
            template = {
                "name": "Custom Transducer Specification",
                "probe_model": "Custom Linear Array",
                "probe_type": "Linear Array (40 mm)",
                "frequency_mhz": 10.0,
                "frequency_range_mhz": "5.0 - 12.0",
                "element_count": 128,
                "sound_velocity_m_s": 1540,
                "max_depth_mm": 45.0,
                "native_fps": 112.0,
                "pixel_spacing_mm": 0.05,
                "target_muscle": "Biceps Brachii",
                "notes": "Custom calibration parameters provided by user."
            }
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2)
            QMessageBox.information(self, "Export Template", f"Preset template successfully saved to:\n{dest_path}")


# =====================================================================
# JENDELA UTAMA: BICEPS BRACHII PIPELINE STUDIO
# =====================================================================
class BicepsPipelineStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        # Default settings: English (default) & Dark Eye-Care mode
        self.current_lang = "en"
        self.current_theme = "dark"

        self.resize(1620, 960)

        # State Data
        self.current_bin_file = None
        self.current_image_dir = None
        self.image_paths = []
        self.masks_cache = []
        self.superficial_depth_mm = []
        self.deep_depth_mm = []
        self.thickness_mm = []
        self.timestamps = []
        self.pixel_spacing_mm = 0.05       # Standard Telemed calibrated spacing
        self.fps = 112.0                   # Native RF dynamic cine frame rate
        self.active_preset_name = "Telemed ArtUs L12-5N40-A4"

        # State Tampilan & Player
        self.current_frame_idx = 0
        self.is_playing = False
        self.view_mode = "overlay"  # 'overlay', 'side_by_side', 'raw'

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.on_timer_tick)

        self.init_ui()
        self.apply_theme()
        self.retranslate_ui()
        self.discover_existing_trials()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 10, 14, 10)
        main_layout.setSpacing(9)

        # -------------------------------------------------------------
        # 1. HEADER & TOP RIBBON TOOLBAR
        # -------------------------------------------------------------
        header_bar = QHBoxLayout()
        header_bar.setSpacing(10)

        # Logo / Title
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("headerTitle")
        self.lbl_subtitle = QLabel()
        self.lbl_subtitle.setObjectName("headerSubtitle")
        title_box.addWidget(self.lbl_title)
        title_box.addWidget(self.lbl_subtitle)
        header_bar.addLayout(title_box)

        header_bar.addStretch()

        # Selector Bahasa & Tema
        self.combo_lang = QComboBox()
        self.combo_lang.setFixedWidth(130)
        self.combo_lang.addItem("🌐 English", "en")
        self.combo_lang.addItem("🌐 繁體中文", "zh")
        self.combo_lang.currentIndexChanged.connect(self.on_language_changed)
        header_bar.addWidget(self.combo_lang)

        self.combo_theme = QComboBox()
        self.combo_theme.setFixedWidth(140)
        self.combo_theme.addItem("🌙 Dark Mode", "dark")
        self.combo_theme.addItem("☀️ Light Mode", "light")
        self.combo_theme.currentIndexChanged.connect(self.on_theme_changed)
        header_bar.addWidget(self.combo_theme)

        # Preset Dropdown
        self.lbl_preset_tag = QLabel()
        header_bar.addWidget(self.lbl_preset_tag)
        self.combo_presets = QComboBox()
        self.combo_presets.setFixedWidth(240)
        self.combo_presets.currentIndexChanged.connect(self.on_preset_selected)
        header_bar.addWidget(self.combo_presets)

        # Sample Datasets & Presets Hub Button
        self.btn_sample_hub = QPushButton()
        self.btn_sample_hub.setObjectName("btnPlum")
        self.btn_sample_hub.clicked.connect(self.open_sample_data_dialog)
        header_bar.addWidget(self.btn_sample_hub)

        # Tombol Pilih File .BIN
        self.btn_select_bin = QPushButton()
        self.btn_select_bin.setObjectName("btnCharcoal")
        self.btn_select_bin.clicked.connect(self.on_select_bin_file)
        header_bar.addWidget(self.btn_select_bin)

        # Tombol Ekstraksi MATLAB
        self.btn_run_matlab = QPushButton()
        self.btn_run_matlab.setObjectName("btnSage")
        self.btn_run_matlab.clicked.connect(self.on_run_matlab_extraction)
        header_bar.addWidget(self.btn_run_matlab)

        # Tombol Pilih Folder Foto Langsung
        self.btn_select_folder = QPushButton()
        self.btn_select_folder.setObjectName("btnCharcoal")
        self.btn_select_folder.clicked.connect(self.on_select_images_folder)
        header_bar.addWidget(self.btn_select_folder)

        # Tombol Jalankan Segmentasi AI
        self.btn_run_segmentation = QPushButton()
        self.btn_run_segmentation.setObjectName("btnPlum")
        self.btn_run_segmentation.clicked.connect(self.on_run_ai_segmentation)
        header_bar.addWidget(self.btn_run_segmentation)

        main_layout.addLayout(header_bar)

        # Progress Bar Global
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # -------------------------------------------------------------
        # 2. AREA TENGAH: DISPLAY CITRA ULTRASONOGRAFI & KONTROL
        # -------------------------------------------------------------
        content_splitter = QSplitter(Qt.Horizontal)

        # Panel Kiri: Kanvas Citra & Visualisasi
        image_panel = QFrame()
        image_panel.setObjectName("panelCard")
        vbox_img = QVBoxLayout(image_panel)
        vbox_img.setContentsMargins(12, 10, 12, 10)
        vbox_img.setSpacing(8)

        # Sub-Toolbar Tampilan (Filter & Checkbox)
        layer_bar = QHBoxLayout()
        layer_bar.setSpacing(14)

        self.cb_mask = QCheckBox()
        self.cb_mask.setChecked(True)
        self.cb_mask.stateChanged.connect(self.update_canvas_display)
        layer_bar.addWidget(self.cb_mask)

        self.cb_sup = QCheckBox()
        self.cb_sup.setChecked(True)
        self.cb_sup.stateChanged.connect(self.update_canvas_display)
        layer_bar.addWidget(self.cb_sup)

        self.cb_deep = QCheckBox()
        self.cb_deep.setChecked(True)
        self.cb_deep.stateChanged.connect(self.update_canvas_display)
        layer_bar.addWidget(self.cb_deep)

        self.cb_grid = QCheckBox()
        self.cb_grid.setChecked(True)
        self.cb_grid.stateChanged.connect(self.update_canvas_display)
        layer_bar.addWidget(self.cb_grid)

        layer_bar.addStretch()

        self.combo_view = QComboBox()
        self.combo_view.setFixedWidth(220)
        self.combo_view.currentIndexChanged.connect(self.on_view_mode_changed)
        layer_bar.addWidget(self.combo_view)

        vbox_img.addLayout(layer_bar)

        # Kanvas Utama (QLabel untuk QPixmap)
        self.canvas_label = QLabel()
        self.canvas_label.setAlignment(Qt.AlignCenter)
        self.canvas_label.setObjectName("canvasViewport")
        self.canvas_label.setMinimumHeight(440)
        vbox_img.addWidget(self.canvas_label, stretch=1)

        # Kontrol Playback (Scrubber, Play, FPS)
        player_bar = QHBoxLayout()
        player_bar.setSpacing(8)

        self.btn_play = QPushButton()
        self.btn_play.setFixedWidth(120)
        self.btn_play.setObjectName("btnLime")
        self.btn_play.clicked.connect(self.toggle_play)
        player_bar.addWidget(self.btn_play)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(38)
        self.btn_prev.clicked.connect(self.step_prev)
        player_bar.addWidget(self.btn_prev)

        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(38)
        self.btn_next.clicked.connect(self.step_next)
        player_bar.addWidget(self.btn_next)

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.valueChanged.connect(self.on_slider_moved)
        player_bar.addWidget(self.slider_frame, stretch=1)

        self.lbl_frame_info = QLabel("Frame: 0 / 0 | 0.00 s")
        self.lbl_frame_info.setObjectName("lblMonoInfo")
        player_bar.addWidget(self.lbl_frame_info)

        self.combo_fps = QComboBox()
        self.combo_fps.setFixedWidth(140)
        self.combo_fps.currentIndexChanged.connect(self.on_fps_changed)
        player_bar.addWidget(self.combo_fps)

        vbox_img.addLayout(player_bar)
        content_splitter.addWidget(image_panel)

        # Panel Kanan: Kartu Metrik Klinis & Aksi Ekspor
        right_panel = QFrame()
        right_panel.setFixedWidth(340)
        right_panel.setObjectName("panelCard")
        vbox_right = QVBoxLayout(right_panel)
        vbox_right.setContentsMargins(14, 14, 14, 14)
        vbox_right.setSpacing(10)

        self.lbl_metrics_header = QLabel()
        self.lbl_metrics_header.setObjectName("metricsHeader")
        vbox_right.addWidget(self.lbl_metrics_header)

        # Card 1: Tebal Saat Ini (#C7EF00 Electric Lime)
        self.card_current = self.create_metric_card("", "0.00 mm", "#C7EF00")
        vbox_right.addWidget(self.card_current)

        # Card 2: Tebal Istirahat (Rest) (#7E9181 Sage Green)
        self.card_rest = self.create_metric_card("", "0.00 mm", "#7E9181")
        vbox_right.addWidget(self.card_rest)

        # Card 3: Tebal Puncak (Peak Active) (#DBC2CF Soft Mauve)
        self.card_peak = self.create_metric_card("", "0.00 mm", "#DBC2CF")
        vbox_right.addWidget(self.card_peak)

        # Card 4: Delta Kontraksi & Strain (#C7EF00 Electric Lime)
        self.card_delta = self.create_metric_card("", "+0.00 mm (+0.0%)", "#C7EF00")
        vbox_right.addWidget(self.card_delta)

        # Status Kontraksi Real-time
        self.lbl_state_badge = QLabel()
        self.lbl_state_badge.setAlignment(Qt.AlignCenter)
        self.lbl_state_badge.setObjectName("badgeStateRelax")
        vbox_right.addWidget(self.lbl_state_badge)

        vbox_right.addStretch()

        # Tombol Aksi Ekspor
        self.btn_export_csv = QPushButton()
        self.btn_export_csv.clicked.connect(self.export_csv)
        vbox_right.addWidget(self.btn_export_csv)

        self.btn_save_fig = QPushButton()
        self.btn_save_fig.clicked.connect(self.save_visualization_figure)
        vbox_right.addWidget(self.btn_save_fig)

        self.btn_open_folder = QPushButton()
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        vbox_right.addWidget(self.btn_open_folder)

        content_splitter.addWidget(right_panel)
        main_layout.addWidget(content_splitter, stretch=2)

        # -------------------------------------------------------------
        # 3. AREA BAWAH: GRAFIK KINEMATIKA WAKTU-NYATA
        # -------------------------------------------------------------
        self.kinematics_widget = KinematicsGraphWidget(self)
        self.kinematics_widget.seek_signal.connect(self.seek_to_frame)
        self.kinematics_widget.setFixedHeight(175)
        main_layout.addWidget(self.kinematics_widget)

        # Status Bar
        self.setStatusBar(QStatusBar(self))

        # Global Shortcuts
        QShortcut(QKeySequence("Space"), self).activated.connect(self.toggle_play)
        QShortcut(QKeySequence("Left"), self).activated.connect(self.step_prev)
        QShortcut(QKeySequence("Right"), self).activated.connect(self.step_next)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.export_csv)

    def create_metric_card(self, title, init_val, color_hex):
        frame = QFrame()
        frame.setObjectName("metricCard")
        v = QVBoxLayout(frame)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(2)
        lbl_t = QLabel(title.upper())
        lbl_t.setObjectName("cardTitle")
        lbl_v = QLabel(init_val)
        lbl_v.setStyleSheet(f"font-size: 19px; font-weight: 800; color: {color_hex}; font-family: 'Consolas', monospace;")
        v.addWidget(lbl_t)
        v.addWidget(lbl_v)
        frame.title_label = lbl_t
        frame.val_label = lbl_v
        frame.color_hex = color_hex
        return frame

    # -----------------------------------------------------------------
    # THEME ENGINE (HARMONIZED PALETTE FROM D:\Download Move1\Main.png)
    # -----------------------------------------------------------------
    def on_theme_changed(self, idx):
        self.current_theme = self.combo_theme.currentData()
        self.apply_theme()
        self.kinematics_widget.set_theme_and_lang(self.current_theme, self.current_lang)
        self.update_canvas_display()

    def apply_theme(self):
        if self.current_theme == "dark":
            # Google Gemini Dark Palette (#131314 Obsidian, #1E1F20 Surface, #282A2C Card, #A8C7FA Blue, #D0BCFF Violet)
            stylesheet = """
                QWidget {
                    background-color: #131314;
                    color: #E3E3E3;
                    font-family: 'Segoe UI', -apple-system, sans-serif;
                    font-size: 12px;
                }
                #headerTitle {
                    font-size: 16px;
                    font-weight: 900;
                    color: #A8C7FA;
                    letter-spacing: 0.6px;
                }
                #headerSubtitle {
                    font-size: 11px;
                    color: #C4C7C5;
                }
                #panelCard {
                    background-color: #1E1F20;
                    border: 1px solid #37393B;
                    border-radius: 12px;
                }
                #canvasViewport {
                    background-color: #0E0E0F;
                    border-radius: 10px;
                }
                #lblMonoInfo {
                    font-family: 'Consolas', monospace;
                    font-size: 11px;
                    color: #A8C7FA;
                    font-weight: bold;
                }
                #metricsHeader {
                    font-size: 11px;
                    font-weight: bold;
                    color: #C4C7C5;
                    letter-spacing: 0.6px;
                }
                #metricCard {
                    background-color: #282A2C;
                    border: 1px solid #37393B;
                    border-radius: 10px;
                }
                #cardTitle {
                    font-size: 10px;
                    color: #8E918F;
                    font-weight: 800;
                    letter-spacing: 0.8px;
                }
                #badgeStateRelax {
                    background-color: #1E1F20;
                    color: #C4EED0;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 8px;
                    border: 1px solid #37393B;
                }
                #badgeStateActive {
                    background-color: #3C1F24;
                    color: #F28B82;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 8px;
                    border: 1px solid #F28B82;
                }
                QPushButton {
                    background-color: #282A2C;
                    color: #E3E3E3;
                    border: 1px solid #37393B;
                    padding: 7px 14px;
                    border-radius: 8px;
                    font-weight: 700;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #333538;
                    border-color: #A8C7FA;
                }
                QPushButton:pressed {
                    background-color: #1E1F20;
                }
                #btnLime {
                    background-color: #A8C7FA;
                    color: #041E49;
                    border: 1px solid #8AB4F8;
                    font-weight: 900;
                }
                #btnLime:hover { background-color: #8AB4F8; }
                #btnPlum {
                    background-color: #D0BCFF;
                    color: #381E72;
                    border: 1px solid #E8DEF8;
                }
                #btnPlum:hover { background-color: #E8DEF8; }
                #btnSage {
                    background-color: #282A2C;
                    color: #C4EED0;
                    border: 1px solid #37393B;
                }
                #btnSage:hover { background-color: #333538; }
                #btnCharcoal {
                    background-color: #1E1F20;
                    color: #C4C7C5;
                    border: 1px solid #37393B;
                }
                #btnCharcoal:hover { background-color: #282A2C; }
                QComboBox {
                    background-color: #1E1F20;
                    color: #E3E3E3;
                    border: 1px solid #37393B;
                    padding: 5px 10px;
                    border-radius: 8px;
                    font-weight: 500;
                }
                QComboBox QAbstractItemView {
                    background-color: #1E1F20;
                    color: #E3E3E3;
                    selection-background-color: #282A2C;
                    border: 1px solid #37393B;
                }
                QSlider::groove:horizontal {
                    height: 6px;
                    background: #37393B;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #A8C7FA;
                    border: 2px solid #131314;
                    width: 16px;
                    margin: -5px 0;
                    border-radius: 8px;
                }
                QProgressBar {
                    border: 1px solid #37393B;
                    border-radius: 6px;
                    background-color: #1E1F20;
                    text-align: center;
                    color: #E3E3E3;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #A8C7FA;
                    border-radius: 5px;
                }
                QCheckBox {
                    color: #C4C7C5;
                    font-weight: 600;
                }
                QStatusBar {
                    background-color: #0E0E0F;
                    color: #A8C7FA;
                    font-family: 'Consolas', monospace;
                    font-weight: bold;
                    border-top: 1px solid #1E1F20;
                }
                QTabWidget::pane {
                    border: 1px solid #37393B;
                    border-radius: 8px;
                    background-color: #1E1F20;
                }
                QTabBar::tab {
                    background-color: #131314;
                    color: #8E918F;
                    padding: 8px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    font-weight: bold;
                }
                QTabBar::tab:selected {
                    background-color: #1E1F20;
                    color: #A8C7FA;
                    border-bottom: 2px solid #A8C7FA;
                }
            """
        else:
            # Google Gemini Light Palette (#F0F4F9 Base, #FFFFFF Surface, #0B57D0 Blue, #7C4DFF Violet)
            stylesheet = """
                QWidget {
                    background-color: #F0F4F9;
                    color: #1F1F1F;
                    font-family: 'Segoe UI', -apple-system, sans-serif;
                    font-size: 12px;
                }
                #headerTitle {
                    font-size: 16px;
                    font-weight: 900;
                    color: #0B57D0;
                    letter-spacing: 0.6px;
                }
                #headerSubtitle {
                    font-size: 11px;
                    color: #444746;
                }
                #panelCard {
                    background-color: #FFFFFF;
                    border: 1px solid #E0E3E7;
                    border-radius: 12px;
                }
                #canvasViewport {
                    background-color: #000000;
                    border-radius: 10px;
                }
                #lblMonoInfo {
                    font-family: 'Consolas', monospace;
                    font-size: 11px;
                    color: #0B57D0;
                    font-weight: bold;
                }
                #metricsHeader {
                    font-size: 11px;
                    font-weight: bold;
                    color: #444746;
                    letter-spacing: 0.6px;
                }
                #metricCard {
                    background-color: #F8FAFD;
                    border: 1px solid #E0E3E7;
                    border-radius: 10px;
                }
                #cardTitle {
                    font-size: 10px;
                    color: #5E5E5E;
                    font-weight: 800;
                    letter-spacing: 0.8px;
                }
                #badgeStateRelax {
                    background-color: #E6F4EA;
                    color: #137333;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 8px;
                    border: 1px solid #CEEAD6;
                }
                #badgeStateActive {
                    background-color: #FCE8E6;
                    color: #C5221F;
                    font-weight: bold;
                    padding: 8px;
                    border-radius: 8px;
                    border: 1px solid #FAD2CF;
                }
                QPushButton {
                    background-color: #FFFFFF;
                    color: #1F1F1F;
                    border: 1px solid #E0E3E7;
                    padding: 7px 14px;
                    border-radius: 8px;
                    font-weight: 700;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #E9EEF6;
                    border-color: #0B57D0;
                }
                QPushButton:pressed {
                    background-color: #D3E3FD;
                }
                #btnLime {
                    background-color: #0B57D0;
                    color: #FFFFFF;
                    border: 1px solid #0842A0;
                    font-weight: 900;
                }
                #btnLime:hover { background-color: #0842A0; }
                #btnPlum {
                    background-color: #7C4DFF;
                    color: #FFFFFF;
                    border: 1px solid #651FFF;
                }
                #btnPlum:hover { background-color: #651FFF; }
                #btnSage {
                    background-color: #FFFFFF;
                    color: #137333;
                    border: 1px solid #CEEAD6;
                }
                #btnSage:hover { background-color: #E6F4EA; }
                #btnCharcoal {
                    background-color: #FFFFFF;
                    color: #1F1F1F;
                    border: 1px solid #E0E3E7;
                }
                #btnCharcoal:hover { background-color: #E9EEF6; }
                QComboBox {
                    background-color: #FFFFFF;
                    color: #1F1F1F;
                    border: 1px solid #E0E3E7;
                    padding: 5px 10px;
                    border-radius: 8px;
                    font-weight: 500;
                }
                QComboBox QAbstractItemView {
                    background-color: #FFFFFF;
                    color: #1F1F1F;
                    selection-background-color: #D3E3FD;
                    selection-color: #041E49;
                    border: 1px solid #E0E3E7;
                }
                QSlider::groove:horizontal {
                    height: 6px;
                    background: #E0E3E7;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #0B57D0;
                    border: 2px solid #FFFFFF;
                    width: 16px;
                    margin: -5px 0;
                    border-radius: 8px;
                }
                QProgressBar {
                    border: 1px solid #E0E3E7;
                    border-radius: 6px;
                    background-color: #E9EEF6;
                    text-align: center;
                    color: #1F1F1F;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #0B57D0;
                    border-radius: 5px;
                }
                QCheckBox {
                    color: #1F1F1F;
                    font-weight: 600;
                }
                QStatusBar {
                    background-color: #E9EEF6;
                    color: #0B57D0;
                    font-family: 'Consolas', monospace;
                    font-weight: bold;
                    border-top: 1px solid #E0E3E7;
                }
                QTabWidget::pane {
                    border: 1px solid #E0E3E7;
                    border-radius: 8px;
                    background-color: #FFFFFF;
                }
                QTabBar::tab {
                    background-color: #F0F4F9;
                    color: #5E5E5E;
                    padding: 8px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    font-weight: bold;
                }
                QTabBar::tab:selected {
                    background-color: #FFFFFF;
                    color: #0B57D0;
                    border-bottom: 2px solid #0B57D0;
                }
            """

        self.setStyleSheet(stylesheet)
        self.kinematics_widget.set_theme_and_lang(self.current_theme, self.current_lang)

    # -----------------------------------------------------------------
    # LOCALIZATION ENGINE (ENGLISH VS 繁體中文)
    # -----------------------------------------------------------------
    def on_language_changed(self, idx):
        self.current_lang = self.combo_lang.currentData()
        self.retranslate_ui()
        self.kinematics_widget.set_theme_and_lang(self.current_theme, self.current_lang)

    def retranslate_ui(self):
        t = TRANSLATIONS[self.current_lang]
        self.lbl_title.setText(t["app_title"])
        self.lbl_subtitle.setText(t["app_subtitle"])
        self.lbl_preset_tag.setText(t["preset_label"])

        self.btn_sample_hub.setText(t["btn_sample_hub"])
        self.btn_select_bin.setText(t["btn_select_bin"])
        self.btn_run_matlab.setText(t["btn_run_matlab"])
        self.btn_select_folder.setText(t["btn_select_folder"])
        self.btn_run_segmentation.setText(t["btn_run_ai"])

        self.cb_mask.setText(t["cb_mask"])
        self.cb_sup.setText(t["cb_sup"])
        self.cb_deep.setText(t["cb_deep"])
        self.cb_grid.setText(t["cb_grid"])

        curr_v = self.combo_view.currentIndex()
        self.combo_view.blockSignals(True)
        self.combo_view.clear()
        self.combo_view.addItems([t["view_overlay"], t["view_side_by_side"], t["view_raw"]])
        self.combo_view.setCurrentIndex(max(0, curr_v))
        self.combo_view.blockSignals(False)

        curr_fps = self.combo_fps.currentIndex()
        self.combo_fps.blockSignals(True)
        self.combo_fps.clear()
        self.combo_fps.addItems([t["speed_30"], t["speed_60"], t["speed_112"], t["speed_15"]])
        self.combo_fps.setCurrentIndex(max(0, curr_fps))
        self.combo_fps.blockSignals(False)

        self.btn_play.setText(t["btn_pause"] if self.is_playing else t["btn_play"])

        self.lbl_metrics_header.setText(t["metrics_title"])
        self.card_current.title_label.setText(t["card_current"].upper())
        self.card_rest.title_label.setText(t["card_rest"].upper())
        self.card_peak.title_label.setText(t["card_peak"].upper())
        self.card_delta.title_label.setText(t["card_delta"].upper())

        self.lbl_state_badge.setText(t["status_relax"])

        self.btn_export_csv.setText(t["btn_export_csv"])
        self.btn_save_fig.setText(t["btn_save_fig"])
        self.btn_open_folder.setText(t["btn_open_folder"])

        self.statusBar().showMessage(t["ready_status"])

    # -----------------------------------------------------------------
    # INGESTION & DATASET DISCOVERY
    # -----------------------------------------------------------------
    def discover_existing_trials(self):
        t = TRANSLATIONS[self.current_lang]
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()
        self.combo_presets.addItem(t["preset_prompt"], "")

        images_base = os.path.join(PARENT_ROOT, "dataset", "images")
        if os.path.exists(images_base):
            for subj in sorted(os.listdir(images_base)):
                subj_path = os.path.join(images_base, subj)
                if os.path.isdir(subj_path) and not subj.lower().startswith("subject"):
                    # Check 1 level deep (direct frames)
                    direct_frames = glob.glob(os.path.join(subj_path, "frame_*.png"))
                    if direct_frames:
                        self.combo_presets.addItem(f"{subj} ({len(direct_frames)} frames)", subj_path)

                    # Check 2 levels deep (trials subfolder)
                    for trial in sorted(os.listdir(subj_path)):
                        t_path = os.path.join(subj_path, trial)
                        if os.path.isdir(t_path):
                            frames = glob.glob(os.path.join(t_path, "frame_*.png"))
                            if len(frames) > 0:
                                self.combo_presets.addItem(f"{subj} / {trial} ({len(frames)} frames)", t_path)

        self.combo_presets.blockSignals(False)

    def on_preset_selected(self, idx):
        path = self.combo_presets.currentData()
        if path and os.path.exists(path):
            self.load_images_from_folder(path)

    def open_sample_data_dialog(self):
        dlg = SampleDataAndPresetDialog(self, current_lang=self.current_lang, current_theme=self.current_theme)
        dlg.load_folder_signal.connect(self.load_images_from_folder)
        dlg.preset_applied_signal.connect(self.apply_ultrasound_preset)
        dlg.exec_()

    def apply_ultrasound_preset(self, preset):
        if not preset:
            return
        self.active_preset_name = preset.get("name", "Custom Preset")
        self.pixel_spacing_mm = float(preset.get("pixel_spacing_mm", self.pixel_spacing_mm))
        self.fps = float(preset.get("native_fps", self.fps))

        # Update FPS combo box index if matching
        if abs(self.fps - 112.0) < 1.0:
            self.combo_fps.setCurrentIndex(2)
        elif abs(self.fps - 60.0) < 1.0:
            self.combo_fps.setCurrentIndex(1)
        elif abs(self.fps - 30.0) < 1.0:
            self.combo_fps.setCurrentIndex(0)

        self.statusBar().showMessage(f"Preset Applied: {self.active_preset_name} | Spacing: {self.pixel_spacing_mm} mm/px | FPS: {self.fps}")
        self.update_canvas_display()

    def on_select_bin_file(self):
        default_dir = os.path.join(PARENT_ROOT, "BIN FILE")
        if not os.path.exists(default_dir):
            default_dir = PARENT_ROOT

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Telemed ArtUs BIN File", default_dir, "Telemed BIN (*.bin)"
        )
        if file_path:
            self.current_bin_file = file_path
            self.statusBar().showMessage(f"BIN selected: {os.path.basename(file_path)}")

    def on_run_matlab_extraction(self):
        t = TRANSLATIONS[self.current_lang]
        if not self.current_bin_file:
            QMessageBox.warning(self, "Warning", t["warning_no_bin"])
            return

        self.statusBar().showMessage(t["extracting_status"])
        self.btn_run_matlab.setEnabled(False)
        self.progress_bar.setValue(10)

        self.matlab_worker = MatlabExtractionThread(self.current_bin_file)
        self.matlab_worker.progress_signal.connect(self.on_worker_progress)
        self.matlab_worker.finished_signal.connect(self.on_matlab_finished)
        self.matlab_worker.start()

    def on_worker_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.statusBar().showMessage(msg)

    def on_matlab_finished(self, success, out_dir, msg):
        self.btn_run_matlab.setEnabled(True)
        if success:
            QMessageBox.information(self, "Extraction Finished", f"{msg}\n\nFolder: {out_dir}")
            self.load_images_from_folder(out_dir)
            self.discover_existing_trials()
        else:
            QMessageBox.critical(self, "Extraction Error", msg)

    def on_select_images_folder(self):
        default_dir = os.path.join(PARENT_ROOT, "dataset", "images")
        dir_path = QFileDialog.getExistingDirectory(self, "Select Ultrasound Frames Folder", default_dir)
        if dir_path:
            self.load_images_from_folder(dir_path)

    def load_images_from_folder(self, folder_path):
        t = TRANSLATIONS[self.current_lang]
        self.current_image_dir = folder_path
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "frame_*.png")))
        if not self.image_paths:
            self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))

        if not self.image_paths:
            QMessageBox.warning(self, "Warning", t["warning_empty_folder"].format(folder=folder_path))
            return

        n_frames = len(self.image_paths)
        self.slider_frame.setRange(0, n_frames - 1)
        self.slider_frame.setValue(0)
        self.current_frame_idx = 0

        # Reset caches
        self.masks_cache = [None] * n_frames
        self.superficial_depth_mm = [14.0] * n_frames
        self.deep_depth_mm = [28.0] * n_frames
        self.thickness_mm = [14.0] * n_frames
        self.timestamps = np.array([i / self.fps for i in range(n_frames)])

        # Read metadata if present
        meta_file = os.path.join(folder_path, "metadata.txt")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as fm:
                    for line in fm:
                        if "FPS:" in line:
                            self.fps = float(line.split(":")[1].strip())
                        elif "Pixel_Spacing_mm:" in line:
                            self.pixel_spacing_mm = float(line.split(":")[1].strip())
            except Exception:
                pass

        self.update_canvas_display()
        self.kinematics_widget.update_data(self.timestamps, self.thickness_mm)

        self.statusBar().showMessage(t["folder_loaded"].format(folder=os.path.basename(folder_path), count=n_frames))

    # -----------------------------------------------------------------
    # INFERENSI SEGMENTASI AI
    # -----------------------------------------------------------------
    def on_run_ai_segmentation(self):
        t = TRANSLATIONS[self.current_lang]
        if not self.image_paths:
            QMessageBox.warning(self, "Warning", t["warning_no_images"])
            return

        self.btn_run_segmentation.setEnabled(False)
        self.btn_run_segmentation.setText(t["btn_running_ai"])
        self.progress_bar.setValue(5)

        self.ai_worker = SegmentationInferenceThread(
            self.image_paths,
            pixel_spacing_mm=self.pixel_spacing_mm,
            fps=self.fps
        )
        self.ai_worker.progress_signal.connect(self.on_worker_progress)
        self.ai_worker.frame_done_signal.connect(self.on_ai_frame_done)
        self.ai_worker.finished_signal.connect(self.on_ai_finished)
        self.ai_worker.start()

    def on_ai_frame_done(self, idx, sup, deep, thick):
        if idx < len(self.superficial_depth_mm):
            self.superficial_depth_mm[idx] = sup
            self.deep_depth_mm[idx] = deep
            self.thickness_mm[idx] = thick

    def on_ai_finished(self, success, results):
        t = TRANSLATIONS[self.current_lang]
        self.btn_run_segmentation.setEnabled(True)
        self.btn_run_segmentation.setText(t["btn_run_ai"])

        if success:
            self.masks_cache = results["masks"]
            self.superficial_depth_mm = results["superficial_mm"]
            self.deep_depth_mm = results["deep_mm"]
            self.thickness_mm = results["thickness_mm"]
            self.timestamps = results["timestamps"]

            rest_val = float(np.percentile(self.thickness_mm, 10))
            peak_val = float(np.max(self.thickness_mm))
            delta_val = peak_val - rest_val
            strain_pct = (delta_val / rest_val * 100.0) if rest_val > 0 else 0.0

            self.card_rest.val_label.setText(f"{rest_val:.2f} mm")
            self.card_peak.val_label.setText(f"{peak_val:.2f} mm")
            self.card_delta.val_label.setText(f"+{delta_val:.2f} mm (+{strain_pct:.1f}%)")

            self.kinematics_widget.update_data(
                self.timestamps, self.thickness_mm,
                rest_baseline_mm=rest_val, peak_mm=peak_val
            )
            self.update_canvas_display()

            body_txt = t["seg_complete_body"].format(
                count=len(self.image_paths),
                rest=rest_val,
                peak=peak_val,
                delta=delta_val,
                pct=strain_pct,
                fps=self.fps
            )
            QMessageBox.information(self, t["seg_complete_title"], body_txt)
        else:
            QMessageBox.critical(self, "AI Error", f"Segmentation error:\n{results.get('error', 'Unknown')}")

    # -----------------------------------------------------------------
    # PLAYER CINE CONTROLS
    # -----------------------------------------------------------------
    def toggle_play(self):
        t = TRANSLATIONS[self.current_lang]
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self.btn_play.setText(t["btn_play"])
        else:
            if not self.image_paths:
                return
            interval_ms = int(1000.0 / self.fps)
            self.play_timer.start(interval_ms)
            self.is_playing = True
            self.btn_play.setText(t["btn_pause"])

    def on_timer_tick(self):
        if self.current_frame_idx < len(self.image_paths) - 1:
            self.current_frame_idx += 1
        else:
            self.current_frame_idx = 0
        self.slider_frame.setValue(self.current_frame_idx)

    def on_fps_changed(self, idx):
        fps_map = {0: 30.0, 1: 60.0, 2: 112.0, 3: 15.0}
        self.fps = fps_map.get(idx, 60.0)
        if self.is_playing:
            self.play_timer.setInterval(int(1000.0 / self.fps))
        if len(self.image_paths) > 0:
            self.timestamps = np.array([i / self.fps for i in range(len(self.image_paths))])
            self.kinematics_widget.update_data(self.timestamps, self.thickness_mm)

    def on_view_mode_changed(self, idx):
        modes = ["overlay", "side_by_side", "raw"]
        self.view_mode = modes[idx]
        self.update_canvas_display()

    def step_prev(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.slider_frame.setValue(self.current_frame_idx)

    def step_next(self):
        if self.current_frame_idx < len(self.image_paths) - 1:
            self.current_frame_idx += 1
            self.slider_frame.setValue(self.current_frame_idx)

    def seek_to_frame(self, frame_idx):
        if 0 <= frame_idx < len(self.image_paths):
            self.current_frame_idx = frame_idx
            self.slider_frame.setValue(frame_idx)

    def on_slider_moved(self, val):
        self.current_frame_idx = val
        self.update_canvas_display()

    def update_canvas_display(self):
        if not self.image_paths or self.current_frame_idx >= len(self.image_paths):
            return

        t = TRANSLATIONS[self.current_lang]
        f_idx = self.current_frame_idx
        img_path = self.image_paths[f_idx]
        raw_bmode = cv2.imread(img_path)
        if raw_bmode is None:
            return

        h, w, _ = raw_bmode.shape
        t_sec = f_idx / self.fps
        self.lbl_frame_info.setText(f"Frame: {f_idx + 1} / {len(self.image_paths)} | {t_sec:.2f} s")

        mask = self.masks_cache[f_idx] if f_idx < len(self.masks_cache) else None
        th_val = self.thickness_mm[f_idx] if f_idx < len(self.thickness_mm) else 0.0
        sup_d = self.superficial_depth_mm[f_idx] if f_idx < len(self.superficial_depth_mm) else 14.0
        deep_d = self.deep_depth_mm[f_idx] if f_idx < len(self.deep_depth_mm) else 28.0

        self.card_current.val_label.setText(f"{th_val:.2f} mm")
        is_active = th_val > 18.0

        if is_active:
            self.lbl_state_badge.setText(t["status_active"])
            self.lbl_state_badge.setObjectName("badgeStateActive")
            self.lbl_state_badge.setStyleSheet(
                "background-color: #80475E; color: #FFFFFF; font-weight: bold; padding: 8px; border-radius: 8px; border: 1px solid #C7EF00;"
            )
        else:
            self.lbl_state_badge.setText(t["status_relax"])
            self.lbl_state_badge.setObjectName("badgeStateRelax")
            if self.current_theme == "dark":
                self.lbl_state_badge.setStyleSheet(
                    "background-color: #28302C; color: #7E9181; font-weight: bold; padding: 8px; border-radius: 8px; border: 1px solid #3D4A43;"
                )
            else:
                self.lbl_state_badge.setStyleSheet(
                    "background-color: #EAEFEA; color: #5D6D60; font-weight: bold; padding: 8px; border-radius: 8px; border: 1px solid #D5DED7;"
                )

        self.kinematics_widget.set_cursor_frame(f_idx)

        # Rendering Composition with Palette (#7E9181 Sage Mask, #C7EF00 Lime Superficial, #80475E Plum Deep)
        disp_bgr = raw_bmode.copy()

        if self.view_mode == "overlay" and mask is not None:
            # 1. Mask fill: Soft Sage Green [129, 145, 126] BGR
            if self.cb_mask.isChecked():
                overlay = disp_bgr.copy()
                overlay[mask > 0.5] = [129, 145, 126]
                disp_bgr = cv2.addWeighted(overlay, 0.35, disp_bgr, 0.65, 0)

            # 2. Aponeurosis Fascia Lines (Electric Lime #C7EF00 & Muted Plum #80475E)
            if self.cb_sup.isChecked():
                y_sup = int(round(sup_d / self.pixel_spacing_mm))
                cv2.line(disp_bgr, (0, y_sup), (w, y_sup), (0, 239, 199), 2, cv2.LINE_AA)  # BGR for #C7EF00 Lime
            if self.cb_deep.isChecked():
                y_deep = int(round(deep_d / self.pixel_spacing_mm))
                cv2.line(disp_bgr, (0, y_deep), (w, y_deep), (94, 71, 128), 2, cv2.LINE_AA)  # BGR for #80475E Plum

        elif self.view_mode == "side_by_side":
            right_img = raw_bmode.copy()
            if mask is not None:
                right_img[mask > 0.5] = [129, 145, 126]
                y_sup = int(round(sup_d / self.pixel_spacing_mm))
                y_deep = int(round(deep_d / self.pixel_spacing_mm))
                cv2.line(right_img, (0, y_sup), (w, y_sup), (0, 239, 199), 2, cv2.LINE_AA)
                cv2.line(right_img, (0, y_deep), (w, y_deep), (94, 71, 128), 2, cv2.LINE_AA)
            disp_bgr = np.hstack([raw_bmode, right_img])

        # Depth Guidelines (Muted Guidelines)
        if self.cb_grid.isChecked():
            render_w = disp_bgr.shape[1]
            for d in [10, 20, 30, 40, 50, 60, 70, 80]:
                y_px = int(round(d / self.pixel_spacing_mm))
                if y_px < h:
                    cv2.line(disp_bgr, (0, y_px), (render_w, y_px), (60, 75, 68), 1, cv2.LINE_AA)
                    cv2.putText(disp_bgr, f"{d}mm", (6, y_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 195, 185), 1)

        # Convert to QPixmap
        rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        rh, rw, ch = rgb.shape
        q_img = QImage(rgb.data, rw, rh, ch * rw, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img)

        scaled_pix = pix.scaled(
            self.canvas_label.width() - 8, self.canvas_label.height() - 8,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.canvas_label.setPixmap(scaled_pix)

    # -----------------------------------------------------------------
    # EXPORT METRICS & REPORTING
    # -----------------------------------------------------------------
    def export_csv(self):
        t = TRANSLATIONS[self.current_lang]
        if len(self.timestamps) == 0:
            QMessageBox.warning(self, "Warning", t["warning_no_data"])
            return

        trial_name = os.path.basename(self.current_image_dir) if self.current_image_dir else "Biceps_Brachii"
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "biceps_segmentation", trial_name)
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, f"{trial_name}_kinematics_telemetry.csv")
        df = pd.DataFrame({
            "Frame_Index": np.arange(1, len(self.timestamps) + 1),
            "Time_Seconds": np.round(self.timestamps, 4),
            "Biceps_Thickness_mm": np.round(self.thickness_mm, 3),
            "Superficial_Fascia_Depth_mm": np.round(self.superficial_depth_mm, 3),
            "Deep_Fascia_Depth_mm": np.round(self.deep_depth_mm, 3),
            "Acquisition_FPS": self.fps,
            "Pixel_Spacing_mm": self.pixel_spacing_mm,
            "Transducer_Preset": self.active_preset_name
        })
        df.to_csv(csv_path, index=False)
        QMessageBox.information(self, "Export Successful", t["export_csv_success"].format(path=csv_path))

    def save_visualization_figure(self):
        t = TRANSLATIONS[self.current_lang]
        if len(self.timestamps) == 0:
            QMessageBox.warning(self, "Warning", t["warning_no_data"])
            return

        trial_name = os.path.basename(self.current_image_dir) if self.current_image_dir else "Biceps_Brachii"
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "biceps_segmentation", trial_name)
        os.makedirs(out_dir, exist_ok=True)

        fig_path = os.path.join(out_dir, f"{trial_name}_kinematics_summary.png")
        self.kinematics_widget.fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        QMessageBox.information(self, "Figure Saved", t["export_fig_success"].format(path=fig_path))

    def open_output_folder(self):
        trial_name = os.path.basename(self.current_image_dir) if self.current_image_dir else "Biceps_Brachii"
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "biceps_segmentation", trial_name)
        os.makedirs(out_dir, exist_ok=True)
        os.startfile(out_dir)


def main():
    app = QApplication(sys.argv)
    window = BicepsPipelineStudio()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
