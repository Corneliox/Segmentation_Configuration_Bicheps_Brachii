"""
=====================================================================
INTERACTIVE HUMAN-IN-THE-LOOP ACTIVE ANNOTATOR (PyTorch CUDA + OpenCV)
=====================================================================
Semi-Automated Reinforcement Annotation Workflow:
1. User annotates / approves initial frames.
2. Background / On-Demand U-Net fine-tuning learns muscle boundaries.
3. Model automatically predicts masks for upcoming frames.
4. User acts as curator: fine-tunes with brush, presses SPACE to confirm.
5. Model gets progressively smarter and more accurate with every frame.
=====================================================================
"""

import os
import glob
import sys
import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import existing project modules
from src.model import build_model
from src.utils import CombinedLoss


class InteractiveAnnotator:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[INFO] Menggunakan Device: {self.device}")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.base_dir)

        self.img_base = os.path.join(self.parent_dir, "dataset", "images")
        self.mask_base = os.path.join(self.parent_dir, "dataset", "masks")
        self.weights_dir = os.path.join(self.base_dir, "weights")
        os.makedirs(self.mask_base, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

        self.model_checkpoint = os.path.join(self.weights_dir, "interactive_unet.pth")
        
        # Build U-Net
        self.model = build_model(self.cfg).to(self.device)
        self.is_trained = False

        if os.path.exists(self.model_checkpoint):
            try:
                ckpt = torch.load(self.model_checkpoint, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
                self.is_trained = True
                print(f"[INFO] Berhasil memuat bobot model terlatih dari: {self.model_checkpoint}")
            except Exception as e:
                print(f"[WARNING] Gagal memuat bobot lama ({e}), memulai model baru...")

        # Interactive states
        self.brush_size = 8
        self.drawing = False
        self.erasing = False
        self.current_mask = None
        self.current_img = None
        self.current_img_disp = None
        self.auto_retrain_interval = 10  # Latih ulang setiap 10 frame yang disetujui
        self.validated_count = 0

    def discover_images(self):
        """Mencari seluruh frame gambar di dataset/images."""
        img_paths = sorted(glob.glob(os.path.join(self.img_base, "**", "frame_*.png"), recursive=True))
        return img_paths

    def predict_mask(self, img_gray):
        """Prediksi mask dari model U-Net."""
        if not self.is_trained:
            # Cold start: belum ada prediksi (mask kosong)
            return np.zeros_like(img_gray, dtype=np.uint8)

        self.model.eval()
        h, w = img_gray.shape
        target_w = int(round(w / 32.0)) * 32
        target_h = int(round(h / 32.0)) * 32

        resized = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        tensor_in = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor_in = torch.tensor(tensor_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(tensor_in)
            pred_arr = pred.squeeze().cpu().numpy()
            pred_bin = (pred_arr > 0.5).astype(np.uint8) * 255

        # Resize kembali ke ukuran asli gambar
        pred_mask = cv2.resize(pred_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        return pred_mask

    def fine_tune_model(self, epochs=15):
        """Melatih U-Net secara bertahap menggunakan seluruh frame yang sudah divalidasi."""
        # Kumpulkan seluruh pasangan image & mask yang sudah ada di dataset/masks
        mask_files = sorted(glob.glob(os.path.join(self.mask_base, "**", "frame_*.png"), recursive=True))
        if len(mask_files) < 2:
            print("[INFO] Butuh minimal 2 frame beranotasi untuk melatih model.")
            return

        print(f"\n⚡ [FINE-TUNING U-NET] Melatih model dengan {len(mask_files)} frame yang telah divalidasi...")
        
        train_samples = []
        for mf in mask_files:
            rel = os.path.relpath(mf, self.mask_base)
            img_path = os.path.join(self.img_base, rel)
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                msk = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
                if img is not None and msk is not None:
                    train_samples.append((img, msk))

        if not train_samples:
            return

        self.model.train()
        criterion = CombinedLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0003, weight_decay=1e-5)

        for ep in range(epochs):
            total_loss = 0.0
            np.random.shuffle(train_samples)
            for img, msk in train_samples:
                h, w = img.shape
                tw = int(round(w / 32.0)) * 32
                th = int(round(h / 32.0)) * 32

                img_r = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                msk_r = cv2.resize(msk, (tw, th), interpolation=cv2.INTER_NEAREST)

                x = (img_r.astype(np.float32) / 255.0 - 0.5) / 0.5
                y = (msk_r > 127).astype(np.float32)

                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

                optimizer.zero_grad()
                out = self.model(x_t)
                loss = criterion(out, y_t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.is_trained = True
        torch.save({"model_state_dict": self.model.state_dict()}, self.model_checkpoint)
        # Also copy to best_model.pth
        best_pth = os.path.join(self.weights_dir, "best_model.pth")
        torch.save({"model_state_dict": self.model.state_dict()}, best_pth)
        print(f"✓ [OK] Model U-Net berhasil di-update! (Loss: {total_loss/len(train_samples):.4f})\n")

    def mouse_callback(self, event, x, y, flags, param):
        if self.current_mask is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.current_mask, (x, y), self.brush_size, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.current_mask, (x, y), self.brush_size, 255, -1)
            elif self.erasing:
                cv2.circle(self.current_mask, (x, y), self.brush_size + 4, 0, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
            cv2.circle(self.current_mask, (x, y), self.brush_size + 4, 0, -1)
        elif event == cv2.EVENT_RBUTTONUP:
            self.erasing = False

    def run(self):
        img_paths = self.discover_images()
        if not img_paths:
            print(f"[ERROR] Tidak ada gambar ditemukan di {self.img_base}")
            return

        window_name = "Semi-Automatic AI Interactive Annotator | Biceps Brachii"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1100, 750)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n====================================================================")
        print("         AI-ASSISTED INTERACTIVE ANNOTATION TOOL READY              ")
        print("====================================================================")
        print(" KONTROL:")
        print("  • [Klik Kiri + Geser]  : Gambar / Tambah area Biceps (Brush)")
        print("  • [Klik Kanan + Geser] : Hapus / Koreksi area")
        print("  • [ [ / ] ]            : Perkecil / Perbesar ukuran kuas")
        print("  • [C]                  : Clear / Kosongkan mask pada frame ini")
        print("  • [T]                  : Latih ulang (Train) U-Net sekarang secara manual")
        print("  • [SPACE]              : Simpan Mask, Reinforce Model & Lanjut ke Frame Berikutnya")
        print("  • [ESC]                : Keluar dari program")
        print("====================================================================\n")

        for idx, img_path in enumerate(img_paths):
            rel_path = os.path.relpath(img_path, self.img_base)
            mask_save_path = os.path.join(self.mask_base, rel_path)
            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)

            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            h, w = img_gray.shape

            # Jika mask sudah ada dari sesi sebelumnya, gunakan itu
            if os.path.exists(mask_save_path):
                self.current_mask = cv2.imread(mask_save_path, cv2.IMREAD_GRAYSCALE)
                is_ai_pred = False
            else:
                # Dapatkan proposal prediksi dari U-Net
                self.current_mask = self.predict_mask(img_gray)
                is_ai_pred = self.is_trained

            while True:
                display = img_bgr.copy()
                
                # Overlay Mask Biceps Brachii (Warna Hijau Transparan)
                color_mask = np.zeros_like(display)
                color_mask[:, :, 1] = self.current_mask # Channel hijau
                
                # Alpha blending
                overlay = cv2.addWeighted(display, 0.7, color_mask, 0.35, 0)
                # Garis kontur batas otot warna cyan menyala
                contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

                # Header HUD
                hud_h = 45
                top_banner = np.zeros((hud_h, w, 3), dtype=np.uint8)
                status_ai = "AI AUTO-PREDICT" if is_ai_pred else ("MANUAL ANNOTATION" if not self.is_trained else "EDITED")
                cv2.putText(top_banner, f"[{idx+1}/{len(img_paths)}] {os.path.basename(img_path)} | Status: {status_ai}", 
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 1)
                
                # Footer HUD
                bot_banner = np.zeros((35, w, 3), dtype=np.uint8)
                cv2.putText(bot_banner, f"Brush: {self.brush_size}px ([/]) | SPACE: Save & Next | T: Retrain Model | ESC: Exit",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                full_view = np.vstack([top_banner, overlay, bot_banner])
                cv2.imshow(window_name, full_view)

                key = cv2.waitKey(20) & 0xFF

                if key == 32:  # SPACE: Save and next
                    cv2.imwrite(mask_save_path, self.current_mask)
                    self.validated_count += 1
                    print(f"[{idx+1}/{len(img_paths)}] ✓ Tersimpan: {rel_path}")

                    # Auto-retrain setiap interval
                    if self.validated_count % self.auto_retrain_interval == 0:
                        self.fine_tune_model(epochs=12)
                    break

                elif key == ord('t') or key == ord('T'):
                    # Manual retrain trigger
                    cv2.imwrite(mask_save_path, self.current_mask)
                    self.fine_tune_model(epochs=20)

                elif key == ord('c') or key == ord('C'):
                    # Clear mask
                    self.current_mask = np.zeros((h, w), dtype=np.uint8)

                elif key == ord('['):
                    self.brush_size = max(2, self.brush_size - 2)
                elif key == ord(']'):
                    self.brush_size = min(40, self.brush_size + 2)

                elif key == 27:  # ESC: Exit
                    print("\nKeluar dari anotasi interaktif. Progres telah tersimpan.")
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print("\n🎉 SELURUH FRAME BERHASIL DIANOTASI & MODEL SIAP DIGUNAKAN!")


if __name__ == "__main__":
    annotator = InteractiveAnnotator()
    annotator.run()
