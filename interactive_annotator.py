"""
=====================================================================
AI-ASSISTED INTERACTIVE ANNOTATOR (Tkinter Native GUI + PyTorch CUDA)
=====================================================================
Semi-Automated Reinforcement Annotation Workflow:
1. Native Tkinter GUI (Compatible with all OpenCV/Python versions).
2. Live interactive brush/eraser for Biceps Brachii polygon/mask.
3. On-demand or automatic incremental U-Net fine-tuning with GPU CUDA.
4. Auto-proposal prediction for upcoming frames.
=====================================================================
"""

import os
import glob
import sys
import yaml
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import build_model
from src.utils import CombinedLoss


class InteractiveAnnotatorTk:
    def __init__(self, root, config_path="configs/config.yaml"):
        self.root = root
        self.root.title("AI-Assisted Interactive Annotator | Biceps Brachii")
        self.root.geometry("1280x860")
        self.root.configure(bg="#121212")

        # Bring window to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Menggunakan Device: {self.device}")

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(self.base_dir)

        self.img_base = os.path.join(self.parent_dir, "dataset", "images")
        self.mask_base = os.path.join(self.parent_dir, "dataset", "masks")
        self.weights_dir = os.path.join(self.base_dir, "weights")
        os.makedirs(self.mask_base, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

        self.model_checkpoint = os.path.join(self.weights_dir, "interactive_unet.pth")
        self.model = build_model(self.cfg).to(self.device)
        self.is_trained = False
        self.is_training_now = False

        if os.path.exists(self.model_checkpoint):
            try:
                ckpt = torch.load(self.model_checkpoint, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
                self.is_trained = True
                print(f"[INFO] Berhasil memuat bobot model terlatih dari: {self.model_checkpoint}")
            except Exception as e:
                print(f"[WARNING] Gagal memuat bobot lama ({e}), memulai model baru...")

        # Interactive states
        self.brush_size = 10
        self.drawing = False
        self.erasing = False
        self.current_idx = 0
        self.validated_count = 0
        self.auto_retrain_interval = 10

        self.image_paths = sorted(glob.glob(os.path.join(self.img_base, "**", "frame_*.png"), recursive=True))
        self.current_img_gray = None
        self.current_mask = None  # NumPy (H, W) uint8 0 or 255
        self.orig_h = 512
        self.orig_w = 219

        self.setup_styles()
        self.create_ui()
        self.bind_events()

        if self.image_paths:
            self.load_frame(0)
        else:
            messagebox.showwarning("No Images", f"Tidak ada frame ditemukan di:\n{self.img_base}")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#181818", foreground="#e0e0e0")
        style.configure("TLabel", background="#181818", foreground="#e0e0e0", font=("Segoe UI", 9))
        style.configure("Header.TLabel", background="#181818", foreground="#00e5ff", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", background="#222222", foreground="#00e676", font=("Consolas", 10, "bold"))
        style.configure("TButton", background="#2c2c2c", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.map("TButton", background=[("active", "#007acc")])
        style.configure("Accent.TButton", background="#007acc", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#0098ff")])

    def create_ui(self):
        # Master Layout: Top Control Bar, Center Canvas, Bottom Timeline & Shortcuts
        top_bar = tk.Frame(self.root, bg="#1c1c1c", height=50, padx=15, pady=8)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        self.lbl_title = ttk.Label(top_bar, text="🔬 AI-ASSISTED ANNOTATOR (ACTIVE LEARNING)", style="Header.TLabel")
        self.lbl_title.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_status = ttk.Label(top_bar, text="Status: Ready", style="Status.TLabel", padding=(10, 3))
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        self.lbl_device = ttk.Label(top_bar, text=f"Device: {self.device.type.upper()}", font=("Consolas", 9), foreground="#888888")
        self.lbl_device.pack(side=tk.RIGHT, padx=10)

        # Center Display Frame
        center_frame = tk.Frame(self.root, bg="#0a0a0a")
        center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Canvas for Interactive Drawing
        self.canvas = tk.Canvas(center_frame, bg="#000000", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Control Panel
        bot_bar = tk.Frame(self.root, bg="#1c1c1c", padx=15, pady=10)
        bot_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Row 1: Actions & Navigation Buttons
        btn_row = tk.Frame(bot_bar, bg="#1c1c1c")
        btn_row.pack(fill=tk.X, pady=(0, 6))

        self.btn_prev = ttk.Button(btn_row, text="◀ Prev Frame (P)", width=16, command=self.prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=3)

        self.btn_next = ttk.Button(btn_row, text="Confirm & Next ▶ (SPACE)", style="Accent.TButton", width=25, command=self.next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=10)

        self.btn_retrain = ttk.Button(btn_row, text="⚡ Retrain U-Net Now (T)", width=22, command=self.trigger_train_thread)
        self.btn_retrain.pack(side=tk.LEFT, padx=5)

        self.btn_clear = ttk.Button(btn_row, text="🧹 Clear Mask (C)", width=16, command=self.clear_mask)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # Brush size controls
        brush_frame = tk.Frame(btn_row, bg="#1c1c1c")
        brush_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Label(brush_frame, text="Brush Size ([ / ]):", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=3)
        self.lbl_brush = ttk.Label(brush_frame, text=f"{self.brush_size}px", font=("Consolas", 10, "bold"), foreground="#00e5ff")
        self.lbl_brush.pack(side=tk.LEFT, padx=3)

        # Row 2: HUD Help Text
        hud_row = tk.Frame(bot_bar, bg="#1c1c1c")
        hud_row.pack(fill=tk.X)

        self.lbl_hud = ttk.Label(
            hud_row, 
            text="[Klik Kiri]: Gambar Area Biceps | [Klik Kanan]: Hapus | [SPACE]: Simpan & Lanjut | [T]: Latih U-Net | [C]: Hapus Mask",
            font=("Segoe UI", 9), 
            foreground="#aaaaaa"
        )
        self.lbl_hud.pack(side=tk.LEFT)

        self.lbl_progress = ttk.Label(hud_row, text="Frame: 0/0", font=("Consolas", 9, "bold"), foreground="#00e5ff")
        self.lbl_progress.pack(side=tk.RIGHT)

    def bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.canvas.bind("<ButtonPress-3>", self.on_right_down)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_up)

        self.root.bind("<space>", lambda e: self.next_frame())
        self.root.bind("<Return>", lambda e: self.next_frame())
        self.root.bind("<p>", lambda e: self.prev_frame())
        self.root.bind("<P>", lambda e: self.prev_frame())
        self.root.bind("<t>", lambda e: self.trigger_train_thread())
        self.root.bind("<T>", lambda e: self.trigger_train_thread())
        self.root.bind("<c>", lambda e: self.clear_mask())
        self.root.bind("<C>", lambda e: self.clear_mask())
        self.root.bind("<bracketleft>", lambda e: self.adjust_brush(-2))
        self.root.bind("<bracketright>", lambda e: self.adjust_brush(2))
        self.canvas.bind("<Configure>", lambda e: self.render_overlay())

    def adjust_brush(self, delta):
        self.brush_size = max(2, min(50, self.brush_size + delta))
        self.lbl_brush.config(text=f"{self.brush_size}px")

    def canvas_to_mask_coords(self, cx, cy):
        if not hasattr(self, 'render_meta') or self.render_meta is None:
            return None, None
        off_x, off_y, scale = self.render_meta
        mx = (cx - off_x) / scale
        my = (cy - off_y) / scale
        if 0 <= mx < self.orig_w and 0 <= my < self.orig_h:
            return int(round(mx)), int(round(my))
        return None, None

    def draw_brush(self, cx, cy, color):
        mx, my = self.canvas_to_mask_coords(cx, cy)
        if mx is not None and self.current_mask is not None:
            cv2.circle(self.current_mask, (mx, my), self.brush_size, color, -1)
            self.render_overlay()

    def on_left_down(self, event):
        self.drawing = True
        self.draw_brush(event.x, event.y, 255)

    def on_left_drag(self, event):
        if self.drawing:
            self.draw_brush(event.x, event.y, 255)

    def on_right_down(self, event):
        self.erasing = True
        self.draw_brush(event.x, event.y, 0)

    def on_right_drag(self, event):
        if self.erasing:
            self.draw_brush(event.x, event.y, 0)

    def on_mouse_up(self, event):
        self.drawing = False
        self.erasing = False

    def clear_mask(self):
        if self.current_mask is not None:
            self.current_mask.fill(0)
            self.render_overlay()

    def load_frame(self, idx):
        if not self.image_paths:
            return

        self.current_idx = max(0, min(len(self.image_paths) - 1, idx))
        img_path = self.image_paths[self.current_idx]
        rel_path = os.path.relpath(img_path, self.img_base)
        mask_path = os.path.join(self.mask_base, rel_path)

        self.current_img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.current_img_gray is None:
            return

        self.orig_h, self.orig_w = self.current_img_gray.shape

        if os.path.exists(mask_path):
            self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.lbl_status.config(text="Status: Validated Ground Truth (Edited)", foreground="#00e676")
        else:
            # Predict proposal with U-Net
            if self.is_trained:
                self.current_mask = self.predict_unet(self.current_img_gray)
                self.lbl_status.config(text="Status: AI Auto-Proposal (Check & Confirm)", foreground="#00e5ff")
            else:
                self.current_mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
                self.lbl_status.config(text="Status: Manual Cold-Start (Draw 5-10 frames to start AI)", foreground="#ffb74d")

        self.lbl_progress.config(text=f"Frame {self.current_idx + 1}/{len(self.image_paths)}")
        self.render_overlay()

    def predict_unet(self, img_gray):
        self.model.eval()
        h, w = img_gray.shape
        tw = int(round(w / 32.0)) * 32
        th = int(round(h / 32.0)) * 32

        resized = cv2.resize(img_gray, (tw, th), interpolation=cv2.INTER_LINEAR)
        x = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        t_in = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(t_in)
            arr = out.squeeze().cpu().numpy()
            bin_mask = (arr > 0.5).astype(np.uint8) * 255

        return cv2.resize(bin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def render_overlay(self):
        if self.current_img_gray is None or self.current_mask is None:
            return

        c_w = max(50, self.canvas.winfo_width())
        c_h = max(50, self.canvas.winfo_height())
        if c_w < 100: c_w, c_h = 900, 650

        # Build RGB display with transparent green overlay
        bgr = cv2.cvtColor(self.current_img_gray, cv2.COLOR_GRAY2BGR)
        color_layer = np.zeros_like(bgr)
        color_layer[:, :, 1] = self.current_mask  # Green mask

        overlay = cv2.addWeighted(bgr, 0.75, color_layer, 0.35, 0)
        
        # Draw cyan outline contour
        contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

        # Scale to canvas
        scale = min(c_w / self.orig_w, c_h / self.orig_h)
        disp_w = max(1, int(self.orig_w * scale))
        disp_h = max(1, int(self.orig_h * scale))

        disp_resized = cv2.resize(overlay, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(cv2.cvtColor(disp_resized, cv2.COLOR_BGR2RGB))
        self.tk_img = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        off_x = (c_w - disp_w) // 2
        off_y = (c_h - disp_h) // 2
        self.canvas.create_image(off_x, off_y, anchor=tk.NW, image=self.tk_img)

        self.render_meta = (off_x, off_y, scale)

    def next_frame(self):
        if not self.image_paths:
            return

        # Save current mask
        img_path = self.image_paths[self.current_idx]
        rel_path = os.path.relpath(img_path, self.img_base)
        mask_path = os.path.join(self.mask_base, rel_path)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)

        cv2.imwrite(mask_path, self.current_mask)
        self.validated_count += 1
        print(f"[{self.current_idx + 1}/{len(self.image_paths)}] ✓ Mask Tersimpan: {rel_path}")

        # Check auto-retrain trigger
        if self.validated_count % self.auto_retrain_interval == 0:
            self.trigger_train_thread()

        if self.current_idx < len(self.image_paths) - 1:
            self.load_frame(self.current_idx + 1)
        else:
            messagebox.showinfo("Complete!", "Selamat! Seluruh frame telah selesai divalidasi.")

    def prev_frame(self):
        if self.current_idx > 0:
            self.load_frame(self.current_idx - 1)

    def trigger_train_thread(self):
        if self.is_training_now:
            return
        thread = threading.Thread(target=self.run_fine_tuning, daemon=True)
        thread.start()

    def run_fine_tuning(self):
        self.is_training_now = True
        self.lbl_status.config(text="Status: ⚡ Training U-Net with GPU...", foreground="#ff5252")

        mask_files = sorted(glob.glob(os.path.join(self.mask_base, "**", "frame_*.png"), recursive=True))
        if len(mask_files) < 2:
            self.is_training_now = False
            self.lbl_status.config(text="Status: Butuh minimal 2 frame untuk train", foreground="#ffb74d")
            return

        print(f"\n⚡ [FINE-TUNING U-NET] Melatih model dengan {len(mask_files)} frame divalidasi...")

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
            self.is_training_now = False
            return

        self.model.train()
        criterion = CombinedLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0003, weight_decay=1e-5)
        epochs = 15

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
        torch.save({"model_state_dict": self.model.state_dict()}, os.path.join(self.weights_dir, "best_model.pth"))

        print(f"✓ [OK] U-Net Model Updated! (Loss: {total_loss/len(train_samples):.4f})\n")
        self.is_training_now = False
        self.lbl_status.config(text="Status: U-Net Model Updated! (Active)", foreground="#00e676")


def main():
    root = tk.Tk()
    app = InteractiveAnnotatorTk(root)
    root.mainloop()


if __name__ == "__main__":
    main()
