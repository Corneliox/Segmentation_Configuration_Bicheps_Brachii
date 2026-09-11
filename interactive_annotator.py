"""
=====================================================================
AI-ASSISTED INTERACTIVE ANNOTATOR (Tkinter Native GUI + PyTorch CUDA)
With Adobe Animate-Style Onion Skinning (Ghost Layer)
=====================================================================
Features:
1. Native Tkinter GUI (Zero OpenCV GUI compatibility issues).
2. Live interactive brush/eraser for Biceps Brachii polygon/mask.
3. Adobe Animate Ghost Layer (Onion Skin): displays frame t-1 mask as
   a translucent amber/orange ghost to track motion and maintain consistency.
4. Instant Ghost Copy ([K]): copy previous frame's mask to current frame.
5. Continuous Active Learning: on-demand or auto incremental fine-tuning.
6. Auto-proposal prediction for new frames via U-Net.
=====================================================================
"""

import os
import glob
import sys
import yaml
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
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
        self.root.title("AI-Assisted Interactive Annotator | Biceps Brachii (Onion Skin Edition)")
        self.root.geometry("1320x880")
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

        # Interactive & Ghost States
        self.brush_size = 10
        self.drawing = False
        self.erasing = False
        self.current_idx = 0
        self.validated_count = 0
        self.auto_retrain_interval = 10

        # Onion Skinning (Ghost Layer)
        self.ghost_enabled = True
        self.prev_mask = None

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

        # Accent buttons
        style.configure("Accent.TButton", background="#007acc", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton", background=[("active", "#0098ff")])

        # Ghost buttons
        style.configure("GhostOn.TButton", background="#e65100", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.map("GhostOn.TButton", background=[("active", "#f57c00")])
        style.configure("GhostOff.TButton", background="#333333", foreground="#888888", font=("Segoe UI", 9, "bold"))
        style.map("GhostOff.TButton", background=[("active", "#444444")])

    def create_ui(self):
        # Master Layout: Top Control Bar, Center Canvas, Bottom Timeline & Shortcuts
        top_bar = tk.Frame(self.root, bg="#1c1c1c", height=50, padx=15, pady=8)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        self.lbl_title = ttk.Label(top_bar, text="🔬 AI-ASSISTED ANNOTATOR (ONION SKIN)", style="Header.TLabel")
        self.lbl_title.pack(side=tk.LEFT, padx=(0, 15))

        self.lbl_status = ttk.Label(top_bar, text="Status: Ready", style="Status.TLabel", padding=(10, 3))
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        # Legend indicator
        legend_frame = tk.Frame(top_bar, bg="#1c1c1c")
        legend_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(legend_frame, text="🟢 Active Mask", foreground="#00e676", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="|", foreground="#555555").pack(side=tk.LEFT, padx=3)
        ttk.Label(legend_frame, text="🟠 Ghost (Frame t-1)", foreground="#ff9800", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="|", foreground="#555555").pack(side=tk.LEFT, padx=3)
        self.lbl_device = ttk.Label(legend_frame, text=f"GPU: {self.device.type.upper()}", font=("Consolas", 9), foreground="#888888")
        self.lbl_device.pack(side=tk.LEFT, padx=5)

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

        self.btn_prev = ttk.Button(btn_row, text="◀ Prev (P)", width=12, command=self.prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=3)

        self.btn_next = ttk.Button(btn_row, text="Confirm & Next ▶ (SPACE)", style="Accent.TButton", width=22, command=self.next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=6)

        # Ghost / Onion Skin Controls
        self.btn_ghost = ttk.Button(btn_row, text="👻 Ghost: ON (G)", style="GhostOn.TButton", width=16, command=self.toggle_ghost)
        self.btn_ghost.pack(side=tk.LEFT, padx=4)

        self.btn_copy = ttk.Button(btn_row, text="📋 Copy Ghost (K)", width=16, command=self.copy_prev_mask)
        self.btn_copy.pack(side=tk.LEFT, padx=4)

        self.btn_retrain = ttk.Button(btn_row, text="⚡ Retrain U-Net (T)", width=18, command=self.trigger_train_thread)
        self.btn_retrain.pack(side=tk.LEFT, padx=4)

        self.btn_clear = ttk.Button(btn_row, text="🧹 Clear (C)", width=12, command=self.clear_mask)
        self.btn_clear.pack(side=tk.LEFT, padx=4)

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
            text="[Klik Kiri]: Gambar | [Klik Kanan]: Hapus | [G]: Toggle Ghost | [K]: Salin Ghost Mask | [SPACE]: Simpan & Lanjut | [T]: Latih U-Net",
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
        self.root.bind("<g>", lambda e: self.toggle_ghost())
        self.root.bind("<G>", lambda e: self.toggle_ghost())
        self.root.bind("<k>", lambda e: self.copy_prev_mask())
        self.root.bind("<K>", lambda e: self.copy_prev_mask())
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

    def toggle_ghost(self):
        self.ghost_enabled = not self.ghost_enabled
        if self.ghost_enabled:
            self.btn_ghost.config(text="👻 Ghost: ON (G)", style="GhostOn.TButton")
            self.lbl_status.config(text="Status: Ghost Layer ON (Onion Skin)", foreground="#ff9800")
        else:
            self.btn_ghost.config(text="👻 Ghost: OFF (G)", style="GhostOff.TButton")
            self.lbl_status.config(text="Status: Ghost Layer OFF", foreground="#888888")
        self.render_overlay()

    def copy_prev_mask(self):
        if self.prev_mask is not None and np.any(self.prev_mask > 0):
            self.current_mask = self.prev_mask.copy()
            self.lbl_status.config(text="Status: Mask Disalin dari Ghost (Frame t-1)", foreground="#ffd54f")
            self.render_overlay()
        else:
            self.lbl_status.config(text="Status: Tidak ada Ghost mask pada frame sebelumnya", foreground="#ffb74d")

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

        # Load Ghost Mask from previous frame (frame idx - 1)
        if self.current_idx > 0:
            prev_img_path = self.image_paths[self.current_idx - 1]
            prev_rel = os.path.relpath(prev_img_path, self.img_base)
            prev_mask_path = os.path.join(self.mask_base, prev_rel)
            if os.path.exists(prev_mask_path):
                self.prev_mask = cv2.imread(prev_mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                self.prev_mask = None
        else:
            self.prev_mask = None

        if os.path.exists(mask_path):
            self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.lbl_status.config(text="Status: Validated Ground Truth (Saved)", foreground="#00e676")
        else:
            # Predict proposal with U-Net
            if self.is_trained:
                self.current_mask = self.predict_unet(self.current_img_gray)
                self.lbl_status.config(text="Status: AI Auto-Proposal (Check & Confirm)", foreground="#00e5ff")
            else:
                self.current_mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
                self.lbl_status.config(text="Status: Manual Cold-Start (Draw or [K] Copy Ghost)", foreground="#ffb74d")

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

        # Build RGB display
        bgr = cv2.cvtColor(self.current_img_gray, cv2.COLOR_GRAY2BGR)
        overlay = bgr.copy()

        # 1. GHOST LAYER (Frame t-1) in Warm Amber/Orange
        if self.ghost_enabled and self.prev_mask is not None and np.any(self.prev_mask > 0):
            ghost_layer = np.zeros_like(bgr)
            ghost_layer[self.prev_mask > 0] = [0, 140, 255]  # Amber/Orange in BGR
            overlay = cv2.addWeighted(overlay, 1.0, ghost_layer, 0.35, 0)

            # Draw Ghost Contour (Amber/Orange line)
            ghost_contours, _ = cv2.findContours(self.prev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, ghost_contours, -1, (0, 140, 255), 2)

        # 2. CURRENT ACTIVE MASK (Frame t) in Neon Green
        if self.current_mask is not None and np.any(self.current_mask > 0):
            curr_layer = np.zeros_like(bgr)
            curr_layer[self.current_mask > 0] = [0, 255, 0]  # Neon Green
            overlay = cv2.addWeighted(overlay, 1.0, curr_layer, 0.40, 0)

            # Draw Current Contour (Cyan line)
            curr_contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, curr_contours, -1, (255, 255, 0), 2)

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
