"""
=====================================================================
AI-ASSISTED INTERACTIVE ANNOTATOR (Tkinter Native GUI + PyTorch CUDA)
Dual-Viewport: Dynamic Temporal Filmstrip (Macro Context) + Ghost Studio
=====================================================================
Features:
1. Dual Viewports:
   - Left Viewport: Dynamic 3-frame temporal filmstrip [Past t-stride | Current Raw t | Future t+stride]
     with adjustable stride (1, 5, 10, 20 frames) so you can actually see muscle motion & dynamics!
   - Right Viewport: Interactive annotation studio with Onion Skinning (Ghost Layer from frame n-1)
     and real-time brush/eraser tools.
2. Timeline Scrubber: Interactive horizontal scrubber to instantly jump and inspect any contraction phase.
3. Adobe Animate Ghost Layer (Onion Skin): displays frame n-1 mask as
   a translucent amber/orange ghost to track motion and maintain consistency.
4. Instant Ghost Copy ([K]): copy previous frame's mask to current frame.
5. Continuous Active Learning: on-demand or auto incremental fine-tuning (U-Net).
6. Auto-proposal prediction for new frames via U-Net.
=====================================================================
"""

import os
import glob
import sys
import time
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
        self.root.title("AI-Assisted Interactive Annotator | Dual-Viewport (Macro Context & Ghost Studio)")
        self.root.geometry("1600x920")
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
        self.is_navigating = False
        self.is_scrubbing = False
        self.ai_proposal_enabled = True
        self.last_key_time = 0.0

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

        # Temporal Carousel Settings
        # Default stride 10 frames (~0.33s at 30fps) to clearly reveal muscle contraction & fascial motion!
        self.carousel_stride = 10
        self.carousel_show_masks = False

        # Onion Skinning (Ghost Layer on annotation studio)
        self.ghost_enabled = True
        self.prev_mask = None

        self.image_paths = sorted(glob.glob(os.path.join(self.img_base, "**", "frame_*.png"), recursive=True))
        self.current_img_gray = None
        self.current_mask = None  # NumPy (H, W) uint8 0 or 255
        self.orig_h = 512
        self.orig_w = 219

        # Cached PhotoImages to prevent garbage collection
        self.tk_carousel_img = None
        self.tk_canvas_img = None
        self.render_meta = None

        self.setup_styles()
        self.create_ui()
        self.bind_events()

        # Count existing valid masks
        valid_masks = self.count_valid_masks()
        if valid_masks >= 2:
            self.is_trained = True
            self.lbl_ai_indicator.config(text=f"🤖 U-Net: READY ({valid_masks} masks)", foreground="#00e676")

        if self.image_paths:
            # Auto-resume to first unannotated or blank frame
            start_idx = self.find_first_unannotated_idx()
            self.load_frame(start_idx)
        else:
            messagebox.showwarning("No Images", f"Tidak ada frame ditemukan di:\n{self.img_base}")

    def count_valid_masks(self):
        count = 0
        mask_files = glob.glob(os.path.join(self.mask_base, "**", "frame_*.png"), recursive=True)
        for mf in mask_files:
            m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            if m is not None and np.count_nonzero(m) > 0:
                count += 1
        return count

    def find_first_unannotated_idx(self):
        for idx, p in enumerate(self.image_paths):
            rel = os.path.relpath(p, self.img_base)
            mf = os.path.join(self.mask_base, rel)
            if not os.path.exists(mf):
                return idx
            m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            if m is None or np.count_nonzero(m) == 0:
                return idx
        return 0

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

        # Mini Stride Buttons
        style.configure("Mini.TButton", background="#262626", foreground="#00e5ff", font=("Consolas", 8, "bold"), padding=1)
        style.map("Mini.TButton", background=[("active", "#333333")])

        # AI Predict Button
        style.configure("AIPredict.TButton", background="#6a1b9a", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.map("AIPredict.TButton", background=[("active", "#8e24aa")])

        # Ghost buttons
        style.configure("GhostOn.TButton", background="#e65100", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.map("GhostOn.TButton", background=[("active", "#f57c00")])
        style.configure("GhostOff.TButton", background="#333333", foreground="#888888", font=("Segoe UI", 9, "bold"))
        style.map("GhostOff.TButton", background=[("active", "#444444")])

    def create_ui(self):
        # 1. Top Bar
        top_bar = tk.Frame(self.root, bg="#1c1c1c", height=46, padx=12, pady=5)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        self.lbl_title = ttk.Label(top_bar, text="🔬 AI-ASSISTED ANNOTATOR (ACTIVE U-NET)", style="Header.TLabel")
        self.lbl_title.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_status = ttk.Label(top_bar, text="Status: Ready", style="Status.TLabel", padding=(10, 3))
        self.lbl_status.pack(side=tk.LEFT, padx=8)

        self.lbl_ai_indicator = ttk.Label(top_bar, text="🤖 U-Net: Initializing...", font=("Consolas", 10, "bold"), foreground="#00e5ff")
        self.lbl_ai_indicator.pack(side=tk.LEFT, padx=8)

        # Legend indicator
        legend_frame = tk.Frame(top_bar, bg="#1c1c1c")
        legend_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Label(legend_frame, text="🟢 Active Mask", foreground="#00e676", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=4)
        ttk.Label(legend_frame, text="|", foreground="#555555").pack(side=tk.LEFT, padx=2)
        ttk.Label(legend_frame, text="🟠 Ghost (n-1)", foreground="#ff9800", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=4)
        ttk.Label(legend_frame, text="|", foreground="#555555").pack(side=tk.LEFT, padx=2)
        ttk.Label(legend_frame, text="🎞️ Macro Context", foreground="#00e5ff", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=4)
        ttk.Label(legend_frame, text="|", foreground="#555555").pack(side=tk.LEFT, padx=2)
        self.lbl_device = ttk.Label(legend_frame, text=f"GPU: {self.device.type.upper()}", font=("Consolas", 9), foreground="#888888")
        self.lbl_device.pack(side=tk.LEFT, padx=4)

        # 2. Dual-Viewport Center Area
        center_container = tk.Frame(self.root, bg="#0a0a0a")
        center_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left Viewport: Macro Temporal Filmstrip [Past t-stride | Current t | Future t+stride]
        left_frame = tk.Frame(center_container, bg="#141414", highlightbackground="#2a2a2a", highlightthickness=1)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        left_header = tk.Frame(left_frame, bg="#202020", padx=8, pady=5)
        left_header.pack(side=tk.TOP, fill=tk.X)
        self.lbl_left_title = ttk.Label(
            left_header, 
            text=f"🎞️ TEMPORAL FILMSTRIP [Past t-{self.carousel_stride} | Current t | Future t+{self.carousel_stride}]", 
            font=("Segoe UI", 9, "bold"),
            foreground="#00e5ff"
        )
        self.lbl_left_title.pack(side=tk.LEFT)

        # Stride Controls in Left Header
        stride_ctrl_frame = tk.Frame(left_header, bg="#202020")
        stride_ctrl_frame.pack(side=tk.RIGHT)

        ttk.Label(stride_ctrl_frame, text="Stride:", font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Button(stride_ctrl_frame, text="-", style="Mini.TButton", width=2, command=lambda: self.adjust_stride(-1), takefocus=False).pack(side=tk.LEFT, padx=1)
        self.lbl_stride = ttk.Label(stride_ctrl_frame, text=f"±{self.carousel_stride}f", font=("Consolas", 9, "bold"), foreground="#00e5ff")
        self.lbl_stride.pack(side=tk.LEFT, padx=2)
        ttk.Button(stride_ctrl_frame, text="+", style="Mini.TButton", width=2, command=lambda: self.adjust_stride(1), takefocus=False).pack(side=tk.LEFT, padx=1)

        ttk.Button(stride_ctrl_frame, text="x1", style="Mini.TButton", width=3, command=lambda: self.set_stride(1), takefocus=False).pack(side=tk.LEFT, padx=1)
        ttk.Button(stride_ctrl_frame, text="x5", style="Mini.TButton", width=3, command=lambda: self.set_stride(5), takefocus=False).pack(side=tk.LEFT, padx=1)
        ttk.Button(stride_ctrl_frame, text="x10", style="Mini.TButton", width=3, command=lambda: self.set_stride(10), takefocus=False).pack(side=tk.LEFT, padx=1)
        ttk.Button(stride_ctrl_frame, text="x20", style="Mini.TButton", width=3, command=lambda: self.set_stride(20), takefocus=False).pack(side=tk.LEFT, padx=1)

        self.btn_film_mask = ttk.Button(stride_ctrl_frame, text="Masks: OFF", style="Mini.TButton", width=9, command=self.toggle_filmstrip_masks, takefocus=False)
        self.btn_film_mask.pack(side=tk.LEFT, padx=(4, 0))

        self.canvas_carousel = tk.Canvas(left_frame, bg="#050505", highlightthickness=0)
        self.canvas_carousel.pack(fill=tk.BOTH, expand=True)

        # Right Viewport: Interactive Annotation Studio + Ghost Overlay
        right_frame = tk.Frame(center_container, bg="#141414", highlightbackground="#2a2a2a", highlightthickness=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))

        right_header = tk.Frame(right_frame, bg="#202020", padx=10, pady=5)
        right_header.pack(side=tk.TOP, fill=tk.X)
        self.lbl_right_title = ttk.Label(
            right_header, 
            text="✏️ INTERACTIVE STUDIO [Active Frame n Mask (Green) + Ghost n-1 (Amber)]", 
            font=("Segoe UI", 9, "bold"),
            foreground="#00e676"
        )
        self.lbl_right_title.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(right_frame, bg="#050505", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 3. Video Timeline Scrubber Bar
        timeline_frame = tk.Frame(self.root, bg="#181818", padx=15, pady=4)
        timeline_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(timeline_frame, text="⏩ Timeline:", font=("Segoe UI", 9, "bold"), foreground="#00e5ff").pack(side=tk.LEFT, padx=(0, 6))

        total_frames = max(1, len(self.image_paths))
        self.scale_timeline = tk.Scale(
            timeline_frame,
            from_=0,
            to=total_frames - 1,
            orient=tk.HORIZONTAL,
            bg="#181818",
            fg="#00e5ff",
            troughcolor="#282828",
            activebackground="#007acc",
            highlightthickness=0,
            showvalue=0,
            command=self.on_timeline_scrub
        )
        self.scale_timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.lbl_trial_info = ttk.Label(timeline_frame, text="Trial: Initializing...", font=("Consolas", 9), foreground="#aaaaaa")
        self.lbl_trial_info.pack(side=tk.RIGHT, padx=5)

        # 4. Bottom Control Panel
        bot_bar = tk.Frame(self.root, bg="#1c1c1c", padx=15, pady=8)
        bot_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Row 1: Actions & Navigation Buttons
        btn_row = tk.Frame(bot_bar, bg="#1c1c1c")
        btn_row.pack(fill=tk.X, pady=(0, 5))

        self.btn_prev = ttk.Button(btn_row, text="◀ Prev (Q)", width=10, command=self.prev_frame, takefocus=False)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.btn_next = ttk.Button(btn_row, text="Confirm & Next ▶ (W / SPACE)", style="Accent.TButton", width=25, command=self.next_frame, takefocus=False)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        self.btn_predict = ttk.Button(btn_row, text="🔮 Predict AI (U)", style="AIPredict.TButton", width=14, command=self.manual_predict, takefocus=False)
        self.btn_predict.pack(side=tk.LEFT, padx=3)

        self.btn_jump = ttk.Button(btn_row, text="⏭ Next Unlabeled (J)", width=17, command=self.jump_next_unannotated, takefocus=False)
        self.btn_jump.pack(side=tk.LEFT, padx=3)

        # Ghost / Onion Skin Controls
        self.btn_ghost = ttk.Button(btn_row, text="👻 Ghost: ON (G)", style="GhostOn.TButton", width=14, command=self.toggle_ghost, takefocus=False)
        self.btn_ghost.pack(side=tk.LEFT, padx=3)

        self.btn_copy = ttk.Button(btn_row, text="📋 Copy Ghost (K)", width=14, command=self.copy_prev_mask, takefocus=False)
        self.btn_copy.pack(side=tk.LEFT, padx=3)

        self.btn_retrain = ttk.Button(btn_row, text="⚡ Train U-Net (T)", width=15, command=self.trigger_train_thread, takefocus=False)
        self.btn_retrain.pack(side=tk.LEFT, padx=3)

        self.btn_clear = ttk.Button(btn_row, text="🧹 Clear (C)", width=10, command=self.clear_mask, takefocus=False)
        self.btn_clear.pack(side=tk.LEFT, padx=3)

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
            text="[Klik Kiri]: Gambar | [Klik Kanan]: Hapus | [W / SPACE]: Simpan & Lanjut | [Q]: Prev | [U]: AI Predict | [K]: Copy Ghost | [G]: Ghost | [- / +]: Stride Karousel",
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

        # Key bindings with 'break' to prevent event bubbling and double triggering
        self.root.bind("<space>", lambda e: self.handle_key_nav(1))
        self.root.bind("<Return>", lambda e: self.handle_key_nav(1))
        self.root.bind("<w>", lambda e: self.handle_key_nav(1))
        self.root.bind("<W>", lambda e: self.handle_key_nav(1))
        self.root.bind("<q>", lambda e: self.handle_key_nav(-1))
        self.root.bind("<Q>", lambda e: self.handle_key_nav(-1))
        self.root.bind("<p>", lambda e: self.handle_key_nav(-1))
        self.root.bind("<P>", lambda e: self.handle_key_nav(-1))
        self.root.bind("<u>", lambda e: self.handle_key_predict())
        self.root.bind("<U>", lambda e: self.handle_key_predict())
        self.root.bind("<j>", lambda e: self.handle_key_jump())
        self.root.bind("<J>", lambda e: self.handle_key_jump())
        self.root.bind("<g>", lambda e: self.handle_key_ghost())
        self.root.bind("<G>", lambda e: self.handle_key_ghost())
        self.root.bind("<k>", lambda e: self.handle_key_copy())
        self.root.bind("<K>", lambda e: self.handle_key_copy())
        self.root.bind("<t>", lambda e: self.handle_key_train())
        self.root.bind("<T>", lambda e: self.handle_key_train())
        self.root.bind("<c>", lambda e: self.handle_key_clear())
        self.root.bind("<C>", lambda e: self.handle_key_clear())
        self.root.bind("<bracketleft>", lambda e: self.handle_key_brush(-2))
        self.root.bind("<bracketright>", lambda e: self.handle_key_brush(2))

        # Stride adjustment shortcuts: [-] and [=]/[+]
        self.root.bind("<minus>", lambda e: self.adjust_stride(-1))
        self.root.bind("<equal>", lambda e: self.adjust_stride(1))
        self.root.bind("<plus>", lambda e: self.adjust_stride(1))

        # Dynamic Canvas Resize
        self.canvas.bind("<Configure>", lambda e: self.render_overlay())
        self.canvas_carousel.bind("<Configure>", lambda e: self.render_context_carousel())

    def adjust_stride(self, delta):
        self.set_stride(self.carousel_stride + delta)
        return "break"

    def set_stride(self, stride_val):
        self.carousel_stride = max(1, min(50, stride_val))
        self.lbl_stride.config(text=f"±{self.carousel_stride}f")
        self.lbl_left_title.config(
            text=f"🎞️ TEMPORAL FILMSTRIP [Past t-{self.carousel_stride} | Current t | Future t+{self.carousel_stride}]"
        )
        self.render_context_carousel()

    def toggle_filmstrip_masks(self):
        self.carousel_show_masks = not self.carousel_show_masks
        if self.carousel_show_masks:
            self.btn_film_mask.config(text="Masks: ON", style="Accent.TButton")
        else:
            self.btn_film_mask.config(text="Masks: OFF", style="Mini.TButton")
        self.render_context_carousel()

    def on_timeline_scrub(self, val):
        if self.is_scrubbing or not self.image_paths:
            return
        target_idx = int(float(val))
        if target_idx != self.current_idx and 0 <= target_idx < len(self.image_paths):
            self.load_frame(target_idx)

    def handle_key_nav(self, delta):
        now = time.time()
        if now - self.last_key_time < 0.20:
            return "break"
        self.last_key_time = now
        if delta > 0:
            self.next_frame()
        else:
            self.prev_frame()
        return "break"

    def handle_key_predict(self):
        now = time.time()
        if now - self.last_key_time < 0.20:
            return "break"
        self.last_key_time = now
        self.manual_predict()
        return "break"

    def handle_key_jump(self):
        now = time.time()
        if now - self.last_key_time < 0.20:
            return "break"
        self.last_key_time = now
        self.jump_next_unannotated()
        return "break"

    def handle_key_ghost(self):
        now = time.time()
        if now - self.last_key_time < 0.15:
            return "break"
        self.last_key_time = now
        self.toggle_ghost()
        return "break"

    def handle_key_copy(self):
        now = time.time()
        if now - self.last_key_time < 0.15:
            return "break"
        self.last_key_time = now
        self.copy_prev_mask()
        return "break"

    def handle_key_train(self):
        now = time.time()
        if now - self.last_key_time < 0.20:
            return "break"
        self.last_key_time = now
        self.trigger_train_thread()
        return "break"

    def handle_key_clear(self):
        now = time.time()
        if now - self.last_key_time < 0.15:
            return "break"
        self.last_key_time = now
        self.clear_mask()
        return "break"

    def handle_key_brush(self, delta):
        self.adjust_brush(delta)
        return "break"

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
            self.lbl_status.config(text="Status: Mask Disalin dari Ghost (Frame n-1)", foreground="#ffd54f")
            self.render_overlay()
        else:
            self.lbl_status.config(text="Status: Tidak ada Ghost mask pada frame sebelumnya", foreground="#ffb74d")

    def manual_predict(self):
        if self.current_img_gray is None:
            return
        if not self.is_trained:
            if os.path.exists(self.model_checkpoint):
                try:
                    ckpt = torch.load(self.model_checkpoint, map_location=self.device)
                    self.model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
                    self.is_trained = True
                except Exception:
                    pass
        if self.is_trained:
            self.current_mask = self.predict_unet(self.current_img_gray)
            self.lbl_status.config(text="Status: 🔮 AI Proposal Dihasilkan (U-Net)", foreground="#00e5ff")
            self.render_overlay()
        else:
            messagebox.showwarning("U-Net Belum Siap", "U-Net belum memiliki bobot terlatih.\nSilakan anotasi minimal 5-10 frame terlebih dahulu!")

    def jump_next_unannotated(self):
        if not self.image_paths:
            return
        for i in range(self.current_idx + 1, len(self.image_paths)):
            p = self.image_paths[i]
            rel = os.path.relpath(p, self.img_base)
            mf = os.path.join(self.mask_base, rel)
            if not os.path.exists(mf):
                self.load_frame(i)
                return
            m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            if m is None or np.count_nonzero(m) == 0:
                self.load_frame(i)
                return
        messagebox.showinfo("Selesai", "Semua frame telah memiliki anotasi valid!")

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
                pm = cv2.imread(prev_mask_path, cv2.IMREAD_GRAYSCALE)
                self.prev_mask = pm if (pm is not None and np.count_nonzero(pm) > 0) else None
            else:
                self.prev_mask = None
        else:
            self.prev_mask = None

        has_valid_saved_mask = False
        if os.path.exists(mask_path):
            saved_m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if saved_m is not None and np.count_nonzero(saved_m) > 0:
                self.current_mask = saved_m
                has_valid_saved_mask = True
                self.lbl_status.config(text="Status: Validated Ground Truth (Saved)", foreground="#00e676")

        if not has_valid_saved_mask:
            # Predict proposal with U-Net
            if self.is_trained and self.ai_proposal_enabled:
                self.current_mask = self.predict_unet(self.current_img_gray)
                self.lbl_status.config(text="Status: 🤖 AI Auto-Proposal (Check & Confirm)", foreground="#00e5ff")
            else:
                self.current_mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
                self.lbl_status.config(text="Status: Manual Cold-Start (Draw / [U] Predict AI)", foreground="#ffb74d")

        # Update Timeline Scrubber
        self.is_scrubbing = True
        self.scale_timeline.set(self.current_idx)
        self.is_scrubbing = False

        # Update labels
        trial_name = os.path.basename(os.path.dirname(img_path))
        subj_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        self.lbl_trial_info.config(text=f"{subj_name} | {trial_name}")
        self.lbl_progress.config(text=f"Frame {self.current_idx + 1}/{len(self.image_paths)}")

        self.render_context_carousel()
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

    def render_context_carousel(self):
        """
        Renders the 3-frame macro temporal filmstrip side-by-side:
        [Past t-stride | Current Raw t (Highlighted) | Future t+stride]
        Bounded within the same trial folder so it never crosses patient boundaries!
        """
        if not self.image_paths or self.current_img_gray is None:
            return

        c_w = max(50, self.canvas_carousel.winfo_width())
        c_h = max(50, self.canvas_carousel.winfo_height())
        if c_w < 100: c_w, c_h = 750, 650

        curr_path = self.image_paths[self.current_idx]
        curr_dir = os.path.dirname(curr_path)

        # 1. Determine Past Frame index (bounded by trial start)
        past_idx = self.current_idx - self.carousel_stride
        if past_idx < 0 or os.path.dirname(self.image_paths[past_idx]) != curr_dir:
            # Clamp to first frame of this trial
            for idx in range(self.current_idx, -1, -1):
                if os.path.dirname(self.image_paths[idx]) == curr_dir:
                    past_idx = idx
                else:
                    break

        if past_idx != self.current_idx:
            past_img = cv2.imread(self.image_paths[past_idx], cv2.IMREAD_GRAYSCALE)
            past_bgr = cv2.cvtColor(past_img, cv2.COLOR_GRAY2BGR) if past_img is not None else np.zeros((self.orig_h, self.orig_w, 3), dtype=np.uint8)
            # Optional mask overlay
            if self.carousel_show_masks:
                p_rel = os.path.relpath(self.image_paths[past_idx], self.img_base)
                p_mask_path = os.path.join(self.mask_base, p_rel)
                if os.path.exists(p_mask_path):
                    p_mask = cv2.imread(p_mask_path, cv2.IMREAD_GRAYSCALE)
                    if p_mask is not None and np.any(p_mask > 0):
                        ml = np.zeros_like(past_bgr)
                        ml[p_mask > 0] = [0, 200, 100]
                        past_bgr = cv2.addWeighted(past_bgr, 1.0, ml, 0.35, 0)
            # Badge
            cv2.rectangle(past_bgr, (0, 0), (self.orig_w, 24), (20, 20, 20), -1)
            cv2.putText(past_bgr, f"Past [t-{self.carousel_stride}] (F{past_idx+1})", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            past_bgr = np.zeros((self.orig_h, self.orig_w, 3), dtype=np.uint8)
            cv2.putText(past_bgr, "[TRIAL START]", (self.orig_w // 2 - 50, self.orig_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1, cv2.LINE_AA)

        # 2. Current Frame (Highlight Cyan Border & Badge)
        curr_bgr = cv2.cvtColor(self.current_img_gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(curr_bgr, (0, 0), (self.orig_w - 1, self.orig_h - 1), (255, 255, 0), 2)  # Cyan border
        cv2.rectangle(curr_bgr, (0, 0), (self.orig_w, 24), (0, 120, 150), -1)
        cv2.putText(curr_bgr, f"* Current [t] (F{self.current_idx + 1}) *", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Determine Future Frame index (bounded by trial end)
        future_idx = self.current_idx + self.carousel_stride
        if future_idx >= len(self.image_paths) or os.path.dirname(self.image_paths[future_idx]) != curr_dir:
            # Clamp to last frame of this trial
            for idx in range(self.current_idx, len(self.image_paths)):
                if os.path.dirname(self.image_paths[idx]) == curr_dir:
                    future_idx = idx
                else:
                    break

        if future_idx != self.current_idx:
            future_img = cv2.imread(self.image_paths[future_idx], cv2.IMREAD_GRAYSCALE)
            future_bgr = cv2.cvtColor(future_img, cv2.COLOR_GRAY2BGR) if future_img is not None else np.zeros((self.orig_h, self.orig_w, 3), dtype=np.uint8)
            # Optional mask overlay
            if self.carousel_show_masks:
                f_rel = os.path.relpath(self.image_paths[future_idx], self.img_base)
                f_mask_path = os.path.join(self.mask_base, f_rel)
                if os.path.exists(f_mask_path):
                    f_mask = cv2.imread(f_mask_path, cv2.IMREAD_GRAYSCALE)
                    if f_mask is not None and np.any(f_mask > 0):
                        ml = np.zeros_like(future_bgr)
                        ml[f_mask > 0] = [0, 200, 100]
                        future_bgr = cv2.addWeighted(future_bgr, 1.0, ml, 0.35, 0)
            # Badge
            cv2.rectangle(future_bgr, (0, 0), (self.orig_w, 24), (20, 20, 20), -1)
            cv2.putText(future_bgr, f"Future [t+{self.carousel_stride}] (F{future_idx+1})", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            future_bgr = np.zeros((self.orig_h, self.orig_w, 3), dtype=np.uint8)
            cv2.putText(future_bgr, "[TRIAL END]", (self.orig_w // 2 - 45, self.orig_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1, cv2.LINE_AA)

        # 4px Divider column
        divider = np.full((self.orig_h, 4, 3), 40, dtype=np.uint8)

        # Stitch horizontally
        filmstrip = np.hstack([past_bgr, divider, curr_bgr, divider, future_bgr])
        fh, fw = filmstrip.shape[:2]

        scale = min(c_w / fw, c_h / fh)
        disp_w = max(1, int(fw * scale))
        disp_h = max(1, int(fh * scale))

        disp_resized = cv2.resize(filmstrip, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(cv2.cvtColor(disp_resized, cv2.COLOR_BGR2RGB))
        self.tk_carousel_img = ImageTk.PhotoImage(img_pil)

        self.canvas_carousel.delete("all")
        off_x = (c_w - disp_w) // 2
        off_y = (c_h - disp_h) // 2
        self.canvas_carousel.create_image(off_x, off_y, anchor=tk.NW, image=self.tk_carousel_img)

    def render_overlay(self):
        """
        Renders the active interactive studio on the right canvas with:
        1. Ghost layer (Frame n-1) in translucent amber/orange.
        2. Active mask (Frame n) in translucent neon green + cyan outline.
        """
        if self.current_img_gray is None or self.current_mask is None:
            return

        c_w = max(50, self.canvas.winfo_width())
        c_h = max(50, self.canvas.winfo_height())
        if c_w < 100: c_w, c_h = 750, 650

        # Build RGB display
        bgr = cv2.cvtColor(self.current_img_gray, cv2.COLOR_GRAY2BGR)
        overlay = bgr.copy()

        # 1. GHOST LAYER (Frame n-1) in Warm Amber/Orange
        if self.ghost_enabled and self.prev_mask is not None and np.any(self.prev_mask > 0):
            ghost_layer = np.zeros_like(bgr)
            ghost_layer[self.prev_mask > 0] = [0, 140, 255]  # Amber/Orange in BGR
            overlay = cv2.addWeighted(overlay, 1.0, ghost_layer, 0.35, 0)

            # Draw Ghost Contour (Amber/Orange line)
            ghost_contours, _ = cv2.findContours(self.prev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, ghost_contours, -1, (0, 140, 255), 2)

        # 2. CURRENT ACTIVE MASK (Frame n) in Neon Green
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
        self.tk_canvas_img = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        off_x = (c_w - disp_w) // 2
        off_y = (c_h - disp_h) // 2
        self.canvas.create_image(off_x, off_y, anchor=tk.NW, image=self.tk_canvas_img)

        self.render_meta = (off_x, off_y, scale)

    def next_frame(self):
        if not self.image_paths or self.is_navigating:
            return

        self.is_navigating = True
        try:
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
        finally:
            self.is_navigating = False

    def prev_frame(self):
        if not self.image_paths or self.is_navigating:
            return

        self.is_navigating = True
        try:
            if self.current_idx > 0:
                self.load_frame(self.current_idx - 1)
        finally:
            self.is_navigating = False

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
                if img is not None and msk is not None and np.count_nonzero(msk) > 0:
                    train_samples.append((img, msk))

        if not train_samples:
            self.is_training_now = False
            self.lbl_status.config(text="Status: Tidak ada mask valid (>0 px) untuk training", foreground="#ffb74d")
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

        print(f"✓ [OK] U-Net Model Updated! ({len(train_samples)} masks, Loss: {total_loss/len(train_samples):.4f})\n")
        self.is_training_now = False
        self.lbl_status.config(text="Status: U-Net Model Updated! (Active)", foreground="#00e676")
        self.lbl_ai_indicator.config(text=f"🤖 U-Net: READY ({len(train_samples)} masks)", foreground="#00e676")


def main():
    root = tk.Tk()
    app = InteractiveAnnotatorTk(root)
    root.mainloop()


if __name__ == "__main__":
    main()
