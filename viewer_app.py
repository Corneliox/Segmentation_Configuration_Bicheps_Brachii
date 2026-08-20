"""
=====================================================================
High-Speed Ultrasound Cine Viewer (Fast Cine-Loop Player)
=====================================================================
Designed specifically for smooth, zero-buffering playback of ultrasound 
frame sequences when holding Left/Right arrow keys at up to 120 FPS.
=====================================================================
"""

import os
import glob
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2


class UltrasoundCineViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultrasound High-Speed Cine Viewer | Biceps Brachii")
        self.root.geometry("1100x850")
        self.root.configure(bg="#181818")

        # Core State
        self.image_paths = []
        self.image_cache = {}  # In-memory fast cache
        self.current_idx = 0
        self.is_playing = False
        self.target_fps = 60
        self.last_frame_time = 0
        self.key_held = {"left": False, "right": False}
        self.key_repeat_job = None
        self.play_job = None

        # Base directories
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.parent_dir = os.path.dirname(self.base_dir)
        self.default_dataset_dir = os.path.join(self.parent_dir, "dataset", "images")

        self.setup_styles()
        self.create_ui()
        self.bind_events()

        # Auto-discover trial folders
        self.discover_dataset()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#181818", foreground="#e0e0e0")
        style.configure("TLabel", background="#181818", foreground="#e0e0e0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#181818", foreground="#00e5ff", font=("Segoe UI", 12, "bold"))
        style.configure("TButton", background="#2a2a2a", foreground="#ffffff", borderwidth=1, font=("Segoe UI", 9, "bold"))
        style.map("TButton", background=[("active", "#007acc"), ("pressed", "#005999")])
        style.configure("TCombobox", fieldbackground="#2a2a2a", background="#333333", foreground="#ffffff")
        style.configure("Horizontal.TScale", background="#181818", troughcolor="#333333")

    def create_ui(self):
        # 1. Top Control Bar (Folder & Trial Selectors)
        top_bar = tk.Frame(self.root, bg="#202020", height=60, padx=15, pady=10)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_bar, text="Condition & Trial:", style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 10))

        self.trial_var = tk.StringVar()
        self.trial_dropdown = ttk.Combobox(top_bar, textvariable=self.trial_var, state="readonly", width=40)
        self.trial_dropdown.pack(side=tk.LEFT, padx=5)
        self.trial_dropdown.bind("<<ComboboxSelected>>", self.on_trial_selected)

        btn_browse = ttk.Button(top_bar, text="📁 Open Folder...", command=self.open_custom_folder)
        btn_browse.pack(side=tk.LEFT, padx=10)

        # FPS selector
        ttk.Label(top_bar, text="Speed:").pack(side=tk.LEFT, padx=(20, 5))
        self.fps_var = tk.StringVar(value="60 FPS")
        fps_options = ["15 FPS", "30 FPS", "60 FPS", "120 FPS (Realtime)", "Max Speed"]
        self.fps_dropdown = ttk.Combobox(top_bar, textvariable=self.fps_var, values=fps_options, state="readonly", width=16)
        self.fps_dropdown.pack(side=tk.LEFT, padx=5)
        self.fps_dropdown.bind("<<ComboboxSelected>>", self.on_fps_changed)

        # 2. Main Image Display Canvas
        self.canvas_frame = tk.Frame(self.root, bg="#000000")
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 3. Bottom Timeline & Scrubbing Bar
        bottom_bar = tk.Frame(self.root, bg="#202020", padx=15, pady=12)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Slider and Controls row
        ctrl_row = tk.Frame(bottom_bar, bg="#202020")
        ctrl_row.pack(fill=tk.X, pady=(0, 5))

        self.btn_play = ttk.Button(ctrl_row, text="▶ Play (Space)", width=14, command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_prev = ttk.Button(ctrl_row, text="◀ Prev (←)", width=10, command=self.prev_frame)
        self.btn_prev.pack(side=tk.LEFT, padx=3)

        self.btn_next = ttk.Button(ctrl_row, text="Next (→) ▶", width=10, command=self.next_frame)
        self.btn_next.pack(side=tk.LEFT, padx=3)

        self.timeline_var = tk.DoubleVar()
        self.slider = ttk.Scale(ctrl_row, from_=0, to=100, variable=self.timeline_var, orient=tk.HORIZONTAL, style="Horizontal.TScale", command=self.on_slider_moved)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=15)

        # HUD Info labels
        self.lbl_hud = ttk.Label(bottom_bar, text="Ready | Hold ← or → Arrow Keys to Scrub | Space to Play", font=("Consolas", 10, "bold"), foreground="#00e5ff")
        self.lbl_hud.pack(side=tk.LEFT, pady=(5, 0))

        self.lbl_res = ttk.Label(bottom_bar, text="Resolution: - | Scale: 0.1113 mm/px", font=("Segoe UI", 9), foreground="#888888")
        self.lbl_res.pack(side=tk.RIGHT, pady=(5, 0))

    def bind_events(self):
        # Keyboard bindings for ultra-smooth scrubbing when holding arrow keys
        self.root.bind("<KeyPress-Left>", self.on_left_press)
        self.root.bind("<KeyRelease-Left>", self.on_left_release)
        self.root.bind("<KeyPress-Right>", self.on_right_press)
        self.root.bind("<KeyRelease-Right>", self.on_right_release)
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Home>", lambda e: self.goto_frame(0))
        self.root.bind("<End>", lambda e: self.goto_frame(len(self.image_paths) - 1))
        self.root.bind("<Up>", lambda e: self.adjust_fps(1))
        self.root.bind("<Down>", lambda e: self.adjust_fps(-1))
        self.canvas.bind("<Configure>", lambda e: self.render_current_frame())

    def discover_dataset(self):
        trial_folders = []
        if os.path.exists(self.default_dataset_dir):
            for root, dirs, files in os.walk(self.default_dataset_dir):
                pngs = [f for f in files if f.lower().endswith(".png")]
                if len(pngs) > 0:
                    rel = os.path.relpath(root, self.default_dataset_dir)
                    trial_folders.append((rel, root))

        if trial_folders:
            self.trial_map = dict(trial_folders)
            dropdown_values = sorted(list(self.trial_map.keys()))
            self.trial_dropdown["values"] = dropdown_values
            self.trial_dropdown.current(0)
            self.load_folder(self.trial_map[dropdown_values[0]])
        else:
            self.lbl_hud.config(text="No extracted frames found in dataset/images. Click 'Open Folder' to select images.")

    def on_trial_selected(self, event=None):
        selected = self.trial_var.get()
        if hasattr(self, "trial_map") and selected in self.trial_map:
            self.load_folder(self.trial_map[selected])

    def open_custom_folder(self):
        folder = filedialog.askdirectory(initialdir=self.parent_dir, title="Select Folder Containing PNG/JPG Images")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder_path):
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif")
        files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
        
        if not files:
            messagebox.showwarning("Empty Folder", f"No image files found in:\n{folder_path}")
            return

        self.image_paths = files
        self.image_cache.clear()
        self.current_idx = 0
        self.slider.config(to=len(self.image_paths) - 1)

        # Pre-cache first 50 frames in background memory
        for idx in range(min(50, len(files))):
            self.get_image(idx)

        self.render_current_frame()

    def get_image(self, idx):
        if idx not in self.image_cache:
            img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.image_cache[idx] = img
        return self.image_cache.get(idx)

    def render_current_frame(self):
        if not self.image_paths or self.current_idx >= len(self.image_paths):
            return

        img_np = self.get_image(self.current_idx)
        if img_np is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            canvas_w, canvas_h = 700, 600

        img_h, img_w = img_np.shape

        # Scale to fit canvas while preserving aspect ratio
        scale = min(canvas_w / img_w, canvas_h / img_h)
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))

        resized = cv2.resize(img_np, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(resized)
        self.tk_img = ImageTk.PhotoImage(img_pil)

        # Draw centered
        self.canvas.delete("all")
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.tk_img)

        # Update HUD
        self.timeline_var.set(self.current_idx)
        fname = os.path.basename(self.image_paths[self.current_idx])
        fps_val = 120.0
        time_sec = self.current_idx / fps_val

        hud_text = f"Frame: {self.current_idx + 1:04d}/{len(self.image_paths):04d} | Time: {time_sec:.3f}s | File: {fname}"
        self.lbl_hud.config(text=hud_text)
        self.lbl_res.config(text=f"Native: {img_w}x{img_h} px | Display: {disp_w}x{disp_h} px | 1 px = 0.1113 mm")

    def next_frame(self):
        if self.image_paths:
            self.current_idx = (self.current_idx + 1) % len(self.image_paths)
            self.render_current_frame()

    def prev_frame(self):
        if self.image_paths:
            self.current_idx = (self.current_idx - 1 + len(self.image_paths)) % len(self.image_paths)
            self.render_current_frame()

    def goto_frame(self, idx):
        if self.image_paths:
            self.current_idx = max(0, min(len(self.image_paths) - 1, int(idx)))
            self.render_current_frame()

    def on_slider_moved(self, val):
        self.goto_frame(int(float(val)))

    # Continuous Arrow Key Holding Handler
    def on_left_press(self, event):
        if not self.key_held["left"]:
            self.key_held["left"] = True
            self.prev_frame()
            self.schedule_key_scrub()

    def on_left_release(self, event):
        self.key_held["left"] = False

    def on_right_press(self, event):
        if not self.key_held["right"]:
            self.key_held["right"] = True
            self.next_frame()
            self.schedule_key_scrub()

    def on_right_release(self, event):
        self.key_held["right"] = False

    def schedule_key_scrub(self):
        if self.key_repeat_job:
            self.root.after_cancel(self.key_repeat_job)
        self.run_key_scrub()

    def run_key_scrub(self):
        if self.key_held["right"]:
            self.next_frame()
        elif self.key_held["left"]:
            self.prev_frame()

        if self.key_held["left"] or self.key_held["right"]:
            # Interval based on target FPS (e.g. 1000/60 = 16ms)
            interval_ms = max(5, int(1000 / self.target_fps))
            self.key_repeat_job = self.root.after(interval_ms, self.run_key_scrub)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ Pause (Space)")
            self.play_loop()
        else:
            self.btn_play.config(text="▶ Play (Space)")
            if self.play_job:
                self.root.after_cancel(self.play_job)

    def play_loop(self):
        if self.is_playing and self.image_paths:
            self.next_frame()
            interval_ms = max(5, int(1000 / self.target_fps))
            self.play_job = self.root.after(interval_ms, self.play_loop)

    def on_fps_changed(self, event=None):
        val = self.fps_var.get()
        if "15" in val: self.target_fps = 15
        elif "30" in val: self.target_fps = 30
        elif "60" in val: self.target_fps = 60
        elif "120" in val: self.target_fps = 120
        elif "Max" in val: self.target_fps = 200

    def adjust_fps(self, step):
        speeds = [15, 30, 60, 120, 200]
        curr = self.target_fps
        idx = min(range(len(speeds)), key=lambda i: abs(speeds[i] - curr))
        new_idx = max(0, min(len(speeds) - 1, idx + step))
        self.target_fps = speeds[new_idx]
        self.fps_var.set(f"{self.target_fps} FPS")


def main():
    root = tk.Tk()
    app = UltrasoundCineViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
