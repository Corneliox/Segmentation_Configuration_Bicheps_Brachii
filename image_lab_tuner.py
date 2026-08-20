"""
=====================================================================
ULTRASOUND IMAGE LAB & PARAMETER TUNING STUDIO
=====================================================================
Interactive GUI to tune B-mode processing parameters in real-time:
- Dynamic Range (DR in dB)
- Noise Rejection (Black level clip)
- Gamma / Contrast S-Curve
- Unsharp Mask Sharpening (Texture enhancement)
- Gaussian Blur Sigma
- CLAHE (Adaptive Histogram Equalization)
- Live side-by-side comparison & 1-Click Export to MATLAB / Config
=====================================================================
"""

import os
import sys
import struct
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np


class UltrasoundImageLab:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultrasound Image Lab | B-Mode Parameter Tuning Studio")
        self.root.geometry("1350x880")
        self.root.configure(bg="#121212")

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.parent_dir = os.path.dirname(self.base_dir)

        # Default sample file
        self.current_bin_file = os.path.join(self.parent_dir, "britty", "10rm", "data_open2", "i101_1112.bin")
        if not os.path.exists(self.current_bin_file):
            # Fallback to any bin
            bins = [os.path.join(r, f) for r, d, fs in os.walk(self.parent_dir) for f in fs if f.endswith('.bin')]
            if bins: self.current_bin_file = bins[0]

        self.frames = []
        self.num_samples = 2959
        self.num_lines = 33
        self.current_frame_idx = 0

        self.setup_styles()
        self.create_ui()
        self.load_bin_file(self.current_bin_file)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#181818", foreground="#e0e0e0")
        style.configure("TLabel", background="#181818", foreground="#e0e0e0", font=("Segoe UI", 9))
        style.configure("Header.TLabel", background="#181818", foreground="#00e5ff", font=("Segoe UI", 11, "bold"))
        style.configure("Section.TLabel", background="#222222", foreground="#ffb74d", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", background="#2c2c2c", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.map("TButton", background=[("active", "#007acc")])
        style.configure("TScale", background="#181818", troughcolor="#333333")

    def create_ui(self):
        # Master container: Left = Control Panels, Right = Live Canvas Display
        main_pane = tk.Frame(self.root, bg="#121212")
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # -------------------------------------------------------------
        # LEFT: TUNING CONTROLS PANEL
        # -------------------------------------------------------------
        control_panel = tk.Frame(main_pane, bg="#1c1c1c", width=420, padx=15, pady=12, relief=tk.RIDGE, bd=1)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.pack_propagate(False)

        # Title & File Picker
        ttk.Label(control_panel, text="🔬 ULTRASOUND IMAGE LAB", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))
        self.lbl_file = ttk.Label(control_panel, text=f"File: {os.path.basename(self.current_bin_file)}", font=("Consolas", 8), foreground="#888888")
        self.lbl_file.pack(anchor=tk.W, pady=(0, 10))

        btn_load = ttk.Button(control_panel, text="📂 Load Different .BIN File...", command=self.browse_bin)
        btn_load.pack(fill=tk.X, pady=(0, 15))

        # Sliders Container (Scrollable or compact)
        self.sliders = {}

        # 1. Frame Scrubbing
        self.create_slider(control_panel, "Frame Index", "frame_idx", 0, 100, 0, step=1, is_int=True)
        
        # Section: Dynamic Range & Contrast
        ttk.Label(control_panel, text="── [ 1. Dynamic Range & Levels ] ──", style="Section.TLabel").pack(fill=tk.X, pady=(10, 5))
        self.create_slider(control_panel, "Dynamic Range (DR in dB)", "dr_dB", 30.0, 95.0, 78.0, step=1.0)
        self.create_slider(control_panel, "Gain / Multiplier", "gain", 0.5, 3.0, 1.2, step=0.1)
        self.create_slider(control_panel, "Noise Rejection Clip (%)", "rejection", 0.0, 30.0, 8.0, step=1.0)
        self.create_slider(control_panel, "Gamma / Contrast Curve", "gamma", 0.4, 2.5, 1.1, step=0.05)

        # Section: Detail Enhancement & Filtering
        ttk.Label(control_panel, text="── [ 2. Texture & Edge Enhancement ] ──", style="Section.TLabel").pack(fill=tk.X, pady=(10, 5))
        self.create_slider(control_panel, "Sharpening (Unsharp Mask)", "sharpen", 0.0, 4.0, 1.5, step=0.1)
        self.create_slider(control_panel, "CLAHE Contrast Limit", "clahe", 0.0, 5.0, 2.0, step=0.2)
        self.create_slider(control_panel, "Gaussian Smoothing Sigma", "gauss_sigma", 0.0, 2.0, 0.0, step=0.1)

        # Section: Lateral Aspect Ratio
        ttk.Label(control_panel, text="── [ 3. Spatial Geometry ] ──", style="Section.TLabel").pack(fill=tk.X, pady=(10, 5))
        self.create_slider(control_panel, "Output Width Scaling", "width_scale", 1.0, 3.0, 1.5, step=0.1)

        # Action Buttons
        btn_box = tk.Frame(control_panel, bg="#1c1c1c")
        btn_box.pack(fill=tk.X, pady=(20, 0))

        btn_save = ttk.Button(btn_box, text="💾 Save to MATLAB & Config", command=self.save_parameters)
        btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        btn_reset = ttk.Button(btn_box, text="🔄 Reset", width=10, command=self.reset_defaults)
        btn_reset.pack(side=tk.RIGHT)

        # -------------------------------------------------------------
        # RIGHT: LIVE DUAL COMPARISON CANVAS
        # -------------------------------------------------------------
        display_frame = tk.Frame(main_pane, bg="#000000")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        header_disp = tk.Frame(display_frame, bg="#181818", height=35, padx=10, pady=5)
        header_disp.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header_disp, text="🔴 SEBELUM (Blur / Low Contrast)", font=("Segoe UI", 10, "bold"), foreground="#ff5252").pack(side=tk.LEFT, padx=30)
        ttk.Label(header_disp, text="🟢 SESUDAH (Live Tuned - Tajam & Definisi Tinggi)", font=("Segoe UI", 10, "bold"), foreground="#00e676").pack(side=tk.RIGHT, padx=30)

        self.canvas = tk.Canvas(display_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Configure>", lambda e: self.update_display())

        # HUD Info bar
        self.hud = tk.Label(display_frame, bg="#181818", fg="#00e5ff", font=("Consolas", 10, "bold"), text="Ready")
        self.hud.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))

    def create_slider(self, parent, label_text, key, min_val, max_val, default_val, step=0.1, is_int=False):
        frame = tk.Frame(parent, bg="#1c1c1c")
        frame.pack(fill=tk.X, pady=3)

        lbl_row = tk.Frame(frame, bg="#1c1c1c")
        lbl_row.pack(fill=tk.X)
        ttk.Label(lbl_row, text=label_text, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        val_lbl = ttk.Label(lbl_row, text=f"{default_val:.2f}" if not is_int else f"{int(default_val)}", font=("Consolas", 8, "bold"), foreground="#00e5ff")
        val_lbl.pack(side=tk.RIGHT)

        var = tk.DoubleVar(value=default_val)
        scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL,
                          command=lambda v: self.on_slider_change(key, var, val_lbl, is_int))
        scale.pack(fill=tk.X, pady=(1, 0))

        self.sliders[key] = {"var": var, "lbl": val_lbl, "default": default_val, "is_int": is_int}

    def on_slider_change(self, key, var, val_lbl, is_int):
        val = var.get()
        if is_int:
            val_lbl.config(text=f"{int(round(val))}")
        else:
            val_lbl.config(text=f"{val:.2f}")
        self.update_display()

    def browse_bin(self):
        f = filedialog.askopenfilename(initialdir=self.parent_dir, title="Select ArtUs .BIN File", filetypes=[("BIN files", "*.bin")])
        if f:
            self.load_bin_file(f)

    def load_bin_file(self, bin_path):
        if not os.path.exists(bin_path):
            return

        self.current_bin_file = bin_path
        self.lbl_file.config(text=f"File: {os.path.basename(bin_path)}")

        # Exact binary reader
        with open(bin_path, 'rb') as f:
            hdr_id = f.read(6).decode('ascii', errors='ignore')
            num_header_els = 11 if hdr_id.startswith(('RF0001', 'RF0002', 'RF0003', 'RF0004')) else 7
            f.seek(0, 2)
            total_size = f.tell()
            f.seek(6)

            self.frames = []
            while f.tell() < total_size:
                hdr_bytes = f.read(num_header_els * 4)
                if len(hdr_bytes) < num_header_els * 4: break
                hdr = struct.unpack(f'<{num_header_els}i', hdr_bytes)
                self.num_samples = hdr[6]
                self.num_lines = hdr[7]
                source_id = hdr[3]

                beam_bytes = f.read(3 * self.num_lines * 4)
                if len(beam_bytes) < 3 * self.num_lines * 4: break

                if hdr_id.startswith(('RF0003', 'RF0004')):
                    ts_bytes = f.read(self.num_lines * 4)
                    if len(ts_bytes) < self.num_lines * 4: break

                size_of_frame = self.num_samples * self.num_lines * 2 if source_id == 4 else self.num_samples * self.num_lines
                raw_bytes = f.read(size_of_frame * 2)
                if len(raw_bytes) < size_of_frame * 2: break

                arr = np.frombuffer(raw_bytes, dtype=np.int16)
                I_comp = arr[:self.num_samples * self.num_lines] if source_id == 4 else arr
                mat = np.reshape(I_comp, (self.num_samples, self.num_lines), order='F')
                self.frames.append(mat)

        if self.frames:
            self.sliders["frame_idx"]["var"].set(0)
            self.sliders["frame_idx"]["lbl"].config(text="0")
            # Update scale max
            # Update frame index slider maximum
            self.sliders["frame_idx"]["var"].trace_add("write", lambda *a: None)
            self.update_display()

    def process_tuned_image(self, rf_mat, params):
        # 1. Envelope Detection
        env = np.abs(rf_mat).astype(np.float32)
        env_max = np.max(env) + 1e-9
        norm = (env / env_max) * params["gain"]

        # 2. Dynamic Range (Log Compression)
        dr_dB = params["dr_dB"]
        b_db = (20.0 * np.log10(norm + 1e-9) + dr_dB) / dr_dB
        b_db = np.clip(b_db, 0.0, 1.0)

        # 3. Noise Rejection (Cut low noise to true black)
        rej_norm = params["rejection"] / 100.0
        if rej_norm > 0:
            b_db = np.where(b_db < rej_norm, 0.0, (b_db - rej_norm) / (1.0 - rej_norm + 1e-9))
            b_db = np.clip(b_db, 0.0, 1.0)

        # 4. Gamma S-Curve Contrast
        gamma = params["gamma"]
        b_db = b_db ** gamma

        # 5. Gaussian Filter (if > 0)
        if params["gauss_sigma"] > 0.05:
            b_db = cv2.GaussianBlur(b_db, (0, 0), params["gauss_sigma"])

        # Convert to 8-bit
        img_8u = (np.clip(b_db, 0, 1) * 255.0).astype(np.uint8)

        # 6. Spatial Resampling (Isotropic Aspect Ratio)
        target_h = 512
        base_w = int(round(512 * (9.9 / 57.0)))  # ~89 px
        target_w = int(round(base_w * params["width_scale"]))
        resized = cv2.resize(img_8u, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # 7. Unsharp Masking (Sharpening Edge & Fascicle Texture)
        sharpen_amt = params["sharpen"]
        if sharpen_amt > 0:
            blurred = cv2.GaussianBlur(resized, (0, 0), 1.5)
            sharpened = cv2.addWeighted(resized, 1.0 + sharpen_amt, blurred, -sharpen_amt, 0)
            resized = np.clip(sharpened, 0, 255).astype(np.uint8)

        # 8. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_lim = params["clahe"]
        if clahe_lim > 0:
            clahe_obj = cv2.createCLAHE(clipLimit=clahe_lim, tileGridSize=(8, 8))
            resized = clahe_obj.apply(resized)

        return resized

    def process_untuned_reference(self, rf_mat):
        # The previous blurry baseline (imgaussfilt 0.8, no rejection, linear)
        env = np.abs(rf_mat).astype(np.float32)
        norm = env / (np.max(env) + 1e-9)
        b_db = (20.0 * np.log10(norm + 1e-9) + 78.0) / 78.0
        b_db = np.clip(b_db, 0.0, 1.0)
        b_db = cv2.GaussianBlur(b_db, (0, 0), 0.8)
        img_8u = (b_db * 255.0).astype(np.uint8)
        return cv2.resize(img_8u, (89, 512), interpolation=cv2.INTER_CUBIC)

    def update_display(self):
        if not self.frames:
            return

        frame_idx = int(round(self.sliders["frame_idx"]["var"].get()))
        frame_idx = max(0, min(len(self.frames) - 1, frame_idx))
        rf_mat = self.frames[frame_idx]

        params = {k: v["var"].get() for k, v in self.sliders.items()}

        img_before = self.process_untuned_reference(rf_mat)
        img_after = self.process_tuned_image(rf_mat, params)

        # Build Side-by-Side Dual View
        h, w1 = img_before.shape
        _, w2 = img_after.shape
        gap = 30
        dual_img = np.zeros((h, w1 + w2 + gap), dtype=np.uint8)
        dual_img[:, :w1] = img_before
        dual_img[:, w1 + gap:] = img_after

        # Draw red line border for before, green for after
        dual_bgr = cv2.cvtColor(dual_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dual_bgr, (0, 0), (w1, h - 1), (50, 50, 255), 2)
        cv2.rectangle(dual_bgr, (w1 + gap, 0), (w1 + gap + w2, h - 1), (50, 255, 50), 2)

        # Canvas fit
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        if c_w < 50: c_w, c_h = 750, 600

        scale = min(c_w / dual_bgr.shape[1], c_h / dual_bgr.shape[0])
        disp_w = max(1, int(dual_bgr.shape[1] * scale))
        disp_h = max(1, int(dual_bgr.shape[0] * scale))

        disp_resized = cv2.resize(dual_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(cv2.cvtColor(disp_resized, cv2.COLOR_BGR2RGB))
        self.tk_img = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        off_x = (c_w - disp_w) // 2
        off_y = (c_h - disp_h) // 2
        self.canvas.create_image(off_x, off_y, anchor=tk.NW, image=self.tk_img)

        hud_text = (
            f"Frame {frame_idx + 1:04d}/{len(self.frames):04d} | "
            f"DR: {params['dr_dB']:.1f} dB | Gain: {params['gain']:.1f}x | "
            f"Rej: {params['rejection']:.0f}% | Gamma: {params['gamma']:.2f} | "
            f"Sharpen: {params['sharpen']:.1f} | CLAHE: {params['clahe']:.1f}"
        )
        self.hud.config(text=hud_text)

    def reset_defaults(self):
        for k, v in self.sliders.items():
            if k != "frame_idx":
                v["var"].set(v["default"])
                if v["is_int"]:
                    v["lbl"].config(text=f"{int(v['default'])}")
                else:
                    v["lbl"].config(text=f"{v['default']:.2f}")
        self.update_display()

    def save_parameters(self):
        params = {k: v["var"].get() for k, v in self.sliders.items() if k != "frame_idx"}
        
        # Save to JSON / YAML
        cfg_path = os.path.join(self.base_dir, "configs", "tuned_image_params.json")
        with open(cfg_path, "w") as f:
            json.dump(params, f, indent=4)

        # Update MATLAB convert_bin2dataset.m parameters
        matlab_file = os.path.join(self.base_dir, "matlab_converter", "convert_bin2dataset.m")
        if os.path.exists(matlab_file):
            with open(matlab_file, "r") as f:
                content = f.read()

            # Replace parameters
            content = self.replace_matlab_var(content, "dr_dB", f"{params['dr_dB']:.1f}")
            content = self.replace_matlab_var(content, "gain_val", f"{params['gain']:.2f}")
            content = self.replace_matlab_var(content, "rejection_pct", f"{params['rejection']:.1f}")
            content = self.replace_matlab_var(content, "gamma_val", f"{params['gamma']:.2f}")
            content = self.replace_matlab_var(content, "sharpen_val", f"{params['sharpen']:.2f}")
            content = self.replace_matlab_var(content, "clahe_limit", f"{params['clahe']:.2f}")
            content = self.replace_matlab_var(content, "gauss_sigma", f"{params['gauss_sigma']:.2f}")

            with open(matlab_file, "w") as f:
                f.write(content)

        messagebox.showinfo(
            "Tuning Saved!",
            f"Parameter berhasil disimpan!\n\n"
            f"• Dynamic Range : {params['dr_dB']:.1f} dB\n"
            f"• Noise Rejection: {params['rejection']:.0f}%\n"
            f"• Gamma Curve   : {params['gamma']:.2f}\n"
            f"• Sharpening    : {params['sharpen']:.2f}\n"
            f"• CLAHE Contrast: {params['clahe']:.2f}\n\n"
            f"File MATLAB 'convert_bin2dataset.m' telah otomatis diperbarui dengan parameter ini."
        )

    def replace_matlab_var(self, text, var_name, new_val):
        import re
        pattern = rf"({var_name}\s*=\s*)[^;]+(;\s*%.*)"
        if re.search(pattern, text):
            return re.sub(pattern, rf"\g<1>{new_val}\g<2>", text)
        return text


def main():
    root = tk.Tk()
    app = UltrasoundImageLab(root)
    root.mainloop()


if __name__ == "__main__":
    main()
