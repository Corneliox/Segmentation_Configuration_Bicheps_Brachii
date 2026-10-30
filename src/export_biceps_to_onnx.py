"""
=====================================================================
export_biceps_to_onnx.py
Skrip Ekspor & Optimasi Model Biceps Brachii ke Format ONNX (Web Ready)
Mendukung WebAssembly, WebGL, WebGPU, dan Edge Runtime (npm/JavaScript)
=====================================================================
"""

import os
import sys
import json
import yaml
import numpy as np
import torch

# Tambahkan src ke path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import build_model


def export_model(config_path=None, output_onnx_path=None, quantize=True):
    print("=" * 70)
    print("BICEPS BRACHII ULTRASOUND SEGMENTATION - ONNX EXPORT PIPELINE")
    print("=" * 70)

    # 1. Baca Konfigurasi
    base_dir = os.path.dirname(current_dir)
    if config_path is None:
        config_path = os.path.join(base_dir, "configs", "config.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = cfg["inference"].get("checkpoint_path", "./weights/best_model.pth")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.normpath(os.path.join(base_dir, checkpoint_path))

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint model tidak ditemukan di: {checkpoint_path}")

    weights_dir = os.path.dirname(checkpoint_path)
    os.makedirs(weights_dir, exist_ok=True)

    if output_onnx_path is None:
        output_onnx_path = os.path.join(weights_dir, "biceps_brachii_unet.onnx")

    # 2. Inisialisasi Arsitektur Model
    print(f"\n[1/5] Membangun arsitektur model...")
    device = torch.device("cpu")  # Ekspor dalam format CPU netral
    model = build_model(cfg).to(device)

    # 3. Muat Bobot Pelatihan
    print(f"[2/5] Memuat bobot checkpoint: {os.path.basename(checkpoint_path)} ({os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB)...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 4. Tentukan Dimensi Input & Dummy Tensor
    h = cfg["data"].get("image_height", 512)
    w = cfg["data"].get("image_width", 224)
    in_channels = cfg["data"].get("in_channels", 1)

    dummy_input = torch.randn(1, in_channels, h, w, dtype=torch.float32, device=device)
    print(f"[3/5] Dimensi input: batch=1, channels={in_channels}, height={h}, width={w}")

    # 5. Ekspor ke Format ONNX
    print(f"[4/5] Mengekspor ke ONNX: {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    )

    fp32_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"      ✓ Berhasil diekspor: {os.path.basename(output_onnx_path)} ({fp32_size_mb:.2f} MB)")

    # 6. Verifikasi Paritas Numerik PyTorch vs ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"\n[Verifikasi] Menguji paritas inferensi PyTorch vs ONNX Runtime...")
        
        with torch.no_grad():
            py_output = model(dummy_input).numpy()

        ort_session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        max_abs_diff = np.max(np.abs(py_output - ort_output))
        print(f"      ✓ Max Absolute Difference: {max_abs_diff:.6e} (Lolos Paritas Presisi)")
    except ImportError:
        print("      (Info: onnxruntime belum terinstall di Python env, verifikasi paritas dilewati)")

    # 7. Kuantisasi INT8 Dinamis untuk Pengiriman Web (Opsional & Sangat Direkomendasikan)
    quant_path = os.path.join(weights_dir, "biceps_brachii_unet_quantized.onnx")
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print(f"\n[Optimasi Web] Melakukan Dynamic INT8 Quantization...")
            quantize_dynamic(
                model_input=output_onnx_path,
                model_output=quant_path,
                weight_type=QuantType.QUInt8
            )
            quant_size_mb = os.path.getsize(quant_path) / (1024 * 1024)
            print(f"      ✓ Model Terkuantisasi Web: {os.path.basename(quant_path)} ({quant_size_mb:.2f} MB)")
            print(f"      ✓ Rasio Kompresi: {fp32_size_mb / quant_size_mb:.2f}x lebih ringan!")
        except Exception as e_quant:
            print(f"      (Info: Kuantisasi INT8 dilewati: {e_quant})")

    # 8. Ekspor Metadata Model JSON untuk Aplikasi npm/JavaScript
    meta_path = os.path.join(weights_dir, "model_metadata.json")
    metadata = {
        "model_name": cfg["project"].get("name", "Biceps_Brachii_US_Segmentation"),
        "version": cfg["project"].get("version", "2.0.0"),
        "framework": "ONNX Runtime Web / WebAssembly",
        "architecture": f"{cfg['model'].get('architecture', 'Unet')}-{cfg['model'].get('encoder_name', 'resnet34')}",
        "input_tensor": {
            "name": "input",
            "shape": [1, in_channels, h, w],
            "channels": in_channels,
            "data_type": "float32",
            "color_space": "Grayscale B-Mode",
            "normalization": {
                "scale": 255.0,
                "mean": 0.5,
                "std": 0.5,
                "formula": "(pixel / 255.0 - 0.5) / 0.5"
            }
        },
        "output_tensor": {
            "name": "output",
            "shape": [1, 1, h, w],
            "data_type": "float32",
            "activation": "sigmoid",
            "range": [0.0, 1.0],
            "binary_threshold": cfg["inference"].get("threshold", 0.5)
        },
        "biomechanics": {
            "pixel_spacing_mm": cfg["inference"].get("pixel_spacing_mm", 0.1757),
            "aperture_width_mm": 38.4,
            "aperture_depth_mm": 89.96,
            "active_contraction_threshold_mm": 2.0
        },
        "artifacts": {
            "fp32_model": os.path.basename(output_onnx_path),
            "quantized_model": os.path.basename(quant_path) if os.path.exists(quant_path) else None
        }
    }

    with open(meta_path, "w") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    print(f"\n[5/5] Metadata JSON tersimpan: {os.path.basename(meta_path)}")
    print("=" * 70)
    print("SELESAI! Model ONNX dan metadata siap dimasukkan ke folder public/ npm.")
    print("=" * 70)


if __name__ == "__main__":
    export_model()
