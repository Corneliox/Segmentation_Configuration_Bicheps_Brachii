@echo off
title Export Biceps Brachii PyTorch Model to ONNX (Web Ready)
cd /d "%~dp0"
echo ====================================================================
echo Starting PyTorch to ONNX Export & INT8 Web Quantization...
echo Source Checkpoint: weights\best_model.pth
echo Target Output    : weights\biceps_brachii_unet.onnx
echo Metadata Output  : weights\model_metadata.json
echo ====================================================================
python src\export_biceps_to_onnx.py
echo.
pause
