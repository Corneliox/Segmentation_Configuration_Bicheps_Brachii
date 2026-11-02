@echo off
title Biceps Brachii Ultrasound Segmentation ^& Biomechanics Studio
cd /d "%~dp0"
echo ====================================================================
echo Starting Biceps Brachii Pipeline Studio...
echo - Mode A: Raw .BIN Ingestion via MATLAB HD Preprocessor
echo - Mode B: Pre-extracted Image Folder Ingestion
echo - AI Engine: U-Net Biceps Brachii Segmentation ^& Aponeurosis Tracking
echo - Kinematics: Live Muscle Thickness Waveforms ^& Telemetry Cards
echo ====================================================================
python src\biceps_pipeline_studio.py
pause
