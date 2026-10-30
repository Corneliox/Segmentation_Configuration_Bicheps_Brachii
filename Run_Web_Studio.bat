@echo off
title Launch Biceps Brachii Ultrasound Web Studio
cd /d "%~dp0\web"
echo ====================================================================
echo Starting Biceps Brachii Ultrasound Studio (Vite + React + ONNX Web)
echo ====================================================================

if not exist node_modules (
    echo Installing npm dependencies...
    call npm install
)

echo.
echo Launching Vite Development Server...
start http://localhost:5173
call npm run dev
pause
