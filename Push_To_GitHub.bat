@echo off
title Push Biceps Brachii Web Studio & ONNX Model to GitHub
cd /d "%~dp0"
echo ====================================================================
echo Pushing Updates to GitHub:
echo Repository: https://github.com/Corneliox/Segmentation_Configuration_Bicheps_Brachii.git
echo Branch    : master
echo ====================================================================

echo [1/4] Ensuring Git LFS Tracking...
git lfs install
git lfs track "*.onnx"
git lfs track "*.pth"

echo [2/4] Staging Files...
git add .gitattributes
git add .gitignore
git add .github/
git add web/
git add src/export_biceps_to_onnx.py
git add Export_Model_To_ONNX.bat
git add weights/model_metadata.json
git add weights/biceps_brachii_unet.onnx
git add .

echo [3/4] Creating Commit...
git commit -m "feat: integrate ONNX export, React web studio with onnxruntime-web, and GitHub Actions CI/CD"

echo [4/4] Pushing to GitHub...
git push origin master

echo.
echo ====================================================================
echo Push selesai! Periksa tab 'Actions' di repositori GitHub Anda:
echo https://github.com/Corneliox/Segmentation_Configuration_Bicheps_Brachii/actions
echo ====================================================================
pause
