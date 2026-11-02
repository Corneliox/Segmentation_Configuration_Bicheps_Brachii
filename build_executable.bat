@echo off
title Build Biceps Brachii Studio Executable
cd /d "%~dp0"
echo ====================================================================
echo Building Biceps Brachii Pipeline Studio Windows Executable (.exe)...
echo - PyInstaller Spec: biceps_studio.spec
echo - Output Directory: dist\Biceps_Brachii_Pipeline_Studio
echo ====================================================================

python -m pyinstaller biceps_studio.spec --noconfirm --clean
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] PyInstaller build failed! Please verify Python and PyInstaller packages.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Copying supporting assets into dist folder...
if exist presets xcopy /E /I /Y presets dist\Biceps_Brachii_Pipeline_Studio\presets
if exist configs xcopy /E /I /Y configs dist\Biceps_Brachii_Pipeline_Studio\configs
if exist matlab_converter xcopy /E /I /Y matlab_converter dist\Biceps_Brachii_Pipeline_Studio\matlab_converter
if exist weights xcopy /E /I /Y weights dist\Biceps_Brachii_Pipeline_Studio\weights

echo.
echo ====================================================================
echo Build Succeeded!
echo Standalone Application: dist\Biceps_Brachii_Pipeline_Studio\Biceps_Brachii_Pipeline_Studio.exe
echo ====================================================================
pause
