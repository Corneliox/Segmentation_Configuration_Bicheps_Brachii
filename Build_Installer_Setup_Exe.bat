@echo off
setlocal enabledelayedexpansion
title Biceps Brachii Pipeline Studio - Setup.exe Installer Builder
color 0B

echo ====================================================================
echo   BICEPS BRACHII PIPELINE STUDIO - SETUP.EXE INSTALLER BUILDER
echo   (Single-File Windows Installer Wizard Generator via Inno Setup)
echo ====================================================================
echo.

set "SCRIPT_DIR=%~dp0"
set "SEG_DIR=%SCRIPT_DIR%Segmentation_Configuration"
set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\iscc.exe"

cd /d "%SEG_DIR%"

:: 1. Check Inno Setup Compiler
if not exist "%ISCC_PATH%" (
    where iscc >nul 2>&1
    if %errorlevel% equ 0 (
        set "ISCC_PATH=iscc"
    ) else (
        color 0C
        echo [ERROR] Inno Setup 6 compiler (iscc.exe) was not found!
        echo Please install Inno Setup 6 from: https://jrsoftware.org/isdl.php
        echo Or run: winget install JRSoftware.InnoSetup
        echo.
        pause
        exit /b 1
    )
)

echo [INFO] Detected Inno Setup Compiler: %ISCC_PATH%
echo.

:: 2. Check if PyInstaller dist folder exists
if not exist "dist\Biceps_Brachii_Pipeline_Studio\Biceps_Brachii_Pipeline_Studio.exe" (
    echo [INFO] PyInstaller executable binaries not found. Building with PyInstaller first...
    call "%SEG_DIR%\build_executable.bat"
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] PyInstaller compilation failed!
        pause
        exit /b 1
    )
)

:: 3. Ensure all supporting folders are present in dist
echo [INFO] Synchronizing runtime assets...
if exist "presets" xcopy /E /I /Y "presets" "dist\Biceps_Brachii_Pipeline_Studio\presets" >nul
if exist "configs" xcopy /E /I /Y "configs" "dist\Biceps_Brachii_Pipeline_Studio\configs" >nul
if exist "weights" xcopy /E /I /Y "weights" "dist\Biceps_Brachii_Pipeline_Studio\weights" >nul
if exist "matlab_converter" xcopy /E /I /Y "matlab_converter" "dist\Biceps_Brachii_Pipeline_Studio\matlab_converter" >nul

:: 4. Run Inno Setup Compiler
echo.
echo [INFO] Compiling single-file Windows Installer Wizard (.exe)...
if not exist "dist_installer" mkdir "dist_installer"

"%ISCC_PATH%" /DAppVersion=2.1.0 "installer.iss"
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Inno Setup compilation failed!
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo   SUCCESS! SINGLE-FILE SETUP INSTALLER CREATED SUCCESSFULLY!
echo ====================================================================
echo   Installer Location:
for %%F in ("dist_installer\*.exe") do (
    echo   %%~fF (%%~zF bytes)
)
echo ====================================================================
echo.

explorer "%SEG_DIR%\dist_installer"
pause
exit /b 0
