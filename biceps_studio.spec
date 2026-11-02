# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

BASE_DIR = os.path.abspath(SPECPATH)
SRC_DIR = os.path.join(BASE_DIR, 'src')

datas = [
    (os.path.join(BASE_DIR, 'presets'), 'presets'),
    (os.path.join(BASE_DIR, 'configs'), 'configs'),
    (os.path.join(BASE_DIR, 'matlab_converter'), 'matlab_converter'),
    (SRC_DIR, 'src'),
]

# If weights folder exists, include weights
weights_dir = os.path.join(BASE_DIR, 'weights')
if os.path.exists(weights_dir):
    datas.append((weights_dir, 'weights'))

hidden_imports = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'torch',
    'torchvision',
    'cv2',
    'numpy',
    'pandas',
    'yaml',
    'scipy',
    'model',
    'utils',
]

a = Analysis(
    [os.path.join(SRC_DIR, 'biceps_pipeline_studio.py')],
    pathex=[BASE_DIR, SRC_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'IPython', 'notebook', 'pytest', 'sphinx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Biceps_Brachii_Pipeline_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Biceps_Brachii_Pipeline_Studio',
)
