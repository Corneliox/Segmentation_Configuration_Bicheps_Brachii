# Biceps Brachii Ultrasound Segmentation Pipeline

Automated deep learning segmentation and muscle thickness quantification for **Biceps Brachii** from ArtUs/Telemed B-Mode ultrasound data.

---

## 📌 Project Architecture

The codebase is contained within `Segmentation_Configuration` for clean version control via GitHub. Large binary ultrasound files (`.bin`) and generated dataset images remain outside the repository in the parent directory (`E:\Probefix`).

```text
Probefix/                                       <-- Parent Directory (Data storage)
├── BIN FILE/                                   <-- Raw .bin ultrasound recordings
├── dataset/
│   ├── images/                                 <-- Extracted 8-bit grayscale US frames
│   └── masks/                                  <-- Ground truth binary masks
│
└── Segmentation_Configuration/                 <-- GITHUB REPOSITORY
    ├── configs/
    │   └── config.yaml                         <-- Project parameters & paths
    ├── matlab_converter/
    │   └── convert_bin2dataset.m               <-- B-mode conversion with EchoWave DR 78dB
    ├── src/
    │   ├── dataset.py                          <-- PyTorch Dataset & Augmentations
    │   ├── model.py                            <-- U-Net / SMP Model Architecture
    │   ├── utils.py                            <-- Metrics, Loss, & Muscle Thickness calculation
    │   ├── train.py                            <-- Model Training Loop
    │   └── evaluate.py                         <-- Evaluation & Visual Overlay
    ├── weights/                                <-- Model checkpoints (.pth)
    ├── requirements.txt                        <-- Python dependencies
    ├── .gitignore                              <-- Git exclusions
    └── README.md
```

---

## 🚀 Quickstart Guide

### 1. Convert `.bin` Video to Grayscale Frames
Open MATLAB and run the script:
```matlab
run('matlab_converter/convert_bin2dataset.m')
```
* Applies Hilbert Envelope Detection and **Logarithmic Dynamic Range Compression (DR = 78 dB)** matching EchoWave software.
* Generates normalized 8-bit PNG images in `../dataset/images/`.

### 2. Annotate Ground Truth Masks
Annotate the Biceps Brachii region (between the superficial fascia and the deep aponeurosis):
* Tools: [Labelme](https://github.com/wkentaro/labelme) or [CVAT](https://www.cvat.ai/).
* Save the binary masks (white = Biceps Brachii, black = Background) in `../dataset/masks/` with names matching the image files (e.g. `img_xxx_slice_0001.png`).

### 3. Setup Python Environment
```bash
cd Segmentation_Configuration
pip install -r requirements.txt
```

### 4. Train the Segmentation Model
```bash
python src/train.py
```
* Automatically splits dataset into **80% Train, 10% Validation, 10% Test**.
* Saves the best performing model checkpoint to `weights/best_model.pth`.

### 5. Evaluate and Measure Muscle Thickness
```bash
python src/evaluate.py
```
* Computes **Dice Coefficient** and **IoU (Intersection over Union)** on the test set.
* Automatically measures **Muscle Thickness (in mm)** along the muscle profile.
* Saves visual side-by-side comparison plots in `outputs/`.

---

## ⚙️ Configuration
All hyperparameters and paths are customizable in [`configs/config.yaml`](file:///e:/Probefix/Segmentation_Configuration/configs/config.yaml):
* `model.architecture`: `Unet`, `UnetPlusPlus`, `DeepLabV3Plus`
* `model.encoder_name`: `resnet34`, `resnet50`, `efficientnet-b2`
* `training.epochs`: Number of training epochs (default: `50`)
* `training.batch_size`: Batch size (default: `8`)

---

## 📦 GitHub Versioning Workflow
To initialize and push this configuration to your GitHub repository:

```bash
cd E:\Probefix\Segmentation_Configuration
git init
git add .
git commit -m "Initial commit: Biceps Brachii segmentation pipeline"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```
