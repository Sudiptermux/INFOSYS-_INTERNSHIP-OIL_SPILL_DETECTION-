# 🌊 Oil Spill Detection with Deep Learning

### Semantic segmentation of Sentinel-1 SAR imagery using a U-Net model with a pre-trained ResNet34 encoder backbone, served through an interactive radar-HUD dashboard.

*Infosys Springboard Internship Project*

🏆 **[View Internship Certificate](Infosys_Springboard_Internship_Certificate.pdf)**

---

## 📌 Overview

Oil spills are detected here as a **pixel-level segmentation problem**, not a whole-image classification one. The model outputs a binary mask showing exactly *where* a spill is located in a SAR (Synthetic Aperture Radar) satellite scene, not just whether one is present. This precise location mapping is critical for active maritime response teams to identify the affected boundaries and estimate spill volumes.

The pipeline covers the complete lifecycle:
1. **Speckle Denoising**: Bilateral filtering to clean raw SAR speckle noise while preserving crisp boundary lines.
2. **Augmentation**: Data augmentation using the Albumentations library.
3. **Deep Learning**: A U-Net decoder built over a pre-trained ResNet34 encoder.
4. **Interactive Dashboard**: A FastAPI-based radar-HUD console where operators can drop SAR images and receive instant segmentation overlays, area calculations, and live performance metrics.

---

## 🎯 Objectives

- **Precise Localization**: Segment oil-spill regions in SAR satellite imagery at the pixel level.
- **Triage Quantification**: Quantify spill extent as a percentage of the total image area.
- **Responder Dashboard**: Provide an interactive, responsive glassmorphic console that non-technical operators can use directly.
- **Reproducibility**: Ensure the entire pipeline is config-driven and easily reproducible.

---

## 🏗️ Pipeline Architecture

```
Sentinel-1 SAR Images + Masks
              │
              ▼
     Bilateral Denoising
              │
              ▼
   Augmentation (Flip, Rotate, Jitter)
              │
              ▼
   U-Net (ResNet34 encoder)
              │
              ▼
     Dice + BCE blended Loss
              │
              ▼
    IoU / F1 Validation Split
              │
              ▼
     FastAPI HUD Console
 (Upload → Segment → Area % → Alerting)
```

---

## 🧠 Tech Stack

| Layer | Tool / Library |
|---|---|
| **Deep Learning** | PyTorch, `segmentation-models-pytorch` (U-Net + ResNet34) |
| **Augmentation** | Albumentations |
| **Image Processing** | OpenCV, Pillow (PIL), NumPy |
| **Training Context** | Jupyter Notebook / Google Colab |
| **Web App Backend** | FastAPI (Python 3.9+) |
| **Web App Frontend** | HTML5, Vanilla CSS, JS (Interactive Radar HUD Scope) |

---

## 📂 Repository Structure

```
Oil_Spill_Detection/
├── Oil_Spill_Detection.ipynb           # Original notebook submitted for the internship evaluation
├── notebook/
│   └── Oil_Spill_Detection_Refined.ipynb # Refined Colab-run notebook with updated metrics
├── dashboard/                          # The local dashboard application
│   ├── main.py                         # FastAPI backend (loads weights, runs CPU inference)
│   ├── requirements.txt                # Dashboard dependencies
│   ├── DESIGN_NOTES.md                 # Design philosophy behind the Radar-HUD UI
│   ├── models/
│   │   └── best_model_custom_unet.pth  # Pre-trained ResNet34 U-Net weights (94.08% IoU)
│   ├── templates/index.html            # Dashboard frontend template (inline Certificate & HUD console)
│   └── static/
│       ├── style.css                   # Radar-HUD glassmorphic styling
│       ├── script.js                   # Interactive dragging, live uploads & REST API predictions
│       └── certificate.png             # Extracted high-resolution internship credential image
├── plots/                              # Saved training performance charts
│   ├── confusion_matrix.png
│   ├── pixel_intensity_distribution.png
│   ├── train_val_split.png
│   └── training_curves.png
├── Infosys_Springboard_Internship_Certificate.pdf # Original credential PDF
└── README.md                           # This file
```

---

## 🔬 Methodology

- **Preprocessing (Bilateral Filtering)**: We apply bilateral filtering to reduce noise in uniform ocean areas while keeping edge gradients sharp.
- **Model Architecture**: A U-Net structure using a ResNet34 backbone initialized with ImageNet weights, fine-tuned end-to-end.
- **Loss Function**: We blend Dice Loss and BCE Loss (50/50) to optimize for overlap precision on imbalanced spill regions.
- **Inference App**: The FastAPI server loads the state dict dynamically. For uploads, it converts the file to RGB, resizes it to 256x256, preprocesses it, runs a single-channel forward pass, and thresholds the sigmoid probabilities to output the cyan overlay.

---

## 🚀 Model Performance

The pre-trained U-Net model achieves the following validation performance on the Sentinel-1 split:

- **Pixel Accuracy**: **96.01%**
- **Mean IoU (Jaccard Index)**: **94.08%**
- **Dice Coefficient (F1-Score)**: **96.95%**
- **Precision**: **96.29%**
- **Recall (Sensitivity)**: **97.61%**
- **Specificity**: **93.04%**

---

## 🖥️ Running the Dashboard Locally

### 1. Install Dependencies
Initialize your virtual environment and install the dashboard requirements:
```bash
pip install -r dashboard/requirements.txt
```

### 2. Start the FastAPI App
Serve the dashboard using Uvicorn:
```bash
python -m uvicorn dashboard.main:app --port 8000
```

### 3. Open the Interface
Navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser:
* **Interactive Radar HUD Console**: Drag-and-drop or upload any SAR image patch to trigger real-time CPU U-Net predictions.
* **Inline Credential Verification**: View the verified Infosys Springboard internship certificate inline, complete with secure status banners.
* **Model performance panel**: View the active U-Net validation metrics loaded directly from the model checkpoint.

---

## 🌍 Applications

- **Marine Conservation**: Real-time detection of illegal oil discharges from shipping vessels.
- **Coastal Safety**: Port authority alerting and early-response coordination.
- **Compliance Auditing**: Off-shore platform leak inspection and environmental impact reporting.

---

## 📈 Future Enhancements

- [ ] Connect the dashboard directly to Sentinel Hub APIs for live satellite scene feeds.
- [ ] Add hard negative training categories (e.g., look-alikes, algae blooms, ship wakes) to reduce false positives.
- [ ] Deploy the app container to cloud services (AWS ECS or Hugging Face Spaces).
