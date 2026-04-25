#  Flood Area Detection using Satellite Image Segmentation

A deep learning-based semantic segmentation system that detects flood-affected regions from satellite imagery, deployed as an interactive web application.

---

##  Project Overview

This project uses a **ResNet34 U-Net** architecture with pretrained ImageNet weights to perform pixel-wise flood detection on SAR (Synthetic Aperture Radar) satellite images. Each pixel is classified as either **flooded (1)** or **non-flooded (0)**.

---

##  Model Architecture

- **Encoder:** ResNet34 pretrained on ImageNet (transfer learning)
- **Decoder:** U-Net with upsampling blocks
- **Input:** 3-channel image (256×256) derived from VH + VV SAR polarizations
- **Output:** Binary segmentation mask (sigmoid activation)
- **Training Strategy:** Two-phase training
  - Phase 1: Encoder frozen, decoder trained (15 epochs, lr=1e-4)
  - Phase 2: Full model fine-tuned (45 epochs, lr=1e-5)

---

##  Results

| Split | Loss | IoU | Dice Coefficient |
|-------|------|-----|-----------------|
| Train | 0.1786 | 0.8433 | 0.9105 |
| Validation (best) | 0.6608 | 0.6251 | 0.7204 |
| **Test Set** | **0.4587** | **0.6589** | **0.7778** |

---

##  Dataset

- **Source:** [NASA Flood Mapping Dataset](https://source.coop/nasa/floods)
- **Sensor:** Sentinel-1 SAR (C-band)
- **Polarizations:** VH + VV
- **Total source images:** 52 flood events across multiple countries
- **Patches extracted:** 1,040 patches of 256×256 pixels
- **Train/Test split:** Image-level split (41 train / 11 test source images)

---

##  Repository Structure
flood-detection/
├── app.py                    # Streamlit web application

├── final_flood_model.keras   # Trained model weights

├── requirements.txt          # Python dependencies

├── notebook.ipynb            # Training notebook

└── README.md
---

##  Running Locally

**1. Clone the repository:**
```bash
git clone https://github.com/YOURUSERNAME/flood-detection.git
cd flood-detection
```

**2. Create a virtual environment (Python 3.10 recommended):**
```bash
py -3.10 -m venv flood_env
flood_env\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the app:**
```bash
python -m streamlit run app.py
```

---

##  Web Application

The Streamlit app allows users to:
- Upload a satellite image (JPG/PNG/GeoTIFF)
- View the predicted flood segmentation mask
- Adjust the prediction threshold via sidebar slider
- View raw probability map
- Get flood severity classification (Low / Moderate / Severe)

---

##  Loss Function

Combined **Dice + Tversky Loss** (α=0.7):

$$\mathcal{L} = \mathcal{L}_{Dice} + \mathcal{L}_{Tversky}$$

Tversky loss with α=0.7 penalizes missed flood pixels (false negatives) more than false positives, addressing class imbalance where flood pixels represent only ~13% of the dataset.

---

##  Limitations

- Dataset limited to 52 source images — larger datasets would improve generalization
- RGB images are converted to grayscale to simulate SAR input in the web app
- Model performance may vary across geographic regions not seen during training

---

##  Tech Stack

- Python 3.10
- TensorFlow 2.16 / Keras 3
- segmentation-models
- Streamlit
- OpenCV
- Albumentations

