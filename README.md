# ECG Arrhythmia Detection with PyTorch

## Overview
This repository contains deep learning models for detecting **Atrial Fibrillation (AFib)**, **Atrial Flutter (AFlutter)**, and **Premature Ventricular Contractions (PVC)** from 3-lead ECG signals.

- **AFib detection** uses a dedicated 1D CNN model optimized via Bayesian hyperparameter tuning.
- **AFlutter & PVC detection** uses a shared multi-branch convolutional architecture designed for multi-lead time-domain, frequency-domain, and beat-level features, with the ability to scale to additional arrhythmias in the future.

The models are trained and evaluated on publicly available ECG datasets from **[PhysioNet](https://physionet.org/)**, including PTB-XL, Chapman-Shaoxing, Ningbo, and Georgia 12-lead datasets (downsampled to 3 leads).

---

## Features
- Separate specialist model for AFib with hyperparameter tuning via [scikit-optimize](https://scikit-optimize.github.io/).
- Shared multi-task model for AFlutter and PVC with:
  - Residual CNN backbone
  - Multi-head attention
  - Dedicated frequency-domain and beat-morphology branches
  - Lead-presence vector integration
- Custom data preprocessing pipeline for reduced-lead WFDB recordings
- Per-class metrics, PR curves, and confusion matrices
- Easily extendable to other arrhythmias

---

## Model Architecture

### **AFib Specialist Model**
- **Type**: 1D Convolutional Neural Network (CNN)
- **Pipeline**:
  1. **Input**: Single ECG lead segment (2-second windows, 360 Hz sampling).
  2. **Conv Layers**: Three stacked Conv1D layers with ReLU activations to progressively learn temporal features.
  3. **MaxPooling**: Reduces sequence length while retaining dominant activations.
  4. **Global Average Pooling**: Collapses time dimension for global signal representation.
  5. **Dropout Layer**: Regularization to prevent overfitting.
  6. **Dense Layer**: Final binary output neuron with sigmoid activation.
- **Hyperparameter Optimization**: 
  - Bayesian search (`skopt.gp_minimize`) over number of filters, kernel size, dropout, learning rate, and batch size.
  - Optimized for AUROC + AUPRC.

---

### **Shared Multi-Branch Model (AFlutter & PVC)**
- **Purpose**: Detect multiple arrhythmias from 3-lead ECG (I, II, V2), with flexibility to scale to more arrhythmias.
- **Backbone**:  
  - **Residual CNN**: Stacked convolutional blocks with skip connections for stable deep feature extraction.
- **Branches**:
  1. **Time-Domain Branch**: Learns raw waveform morphology and rhythm.
  2. **Frequency-Domain Branch**: Applies FFT to extract spectral features, feeding into dedicated CNN layers.
  3. **Beat-Morphology Branch**: Uses R-peak detection (via `scipy.signal.find_peaks`) on Lead II to encode up to 10 normalized beat locations as auxiliary input.
- **Lead Presence Vector**:  
  - 12-element binary vector indicating which ECG leads are available; supports reduced-lead inference.
- **Fusion & Output**:
  - Concatenates branch outputs + lead vector.
  - Fully connected layers map to sigmoid outputs for each arrhythmia (multi-label classification).
- **Loss Function**: `BCEWithLogitsLoss` (per-label binary classification).
- **Regularization**:
  - Gradient clipping to max norm = 1.0.
  - Early stopping with patience = 5 epochs.

---

## Repository Structure
```
├── afib.py                  # AFib specialist model & training script
├── model_code.py            # Multi-branch CNN architecture (AFlutter & PVC)
├── team_code.py             # Training code for AFlutter & PVC
├── evaluate_model.py        # Model evaluation script with metrics & plots
├── train_model.py           # CLI entry point for model training
├── extract_leads_wfdb.py    # Script to reduce full-lead ECGs to 3-leads
├── helper_code.py           # Utility functions for ECG data processing
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ecg-arrhythmia-detection-pytorch.git
cd ecg-arrhythmia-detection-pytorch

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare the datasets
Download datasets from PhysioNet:
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- [Chapman-Shaoxing](https://physionet.org/content/challenge-2020/1.0.2/)
- [Ningbo](https://physionet.org/content/ningbo-ecg/1.0.0/)
- [Georgia 12-lead ECG](https://physionet.org/content/georgia-12lead-ecg-challenge/1.0.0/)

Reduce them to 3 leads (I, II, V2) using:
```bash
python extract_leads_wfdb.py     -i /path/to/full_dataset     -l I II V2     -o /path/to/reduced_dataset
```

---

### 2. Train the models

**AFib model:**
```bash
python afib.py
```

**AFlutter & PVC model:**
```bash
python train_model.py data_dir model_output_dir
```

---

### 3. Evaluate the models
```bash
python evaluate_model.py
```
Generates:
- Per-class precision, recall, F1
- False positive & false negative rates
- Precision-recall curves (`.png` file)

---

## Results
| Model                | Class     | Precision | Recall | F1    |
|----------------------|-----------|-----------|--------|-------|
| AFib Specialist      | AFib      | 0.98      | 0.98   | 0.98  |
| Shared Model         | AFlutter  | 0.85      | 0.83   | 0.84  |
|                      | PVC       | 0.80      | 0.76   | 0.78  |

---

## Dataset Citation
This work uses datasets from [PhysioNet](https://physionet.org/) under their respective licenses. Please cite:
> Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. *Circulation*. 2000;101(23):e215–e220.

---

## License
MIT License — see LICENSE file for details.

---

## Author
**Yousouf Farooq** – Georgia Tech, Biomedical Engineering & Computer Science  
Contact: [your.email@example.com]
