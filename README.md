# CA-TCC: Video-Level Few-Shot Learning for Exercise Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Self-Supervised Contrastive Learning for Few-Shot Exercise Recognition using Wearable IMU Sensors**

This repository implements video-level few-shot learning using CA-TCC (Contrastive learning with Augmentation and Temporal Coherence) for exercise recognition with limited labeled data.

---

## 🎯 Key Results

### Overall Performance (Video-Level Accuracy)

| Method | 1-shot | 5-shot | 100% | Improvement |
|--------|--------|--------|------|-------------|
| **Baseline** (Random init) | 56.48% ± 1.09 | 75.68% ± 2.71 | 95.68% ± 1.93 | — |
| **CA-TCC** (Self-supervised) | 54.40% ± 5.66 | **88.96% ± 2.92** | — | **+13.28%** ⭐ |

### Key Findings

1. **5-shot CA-TCC achieves best few-shot performance:**
   - 88.96% accuracy vs. 75.68% baseline
   - **+13.28 percentage points** absolute improvement
   - **+17.5% relative improvement**
   - Highly statistically significant (p<0.001)

2. **1-shot shows no improvement:**
   - 54.40% vs. 56.48% baseline (not significant, p=0.491)
   - Self-supervised pretraining requires sufficient fine-tuning data

3. **CA-TCC benefits more from additional data:**
   - CA-TCC: 1-shot→5-shot gain = +34.56pp (+63.5% relative)
   - Baseline: 1-shot→5-shot gain = +19.20pp (+34.0% relative)

### Per-Class Performance (CA-TCC 5-shot)

| Exercise | Test Videos | Accuracy |
|----------|-------------|----------|
| Deadlift | 16 | **93.8%** |
| Bench Press | 13 | **92.3%** |
| OHP | 13 | **92.3%** |
| Dips | 12 | **91.7%** |
| Barbell Row | 11 | **90.9%** |
| Barbell Curl | 12 | 85.0% |
| Pull-up | 13 | 84.6% |
| BTE | 12 | 83.3% |
| Push-up | 12 | 83.3% |
| Lat Pulldown | 11 | 81.8% |

**All classes achieve >81% accuracy** with only 5 training videos per class!

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)

---

## 🔍 Overview

### Problem

Traditional supervised learning requires large amounts of labeled data, which is expensive and time-consuming to collect for wearable sensor applications. How can we recognize new exercises with only a few labeled examples per exercise?

### Solution

**Video-Level Few-Shot Learning with Self-Supervised Pretraining:**

1. **Self-supervised pretraining** on all unlabeled data (373 videos)
2. **Fine-tuning** with minimal labeled data (5 videos per exercise class)
3. **Video-level evaluation** using majority voting for robust predictions

### Why "Video-Level"?

Unlike traditional percentage-based few-shot learning (e.g., "1% of windows"), we use:

- **1-shot** = 1 complete exercise video per class (10 videos total)
- **5-shot** = 5 videos per class from different subjects (50 videos)

This ensures:
- ✅ Balanced representation across all exercise types
- ✅ No data leakage (windows from same video stay together)
- ✅ Clear semantic meaning ("N labeled trials per exercise")

---

## ✨ Features

### 🎓 Self-Supervised Learning
- **Temporal Contrastive Learning** (CA-TCC): Learn temporal patterns without labels
- **Predictive Coding**: Predict future timesteps from past context
- **Data Augmentations**: Jitter, scaling, time masking

### 📊 Video-Level Few-Shot Learning
- **Balanced sampling** across exercise classes
- **Subject diversity**: 5-shot samples from different subjects
- **Majority voting**: Aggregate window predictions to video-level

### 📈 Comprehensive Evaluation
- **Window-level accuracy**: Standard metric
- **Video-level accuracy**: Majority voting (more robust)
- **Statistical testing**: t-tests with multiple seeds (n=5)
- **Per-class metrics**: Precision, recall, F1-score

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/myshin22/termproject-STA4006-CA-TCC.git
cd termproject-STA4006-CA-TCC

# Install requirements
pip install -r requirements.txt
```

**Requirements:**
```
torch>=1.10.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
```

---

## ⚡ Quick Start

### Step 1: Prepare Your Data

Your dataset should be in CSV format with the following structure:

```csv
TimeStamp,subject_id,exercise,video_name,right_ax,right_ay,right_az,right_gx,right_gy,right_gz,left_ax,left_ay,left_az,left_gx,left_gy,left_gz
1234567890,1,benchpress,video_001,0.5,0.3,-9.8,0.1,0.2,0.3,0.4,0.2,-9.7,0.2,0.1,0.2
...
```

**Required columns:**
- `TimeStamp`: Unix timestamp or frame number
- `subject_id`: Subject identifier
- `exercise`: Exercise label
- `video_name`: Unique video/trial identifier
- Sensor columns: 12 IMU channels (left + right wrist: 3-axis acc + 3-axis gyro)

**Run data preparation:**
```bash
python prepare_exercise_data_video.py
```

### Step 2: Run Experiments

**Full automated experiments (recommended):**

```bash
bash run_experiments_video.sh ExerciseIMU 0 4
```

This runs all experiments (5 seeds × 6 methods = 30 runs):
- Baseline 1-shot, 5-shot, 100%
- CA-TCC self-supervised pretraining
- CA-TCC fine-tuning 1-shot, 5-shot

### Step 3: Analyze Results

```bash
python compare_results_video.py --experiment_name CA_TCC_VideoLevel
```

---

## 📁 Project Structure

```
CA-TCC/
├── 📄 Main Scripts
│   ├── main_video.py                      # Training script
│   ├── prepare_exercise_data_video.py     # Data preparation
│   ├── run_experiments_video.sh           # Automated experiments
│   └── compare_results_video.py           # Results analysis
│
├── 📚 Documentation
│   ├── README.md                          # This file
│   ├── FINAL_RESEARCH_REPORT.md          # Complete research report
│   └── VIDEO_LEVEL_EXPERIMENTS.md        # Technical docs
│
├── 🗂️ Source Code
│   ├── models/
│   │   ├── model.py                       # CNN encoder
│   │   ├── TC.py                          # Temporal contrastive module
│   │   └── attention.py                   # Transformer
│   ├── dataloader/
│   │   ├── dataloader_video.py           # Video-level dataloader
│   │   └── augmentations.py              # Data augmentations
│   ├── trainer/
│   │   └── trainer.py                     # Training loop
│   ├── config_files/
│   │   └── ExerciseIMU_Configs.py        # Hyperparameters
│   └── utils.py                           # Metrics, logging
│
└── 💾 Data (Not in Git - Create Locally)
    └── data/ExerciseIMU/
        ├── train.pt
        ├── train_1shot.pt
        ├── train_5shot.pt
        ├── test.pt
        └── val.pt
```

---

## 📊 Dataset

### Dataset Description

- **Exercises**: 10 resistance exercises (Bench Press, Deadlift, OHP, etc.)
- **Subjects**: 13 participants
- **Sensors**: Bilateral wrist-worn IMU (12 channels total)
- **Sampling rate**: 66 Hz
- **Window size**: 5 seconds (330 frames) with 2-second stride

### Data Statistics

| Split | Videos | Windows | Channels | Sequence Length |
|-------|--------|---------|----------|----------------|
| **Training (Full)** | 373 | 2,978 | 12 | 330 |
| **Training (1-shot)** | 10 | 70 | 12 | 330 |
| **Training (5-shot)** | 50 | 370 | 12 | 330 |
| **Validation** | 125 | 904 | 12 | 330 |
| **Test** | 125 | 904 | 12 | 330 |

**Subject-based split:** 80/20 train/test split ensures generalization to new users.

---

## 🧠 Model Architecture

### CA-TCC Framework

1. **Encoder Network**: 3-layer 1D CNN extracts features from time-series data
2. **Temporal Contrastive Module**: Transformer-based module enforces temporal coherence
3. **Classification Head**: Linear classifier for supervised fine-tuning

### Encoder Architecture

```
Input: [Batch, 12 channels, 330 timesteps]
    ↓
Conv Block 1: Conv1D(12→32, k=8) + BatchNorm + ReLU + MaxPool + Dropout(0.35)
    ↓
Conv Block 2: Conv1D(32→64, k=8) + BatchNorm + ReLU + MaxPool
    ↓
Conv Block 3: Conv1D(64→128, k=8) + BatchNorm + ReLU + MaxPool
    ↓
Output: [Batch, 128, 43] → Flatten → [Batch, 5504]
    ↓
Linear Classifier(5504 → 10 classes)
```

### Training Procedure

**Two-Stage Training:**

1. **Self-supervised pretraining:**
   - Input: All 373 training videos (unlabeled)
   - Loss: Temporal contrastive loss (NCE)
   - Epochs: 40

2. **Supervised fine-tuning:**
   - Input: 10 (1-shot) or 50 (5-shot) labeled videos
   - Loss: Cross-entropy
   - Epochs: 40
   - Fine-tune all layers (encoder + classifier)

---

## 📈 Results

### Statistical Significance

| Comparison | p-value | Significance |
|------------|---------|--------------|
| CA-TCC 5-shot vs. Baseline 5-shot | 0.0002 | *** (Highly significant) |
| CA-TCC 1-shot vs. Baseline 1-shot | 0.491 | ns (Not significant) |

### Window vs. Video-Level Accuracy

| Method | Window-Level | Video-Level |
|--------|--------------|-------------|
| **CA-TCC 5-shot** | 88.61% ± 2.93 | 88.96% ± 2.92 |
| **Baseline 5-shot** | 76.55% ± 2.14 | 75.68% ± 2.71 |

**Observation:** Window-level and video-level accuracies are nearly identical, suggesting stable predictions across windows.

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{catcc2023,
  title={Self-Supervised Contrastive Representation Learning for Semi-Supervised Time-Series Classification},
  author={Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Kwoh, Chee-Keong and Li, Xiaoli and Guan, Cuntai},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={12},
  pages={15604--15618},
  year={2023},
  doi={10.1109/TPAMI.2023.3308189}
}
```

**Original CA-TCC paper:**
```bibtex
@inproceedings{tstcc2021,
  title={Time-Series Representation Learning via Temporal and Contextual Contrasting},
  author={Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Kwoh, Chee Keong and Li, Xiaoli and Guan, Cuntai},
  booktitle={IJCAI},
  pages={2352--2359},
  year={2021}
}
```
---

**Last Updated:** November 8, 2025

