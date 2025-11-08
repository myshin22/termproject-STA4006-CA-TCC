# CA-TCC: Video-Level Few-Shot Learning for Exercise Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Self-Supervised Contrastive Learning for Few-Shot Exercise Recognition using Wearable IMU Sensors**

This repository contains the implementation of video-level few-shot learning using CA-TCC (Contrastive learning with Augmentation and Temporal Coherence) for exercise recognition with limited labeled data.

---

## 🎯 Key Results

| Method | 1-shot | 5-shot | Improvement |
|--------|--------|--------|-------------|
| **Baseline** (No pretraining) | 56.48% | 75.68% | — |
| **CA-TCC** (Self-supervised pretraining) | 54.40% | **88.96%** | **+13.28%** ⭐ |

**Key Finding:** Self-supervised contrastive pretraining significantly improves 5-shot learning performance (p<0.001), achieving **17.5% relative improvement** over baselines.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

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
✅ Balanced representation across all exercise types
✅ No data leakage (windows from same video stay together)
✅ Clear semantic meaning ("N labeled trials per exercise")

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

### 🛠️ Production-Ready Code
- Automated experiment runner (`run_experiments_video.sh`)
- Automatic result comparison with statistical significance
- Detailed logging with video statistics
- Clean, modular architecture

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/CA-TCC-VideoLevel.git
cd CA-TCC-VideoLevel

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
openpyxl>=3.0.0  # For Excel export (optional)
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
- `exercise`: Exercise label (e.g., "benchpress", "deadlift")
- `video_name`: Unique video/trial identifier
- Sensor columns: 12 IMU channels (left + right wrist: 3-axis acc + 3-axis gyro)

**Modify the data preparation script:**

Edit `prepare_exercise_data_video.py`:
```python
INPUT_FILE = 'path/to/your/data.csv'
OUTPUT_DIR = 'data/YourDataset'
SAMPLING_RATE = 66  # Your sensor sampling rate (Hz)
```

**Run data preparation:**
```bash
python prepare_exercise_data_video.py
```

This creates:
- `data/YourDataset/train.pt` - Full training set
- `data/YourDataset/train_1shot.pt` - 1 video per class
- `data/YourDataset/train_5shot.pt` - 5 videos per class
- `data/YourDataset/test.pt` - Test set
- `data/YourDataset/val.pt` - Validation set

### Step 2: Create Config File

Create `config_files/YourDataset_Configs.py`:

```python
class Config(object):
    def __init__(self):
        # Data
        self.input_channels = 12      # IMU channels
        self.num_classes = 10         # Number of exercise types

        # Model architecture
        self.kernel_size = 8
        self.final_out_channels = 128
        self.dropout = 0.35
        self.features_len = 43        # Depends on sequence length

        # Training
        self.num_epoch = 40
        self.batch_size = 128
        self.lr = 3e-4
        self.beta1 = 0.9
        self.beta2 = 0.99

        # Data
        self.drop_last = True

        # Import auxiliary configs
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

# Copy these classes from ExerciseIMU_Configs.py
class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8

class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True

class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6
```

### Step 3: Run Experiments

**Option 1: Full automated experiments (recommended)**

```bash
bash run_experiments_video.sh YourDataset 0 4
```

This runs all experiments (5 seeds × 6 methods = 30 runs):
- Baseline 1-shot, 5-shot, 100%
- CA-TCC self-supervised pretraining
- CA-TCC fine-tuning 1-shot, 5-shot

**Option 2: Single experiment (for testing)**

```bash
# Test with 1-shot
python main_video.py \
  --experiment_description "Test" \
  --run_description "Quick" \
  --seed 0 \
  --selected_dataset YourDataset \
  --training_mode supervised_1shot \
  --device cuda:0
```

### Step 4: Analyze Results

```bash
python compare_results_video.py --experiment_name CA_TCC_VideoLevel
```

**Output:**
```
VIDEO-LEVEL ACCURACY (Majority Voting - More Important!)
====================================================================================================
Method                                   Seeds    Mean Acc     Std        vs Baseline 1-shot   vs Baseline 5-shot
----------------------------------------------------------------------------------------------------
Baseline_1shot                           5        56.48        1.0852
Baseline_5shot                           5        75.68        2.7058     p=0.0000 ***
CA-TCC_ft_5shot                          5        88.96        2.9241     p=0.0000 ***         p=0.0002 ***
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
│   ├── README_GITHUB.md                   # This file
│   ├── FINAL_RESEARCH_REPORT.md          # Complete research report
│   ├── README_VIDEO_LEVEL.md             # Detailed guide
│   ├── VIDEO_LEVEL_EXPERIMENTS.md        # Technical docs
│   └── CODE_READING_GUIDE.md             # Code structure guide
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
│   │   └── ExerciseIMU_Configs.py        # Example config
│   └── utils.py                           # Metrics, logging
│
├── 💾 Data (Not in Git - Create Locally)
│   └── data/YourDataset/
│       ├── train.pt
│       ├── train_1shot.pt
│       ├── train_5shot.pt
│       ├── test.pt
│       └── val.pt
│
└── 📊 Results (Not in Git)
    └── experiments_logs/
```

---

## 📖 Documentation

### Quick References

- **[FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md)** - Complete research paper
  - Model architecture details
  - Dataset description
  - Experimental design
  - Results and analysis

- **[README_VIDEO_LEVEL.md](README_VIDEO_LEVEL.md)** - Comprehensive guide
  - Detailed setup instructions
  - All training modes explained
  - Troubleshooting tips

- **[CODE_READING_GUIDE.md](CODE_READING_GUIDE.md)** - Code walkthrough
  - Step-by-step code explanation
  - File reading order
  - Implementation details

### Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `supervised_1shot` | Train from scratch with 1 video/class | Baseline for 1-shot |
| `supervised_5shot` | Train from scratch with 5 videos/class | Baseline for 5-shot |
| `supervised` | Train from scratch with all data | Upper bound |
| `self_supervised` | Self-supervised pretraining | Prepare for CA-TCC |
| `ft_1shot` | Fine-tune pretrained model (1-shot) | CA-TCC 1-shot |
| `ft_5shot` | Fine-tune pretrained model (5-shot) | CA-TCC 5-shot |

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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original CA-TCC implementation: [emadeldeen24/TS-TCC](https://github.com/emadeldeen24/TS-TCC)
- Video-level few-shot learning extension: This work
- Dataset: Custom exercise IMU dataset

---

## 🔧 Troubleshooting

### Common Issues

**1. Data shape mismatch**
```python
# Check your data shape
import torch
data = torch.load('data/YourDataset/train.pt')
print(data['samples'].shape)  # Should be [N, 12, seq_len]
```

**2. CUDA out of memory**
```python
# Reduce batch size in config file
self.batch_size = 64  # or 32
```

**3. Import errors**
```bash
# Make sure you're in the CA-TCC directory
cd /path/to/CA-TCC
python main_video.py ...
```

### Getting Help

1. Check [CODE_READING_GUIDE.md](CODE_READING_GUIDE.md) for code explanations
2. Read [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md) for methodology
3. Open an issue on GitHub with:
   - Error message
   - Data shape
   - Config file
   - Command you ran

---

## 🚀 Advanced Usage

### Custom Augmentations

Edit `dataloader/augmentations.py`:
```python
class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5  # Stronger jitter
        self.max_seg = 12              # More time masking
```

### Hyperparameter Tuning

Edit `config_files/YourDataset_Configs.py`:
```python
self.lr = 1e-4           # Lower learning rate
self.num_epoch = 60      # More epochs
self.dropout = 0.5       # More regularization
```

### Multi-GPU Training

```bash
# Set device in command
python main_video.py --device cuda:1 ...
```

---

## 📊 Expected Performance

| Dataset | 1-shot | 5-shot | 100% |
|---------|--------|--------|------|
| ExerciseIMU (10 classes) | 54-56% | 86-90% | 94-96% |
| Your dataset | Varies | Varies | Varies |

**Note:** Performance depends on:
- Number of classes
- Data quality
- Sensor noise level
- Inter-class similarity

---
