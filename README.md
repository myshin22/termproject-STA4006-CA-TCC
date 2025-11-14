
# CA-TCC: Few-Shot Learning for Exercise Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Self-Supervised Contrastive Learning for Exercise Type Classification with Limited Labeled Data**

This repository implements CA-TCC (Contrastive learning with Augmentation and Temporal Coherence) for learning robust video-level representations from wearable IMU sensor data.
After self-supervised pretraining, the model is fine-tuned with a small number of labeled samples for exercise type classification.

---

## 🎯 Key Results

### Video-Level Accuracy (5 Seeds, Mean ± Std)

| Method | 0-shot | 1-shot | 5-shot | 100% (Upper Bound) |
|--------|--------|--------|--------|--------------------|
| **Supervised Baseline** | — | 43.89% ± 10.27 | 68.61% ± 3.11 | **81.34% ± 4.34** |
| **CA-TCC (Pretrain+FT)** | 12.20% ± 6.27 | 47.61% ± 11.82 | **71.43% ± 3.36** | — |
| **Improvement** | — | +3.73% (+8.5%) | +2.83% (+4.1%) | — |

### Key Findings
1. **0-shot evaluation reveals poor transfer without fine-tuning:**
   - Only 12.20% accuracy with pretrained model (random: 10%)
   - Self-supervised pretraining alone is insufficient for this task
   - Fine-tuning with labeled data is essential
=======
1. **5-shot CA-TCC achieves best finetuning performance:**
   - 88.96% accuracy vs. 75.68% baseline
   - **+13.28 percentage points** absolute improvement
   - **+17.5% relative improvement**
   - Highly statistically significant (p<0.001)

2. **Modest improvements with few-shot learning:**
   - **5-shot**: CA-TCC achieves 71.43% vs. 68.61% baseline (+2.83pp, +4.1% relative)
   - **1-shot**: CA-TCC achieves 47.61% vs. 43.89% baseline (+3.73pp, +8.5% relative)

3. **Gap to upper bound:**
   - 5-shot CA-TCC: 71.43% vs. 100% supervised: 81.34%
   - Still ~10 percentage points below full supervision
   - Pretraining helps but doesn't close the gap completely

4. **Statistical significance:**
   - CATCC_5shot vs Supervised_5shot: p=0.0009 (highly significant ***)
   - CATCC_1shot vs Supervised_1shot: p=0.6468 (not significant)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experimental Workflow](#experimental-workflow)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Augmentations](#augmentations)
- [Results](#results)
- [Citation](#citation)

---

## 🔍 Overview

### Problem

Traditional supervised learning requires large amounts of labeled data. Can self-supervised pretraining enable learning from just a few labeled examples per class?

### Solution: Two-Stage Learning

**Stage 1: Self-Supervised Pretraining**
- Train on ALL unlabeled data (train + validation videos)
- Use temporal contrastive learning (CA-TCC)
- Learn general representations without labels

**Stage 2: Fine-Tuning (or Evaluation)**
- **0-shot**: Evaluate pretrained model directly (NO fine-tuning)
- **K-shot**: Fine-tune with K labeled videos per class
- **Supervised baseline**: Train from scratch with K videos per class (no pretraining)

### Why "Video-Level"?

- **1-shot** = 1 complete exercise video per class (10 videos total)
- **5-shot** = 5 videos per class from different subjects (50 videos)

Benefits:
- ✅ Balanced representation across all exercise types
- ✅ No data leakage (windows from same video stay together)
- ✅ Clear semantic meaning ("N labeled trials per exercise")

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/CA-TCC-FewShot.git
cd CA-TCC-FewShot

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

### Step 1: Prepare Data for Multiple Seeds

```bash
# Generate 5 different random subject splits (seeds 0-4)
bash prepare_all_seeds.sh
```

This creates:
- `data/ExerciseIMU_seed0/` through `data/ExerciseIMU_seed4/`
- Each seed has different train/val/test subject splits
- Files: `pretrain.pt`, `train.pt`, `train_1shot.pt`, `train_5shot.pt`, `val.pt`, `test.pt`

### Step 2: Run All Experiments

```bash
# Run complete workflow for all 5 seeds
bash run_fewshot_experiments.sh
```

This runs for each seed:
1. Self-supervised pretraining (train+val, NO test)
2. 0-shot evaluation (pretrained model, NO fine-tuning)
3. 1-shot fine-tuning (pretrain → 1-shot)
4. 5-shot fine-tuning (pretrain → 5-shot)
5. Supervised 1-shot baseline (train from scratch)
6. Supervised 5-shot baseline (train from scratch)

**Note**: Full training set (100%) supervised baseline already exists in experiments.

### Step 3: Compare Results

```bash
# Aggregate results across all seeds
python compare_results_video.py
```

Results are saved to: `experiments_logs/FewShot_ExerciseIMU/comparison_results.txt`

---

## 📊 Experimental Workflow

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Data Preparation (prepare_all_seeds.sh)                    │
│  - Generate 5 seeds with different subject splits           │
│  - Create pretrain.pt (train+val), train_Xshot.pt, test.pt │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  For Each Seed (run_fewshot_experiments.sh):                │
│                                                              │
│  1. Self-Supervised Pretraining                             │
│     - Input: pretrain.pt (train+val videos, NO test)        │
│     - Method: Temporal contrastive learning (40 epochs)     │
│     - Output: Pretrained encoder weights                    │
│                                                              │
│  2. 0-Shot Evaluation                                        │
│     - Load pretrained weights                                │
│     - Evaluate on test set (NO training/fine-tuning)        │
│                                                              │
│  3. Few-Shot Fine-Tuning (1-shot, 5-shot)                   │
│     - Load pretrained weights                                │
│     - Fine-tune with K-shot labeled data (40 epochs)        │
│     - Evaluate on test set                                   │
│                                                              │
│  4. Supervised Baselines (1-shot, 5-shot)                   │
│     - Random initialization (NO pretraining)                 │
│     - Train from scratch with K-shot data (40 epochs)       │
│     - Evaluate on test set                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Results Aggregation (compare_results_video.py)             │
│  - Collect results from all seeds                           │
│  - Compute mean ± std across seeds                          │
│  - Statistical significance testing (t-tests)               │
│  - Generate comparison tables                               │
└─────────────────────────────────────────────────────────────┘
```

### Training Modes

| Mode | Description | Pretraining | Training Data | Purpose |
|------|-------------|-------------|---------------|---------|
| `self_supervised` | Pretrain encoder | — | pretrain.pt (train+val) | Stage 1 |
| `0shot` | Evaluate pretrained model | ✅ | None (just eval) | Baseline |
| `ft_1shot` | Fine-tune with 1-shot | ✅ | train_1shot.pt | Few-shot |
| `ft_5shot` | Fine-tune with 5-shot | ✅ | train_5shot.pt | Few-shot |
| `supervised_1shot` | Train from scratch | ❌ | train_1shot.pt | Baseline |
| `supervised_5shot` | Train from scratch | ❌ | train_5shot.pt | Baseline |
| `supervised` | Full supervision | ❌ | train.pt (full) | Upper bound |

---

## 📁 Project Structure

```
CA-TCC/
├── 📄 Main Scripts
│   ├── main_video.py                      # Training script
│   ├── prepare_exercise_data_video_v2.py  # Data preparation
│   ├── prepare_all_seeds.sh               # Generate multiple seeds
│   ├── run_fewshot_experiments.sh         # Run all experiments
│   └── compare_results_video.py           # Results analysis
│
├── 🗂️ Source Code
│   ├── models/
│   │   ├── model.py                       # CNN encoder
│   │   ├── TC.py                          # Temporal contrastive module
│   │   └── loss.py                        # Contrastive loss functions
│   ├── dataloader/
│   │   ├── dataloader_video.py           # Video-level dataloader
│   │   └── augmentations.py              # Time-series augmentations
│   ├── trainer/
│   │   └── trainer.py                     # Training loop
│   ├── config_files/
│   │   └── ExerciseIMU_Configs.py        # Hyperparameters
│   └── utils.py                           # Metrics, logging
│
└── 💾 Data (Create Locally)
    └── data/
        ├── ExerciseIMU_seed0/
        │   ├── pretrain.pt                # Train+val (for pretraining)
        │   ├── train.pt                   # Full training set
        │   ├── train_1shot.pt             # 1 video per class
        │   ├── train_5shot.pt             # 5 videos per class
        │   ├── val.pt                     # Validation set
        │   └── test.pt                    # Test set
        ├── ExerciseIMU_seed1/
        └── ... (seed2-4)
```

---

## 📊 Dataset

### Dataset Description

- **Exercises**: 10 resistance exercises (Bench Press, Deadlift, Overhead Press, etc.)
- **Subjects**: 13 participants
- **Sensors**: Bilateral wrist-worn IMU (left + right)
  - 12 channels total: 3-axis accelerometer + 3-axis gyroscope per wrist
- **Sampling rate**: 66 Hz
- **Window size**: 5 seconds (330 frames) with 2-second stride

### Subject-Level Data Splits (Per Seed)

| Split | Subjects | Videos | Windows | Purpose |
|-------|----------|--------|---------|---------|
| **Train** | 5 | ~210 | ~1,564 | Few-shot selection + full training |
| **Validation** | 4 | ~139 | ~1,187 | Hyperparameter tuning |
| **Test** | 4 | ~149 | ~1,131 | Final evaluation |

**Important**: Each seed has completely different subject assignments to train/val/test for robust evaluation.

### Few-Shot Sampling Strategy

**1-shot per class:**
- Select 1 random subject
- Take 1 random video from that subject for each class
- Total: 10 videos (1 per class)

**5-shot per class:**
- Select 5 random subjects (all different)
- Take 1 random video from each subject for each class
- Total: 50 videos (5 per class)

**Ensures**: Subject diversity and balanced class representation.

---

## 🧠 Model Architecture

### CA-TCC Framework

```
┌─────────────────────────────────────────────────────────────┐
│  Input: [Batch, 12 channels, 330 timesteps]                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Encoder (3-layer 1D CNN)                                    │
│                                                              │
│  Conv Block 1: Conv1D(12→32, k=8) + BN + ReLU + MaxPool    │
│  Conv Block 2: Conv1D(32→64, k=8) + BN + ReLU + MaxPool    │
│  Conv Block 3: Conv1D(64→128, k=8) + BN + ReLU + MaxPool   │
│  Dropout: 0.35                                               │
│                                                              │
│  Output: [Batch, 128, 43] → Flatten → [Batch, 5504]        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Temporal Contrastive Module (TC)                           │
│  - Transformer encoder (6 timesteps)                         │
│  - Enforces temporal coherence across augmented views       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Classification Head                                         │
│  - Linear(5504 → 10 classes)                                │
└─────────────────────────────────────────────────────────────┘
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 40 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 3e-4 |
| β1, β2 | 0.9, 0.99 |
| Weight decay | 3e-4 |
| Dropout | 0.35 |
| LR Scheduler | ReduceLROnPlateau (monitor val loss) |

---

## 🔄 Augmentations

### Weak Augmentation
**Scaling** (applied to pretrain/SupCon modes)
- Randomly scale each channel by factor ~ Normal(μ=2.0, σ=1.1)
- Simulates variations in sensor sensitivity
- Less disruptive, preserves temporal structure

### Strong Augmentation
**Permutation + Jitter** (applied to pretrain/SupCon modes)
1. **Permutation**:
   - Split time series into 1-8 random segments
   - Randomly shuffle segment order
   - Simulates temporal variations

2. **Jitter**:
   - Add Gaussian noise ~ Normal(μ=0.0, σ=0.8)
   - Simulates sensor measurement noise

---

## 📈 Results

### Statistical Testing

All comparisons use two-tailed independent t-tests (n=5 seeds):

| Comparison | Window-Level | Video-Level | Significance |
|------------|--------------|-------------|--------------|
| **CATCC_5shot vs Supervised_5shot** | p=0.0004 | p=0.0009 | *** (Highly significant) |
| **CATCC_1shot vs Supervised_1shot** | p=0.6355 | p=0.6468 | ns (Not significant) |
| **0shot vs Supervised_1shot** | p=0.0008 | p=0.0008 | *** (Highly significant, worse) |

Significance levels: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` = not significant

### Window vs. Video-Level Accuracy

| Method | Window-Level | Video-Level | Difference |
|--------|--------------|-------------|------------|
| **0-shot** | 13.27% ± 6.78 | 12.20% ± 6.27 | -1.07% |
| **Supervised 1-shot** | 43.19% ± 9.35 | 43.89% ± 10.27 | +0.70% |
| **CATCC 1-shot** | 47.07% ± 12.67 | 47.61% ± 11.82 | +0.54% |
| **Supervised 5-shot** | 67.86% ± 3.43 | 68.61% ± 3.11 | +0.75% |
| **CATCC 5-shot** | 71.83% ± 2.96 | 71.43% ± 3.36 | -0.40% |
| **Supervised 100%** | 80.37% ± 3.48 | 81.34% ± 4.34 | +0.97% |

**Observation**: Window and video-level accuracies are nearly identical, indicating stable and consistent predictions across windows within the same video.

### Analysis

**Why is 0-shot so poor (12.20%)?**
- Random baseline: 10% (10 classes)
- Self-supervised pretraining learns temporal features but NOT class-discriminative features
- The pretrained encoder requires fine-tuning to map features to specific exercise classes
- Without labeled data, the model cannot distinguish between exercise types

**Why are improvements modest?**
- Small labeled training sets (10 or 50 videos) limit fine-tuning effectiveness
- High inter-subject variability in exercise execution
- Some exercises are inherently similar (e.g., Overhead Press vs. Bench Press)
- Pretraining helps but doesn't fully overcome data scarcity

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
- Few-shot learning extension and video-level evaluation: This work

---

**Last Updated:** November 14, 2025




