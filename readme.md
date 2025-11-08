# CA-TCC Video-Level Few-Shot Learning

**Clean Repository Structure - Video-Level SSH Experiments Only**

---

## 📁 Directory Structure

```
CA-TCC/
├── 📄 Core Scripts (Video-Level)
│   ├── main_video.py                      # Main training script
│   ├── prepare_exercise_data_video.py     # Data preparation
│   ├── run_experiments_video.sh           # Automated experiment runner
│   └── compare_results_video.py           # Results analysis
│
├── 📚 Documentation
│   ├── FINAL_RESEARCH_REPORT.md          # Complete research report
│   ├── QUICK_START_VIDEO_LEVEL.md        # Quick start guide
│   ├── VIDEO_LEVEL_EXPERIMENTS.md        # Technical documentation
│   └── README.md                          # Original README
│
├── 🗂️ Code Modules
│   ├── models/
│   │   ├── model.py                       # CNN Encoder
│   │   ├── TC.py                          # Temporal Contrastive module
│   │   └── attention.py                   # Transformer attention
│   ├── dataloader/
│   │   ├── dataloader_video.py           # Video-level dataloader
│   │   └── augmentations.py              # Data augmentations
│   ├── trainer/
│   │   └── trainer.py                     # Training loop
│   ├── config_files/
│   │   └── ExerciseIMU_Configs.py        # Hyperparameters
│   └── utils.py                           # Metrics, logging
│
├── 💾 Data
│   └── data/ExerciseIMU/
│       ├── train.pt                       # Full training (373 videos)
│       ├── train_1shot.pt                 # 1-shot (10 videos)
│       ├── train_5shot.pt                 # 5-shot (50 videos)
│       ├── val.pt                         # Validation (125 videos)
│       ├── test.pt                        # Test (125 videos)
│       └── label_mapping.json             # Exercise labels
│
├── 📊 Results
│   └── experiments_logs/CA_TCC_VideoLevel/
│       ├── Baseline_1shot/                # 1-shot baseline
│       ├── Baseline_5shot/                # 5-shot baseline
│       ├── Baseline_100p/                 # 100% baseline
│       └── CATCC/                         # CA-TCC results
│
├── 🗄️ Backup (Old Code)
│   ├── _old_code/                         # Previous scripts
│   └── _old_experiments/                  # Previous results
│
└── 🐳 Docker (Optional)
    ├── Dockerfile
    ├── docker-compose.yml
    └── docker-run.sh
```

---

## 🚀 Quick Start

### 1. Prepare Data
```bash
python prepare_exercise_data_video.py
```

### 2. Run All Experiments
```bash
# Run with seeds 0-4 (5 repetitions)
bash run_experiments_video.sh ExerciseIMU 0 4
```

### 3. Analyze Results
```bash
python compare_results_video.py
```

---

## 📝 File Descriptions

### Core Scripts

| File | Purpose |
|------|---------|
| `main_video.py` | Training script with video statistics reporting |
| `prepare_exercise_data_video.py` | Create video-level few-shot splits |
| `run_experiments_video.sh` | Automated experiment runner (5 methods × 5 seeds) |
| `compare_results_video.py` | Statistical analysis of results |

### Key Features

**Video-Level Few-Shot:**
- 1-shot = 1 video per exercise class (10 videos total)
- 5-shot = 5 videos per class from different subjects (50 videos)
- Perfectly balanced across all 10 exercise classes

**Automatic Statistics:**
```
TRAIN: 70 windows, 10 videos
VALIDATION: 904 windows, 125 videos
TEST: 904 windows, 125 videos
```

---

## 📊 Expected Results

| Method | 1-shot | 5-shot |
|--------|--------|--------|
| Baseline | 56.48% | 75.68% |
| **CA-TCC** | 54.40% | **88.96%** ⭐ |
| **Improvement** | -2.08% | **+13.28%** |

---

## 🔧 Training Modes

### Baselines (No Pretraining)
```bash
python main_video.py --training_mode supervised_1shot
python main_video.py --training_mode supervised_5shot
python main_video.py --training_mode supervised
```

### CA-TCC (With Self-Supervised Pretraining)
```bash
# Step 1: Pretrain
python main_video.py --training_mode self_supervised

# Step 2: Fine-tune
python main_video.py --training_mode ft_1shot
python main_video.py --training_mode ft_5shot
```

---

## 📖 Documentation

- **FINAL_RESEARCH_REPORT.md** - Complete research report with:
  - Model architecture
  - Dataset description
  - Data split methodology
  - Experimental design
  - Results and analysis

- **QUICK_START_VIDEO_LEVEL.md** - Quick reference guide

- **VIDEO_LEVEL_EXPERIMENTS.md** - Technical details

---

## 🗄️ Backup Files

All old/deprecated files moved to:
- `_old_code/` - Previous scripts
- `_old_experiments/` - Previous experiment results

You can safely delete these directories if not needed:
```bash
rm -rf _old_code _old_experiments
```

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- Python 3.8+
- PyTorch 1.10+
- NumPy, Pandas, scikit-learn

---

## 💡 Tips

**Quick test (1 seed only):**
```bash
bash run_experiments_video.sh ExerciseIMU 0 0
```

**Check specific results:**
```bash
# Best performance: CA-TCC 5-shot
cd experiments_logs/CA_TCC_VideoLevel/CATCC/ft_5shot_seed_0
cat *_TRIAL_classification_report.csv
```

**Clean old results:**
```bash
rm -rf experiments_logs/CA_TCC_VideoLevel
```

---

## 🎯 Key Differences from Old Code

| Aspect | Old Pipeline | New Video-Level |
|--------|-------------|-----------------|
| Few-shot definition | % of windows | N videos/class |
| Script | `main.py` | `main_video.py` |
| Dataloader | `dataloader.py` | `dataloader_video.py` |
| Experiment runner | `run_experiments.sh` | `run_experiments_video.sh` |
| Statistics | Window counts only | **Window + Video counts** |
| Naming | `1p`, `5p` | `1shot`, `5shot` |

---

## ✅ What's Included

✅ Video-level few-shot learning
✅ Balanced sampling across classes
✅ Subject diversity (5-shot from different subjects)
✅ Automatic video statistics reporting
✅ Statistical significance testing
✅ Complete research report

---

## 📧 Contact

For questions about video-level experiments, see documentation files.

For original CA-TCC paper, see README.md.

---

**Last Updated:** November 8, 2025
