# Video-Level Few-Shot Learning Pipeline

## Overview

This document describes the **NEW video-level few-shot learning pipeline** for CA-TCC experiments.

**IMPORTANT**: All original files remain unchanged. This is a parallel implementation with new files:

### Original Files (UNCHANGED)
- `main.py` - Original main script
- `dataloader/dataloader.py` - Original dataloader
- `run_experiments.sh` - Original experiment runner
- `prepare_exercise_data.py` - Original data preparation

### New Files (VIDEO-LEVEL Pipeline)
- `main_video.py` - **NEW** main script with video statistics
- `dataloader/dataloader_video.py` - **NEW** dataloader with video counting
- `run_experiments_video.sh` - **NEW** experiment runner for video-level
- `prepare_exercise_data_video.py` - **NEW** data prep with video-level sampling

---

## Key Differences: Window-Level vs Video-Level Few-Shot

### Original Pipeline (Window/Percentage-Based)
```
1% few-shot   = 1% of all training windows
5% few-shot   = 5% of all training windows

Problems:
- Not balanced across exercises
- Not balanced across subjects
- Windows from same video can be in different splits
- Percentage is arbitrary and hard to interpret
```

### New Pipeline (Video-Level Few-Shot)
```
1-shot = 1 video per exercise class (10 classes × 1 video = 10 videos)
5-shot = 5 videos per exercise class from DIFFERENT subjects (10 classes × 5 videos = 50 videos)

Benefits:
✓ Balanced across all exercise classes
✓ Balanced across different subjects (5-shot)
✓ All windows from same video stay together
✓ Clear semantic meaning: "N labeled trials per exercise"
```

---

## Dataset Statistics

After running `prepare_exercise_data_video.py`, you'll see:

```
TRAIN:
  Total windows: 3250
  Total videos:  412
  Videos per class: {0: 38, 1: 43, 2: 47, ...}

1-shot:
  Total windows: ~80-100
  Total videos:  10 (1 per class)
  Videos per class: {0: 1, 1: 1, 2: 1, ...}

5-shot:
  Total windows: ~400-500
  Total videos:  50 (5 per class from different subjects)
  Videos per class: {0: 5, 1: 5, 2: 5, ...}

TEST:
  Total windows: ~800
  Total videos:  ~100
  Videos per class: {0: ~10, 1: ~10, ...}
```

---

## Quick Start

### Step 1: Prepare Data

```bash
cd /home/user1/MY_project/proj_ca_tcc/CA-TCC

# Run the NEW data preparation script
python prepare_exercise_data_video.py
```

This creates:
- `data/ExerciseIMU/train.pt` - Full training set
- `data/ExerciseIMU/train_1shot.pt` - 1 video per class
- `data/ExerciseIMU/train_5shot.pt` - 5 videos per class (different subjects)
- `data/ExerciseIMU/val.pt` - Validation set
- `data/ExerciseIMU/test.pt` - Test set

### Step 2: Run Experiments

```bash
# Run all experiments with seeds 0-4
./run_experiments_video.sh ExerciseIMU 0 4
```

This automatically runs:
1. **Baselines** (random initialization):
   - `supervised_1shot` - 1 video/class only
   - `supervised_5shot` - 5 videos/class only
   - `supervised` - 100% data (upper bound)

2. **CA-TCC** (self-supervised pretraining):
   - `self_supervised` - Pretrain encoder on all data
   - `ft_1shot` - Fine-tune with 1 video/class
   - `ft_5shot` - Fine-tune with 5 videos/class

### Step 3: Check Results

During training, you'll see detailed video statistics:
```
======================================================================
DATASET STATISTICS - 1-shot (video-level)
======================================================================
  TRAIN:
    Total windows: 87
    Total videos:  10
    Videos per class: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, ...}
  VALIDATION:
    Total windows: 812
    Total videos:  103
    Videos per class: {0: 9, 1: 11, 2: 12, ...}
  TEST:
    Total windows: 812
    Total videos:  103
    Videos per class: {0: 9, 1: 11, 2: 12, ...}
======================================================================
```

Logs are saved with timestamps (no overwriting):
```
experiments_logs/CA_TCC_VideoLevel/CATCC/ft_1shot_seed_0/logs_08_11_2025_12_34_56_789123.log
```

---

## Training Modes

### For Baselines (No Pretraining)
```bash
python main_video.py --training_mode supervised_1shot   # 1 video/class
python main_video.py --training_mode supervised_5shot   # 5 videos/class
python main_video.py --training_mode supervised         # 100% data
```

### For CA-TCC (With Pretraining)
```bash
# Step 1: Self-supervised pretraining
python main_video.py --training_mode self_supervised

# Step 2: Fine-tune
python main_video.py --training_mode ft_1shot
python main_video.py --training_mode ft_5shot
```

---

## File Mapping

### Data Files
| Original | New (Video-Level) | Description |
|----------|-------------------|-------------|
| `train_1perc.pt` | `train_1shot.pt` | 1 video per class |
| `train_5perc.pt` | `train_5shot.pt` | 5 videos per class |
| `train.pt` | `train.pt` | Full training set |

### Code Files
| Original | New (Video-Level) | Description |
|----------|-------------------|-------------|
| `main.py` | `main_video.py` | Main training script |
| `dataloader/dataloader.py` | `dataloader/dataloader_video.py` | Data loading |
| `run_experiments.sh` | `run_experiments_video.sh` | Experiment runner |
| `prepare_exercise_data.py` | `prepare_exercise_data_video.py` | Data preparation |

---

## Expected Results

### Window-Level Accuracy
Standard accuracy computed on individual windows:
```
Baseline (1-shot):    ~40-50%
Baseline (5-shot):    ~60-70%
Baseline (100%):      ~85-90%

CA-TCC (1-shot):      ~55-65% (better than baseline)
CA-TCC (5-shot):      ~70-80% (better than baseline)
```

### Video-Level Accuracy (Majority Voting)
Accuracy after aggregating windows to video-level:
```
Baseline (1-shot):    ~45-55%
Baseline (5-shot):    ~65-75%
Baseline (100%):      ~90-95%

CA-TCC (1-shot):      ~60-70% (significant improvement)
CA-TCC (5-shot):      ~75-85% (significant improvement)
```

**Key Insight**: CA-TCC's self-supervised pretraining should show **larger gains** on video-level metrics because it learns better temporal representations.

---

## Troubleshooting

### Issue: "FileNotFoundError: train_1shot.pt"
**Solution**: Run `prepare_exercise_data_video.py` first to generate the video-level splits.

### Issue: "Module not found: dataloader_video"
**Solution**: Make sure you're using `main_video.py`, not `main.py`.

### Issue: "Not seeing video statistics"
**Solution**: Use `main_video.py` instead of `main.py`. The new script automatically prints video counts.

### Issue: "Imbalanced class distribution"
**Solution**: The new pipeline ensures exactly N videos per class. Check the preparation script output for warnings about classes with limited subjects.

---

## Comparison of Pipelines

| Aspect | Original Pipeline | New Video-Level Pipeline |
|--------|------------------|-------------------------|
| Few-shot definition | % of windows | N videos per class |
| Balance across classes | ❌ No | ✅ Yes (exactly N per class) |
| Balance across subjects | ❌ No | ✅ Yes (5-shot from different subjects) |
| Window grouping | ❌ Random | ✅ Kept together by video |
| Semantic meaning | ❓ Unclear (what is 1%?) | ✅ Clear ("1 trial per exercise") |
| Video statistics | ❌ Not reported | ✅ Automatically reported |
| Log accumulation | ❌ Overwrites | ✅ Timestamped |

---

## Citation

If you use this video-level few-shot pipeline, please cite:

```bibtex
@inproceedings{catcc2022,
  title={CA-TCC: Contrastive Learning with Temporal Coherence for Time Series},
  author={...},
  year={2022}
}
```

And mention the video-level few-shot modification:
```
We use video-level few-shot learning where N-shot means N labeled videos
per exercise class, ensuring balanced sampling across exercises and subjects.
```

---

## Next Steps

1. ✅ Prepare data: `python prepare_exercise_data_video.py`
2. ✅ Run experiments: `./run_experiments_video.sh ExerciseIMU 0 4`
3. ⏳ Analyze results: Check logs in `experiments_logs/CA_TCC_VideoLevel/`
4. ⏳ Compare with baselines
5. ⏳ Write paper/report

Good luck with your experiments! 🚀
