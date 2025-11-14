"""
Prepare Exercise IMU dataset for CA-TCC with PROPER few-shot learning.

Updated strategy (same as SSL-wearables):
1. Subject-level split: 5 train / 4 val / 4 test (no data leakage)
2. Few-shot sampling:
   - 1-shot: 1 subject, 1 video per class
   - 5-shot: 5 subjects, 1 video per subject per class
3. Window size: 5 seconds @ 66Hz (330 frames) - CA-TCC standard
4. All 12 channels (accelerometer + gyroscope)

This ensures:
- Complete subject separation (no data leakage)
- Diversity in few-shot learning (multiple subjects)
- Sufficient val/test data for reliable evaluation
- Comparable with SSL-wearables experiments
"""

import pandas as pd
import numpy as np
import torch
import os
import json
import argparse
from collections import Counter, defaultdict

# Parse arguments
parser = argparse.ArgumentParser(description='Prepare Exercise IMU dataset for CA-TCC few-shot learning')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible splits')
parser.add_argument('--input_file', type=str,
                    default='/home/user1/MY_project/proj_ca_tcc/data/merged_dataset_imu_filtered.csv',
                    help='Path to input CSV file')
parser.add_argument('--output_dir', type=str,
                    default='/home/user1/MY_project/proj_ca_tcc/CA-TCC/data/ExerciseIMU',
                    help='Output directory for processed data')
args = parser.parse_args()

# Configuration
INPUT_FILE = args.input_file
OUTPUT_DIR = args.output_dir
SAMPLING_RATE = 66  # Hz
WINDOW_SIZE = 5 * SAMPLING_RATE  # 5 seconds = 330 frames (CA-TCC standard)
STRIDE = 2 * SAMPLING_RATE       # 2 seconds = 132 frames
SEED = args.seed

# Set random seed
np.random.seed(SEED)

# Sensor columns (12 channels - accelerometer + gyroscope)
SENSOR_COLS = [
    'right_ax', 'right_ay', 'right_az',
    'right_gx', 'right_gy', 'right_gz',
    'left_ax', 'left_ay', 'left_az',
    'left_gx', 'left_gy', 'left_gz'
]

print("=" * 80)
print("CA-TCC Exercise IMU Dataset Preparation for Few-Shot Learning")
print("=" * 80)
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Seed:   {SEED}")
print(f"Strategy: 1-shot=1 subject, 5-shot=5 subjects")
print(f"Window:  {WINDOW_SIZE} frames (5 sec) - CA-TCC standard")
print()

# Load data
print("[1/9] Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"  Shape: {df.shape}")
print(f"  Subjects: {sorted(df['subject_id'].unique())}")
print(f"  Total subjects: {df['subject_id'].nunique()}")
print(f"  Exercises: {sorted(df['exercise'].unique())}")
print(f"  Total classes: {df['exercise'].nunique()}")
print()

# Create label encoding
print("[2/9] Creating label encoding...")
label_encoder = {ex: idx for idx, ex in enumerate(sorted(df['exercise'].unique()))}
label_decoder = {idx: ex for ex, idx in label_encoder.items()}
df['label'] = df['exercise'].map(label_encoder)

print(f"  Label mapping:")
for ex, idx in sorted(label_encoder.items()):
    print(f"    {idx}: {ex}")
print()

num_classes = len(label_encoder)

# Save label mapping
label_mapping = {
    'encoder': label_encoder,
    'decoder': {str(k): v for k, v in label_decoder.items()}
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'label_mapping.json'), 'w') as f:
    json.dump(label_mapping, f, indent=2)

# Segment data with sliding window
print("[3/9] Segmenting data with sliding window...")
segments = []
labels = []
subjects = []
video_ids = []

# Group by video_name
grouped = df.groupby(['video_name'])
total_groups = len(grouped)
skipped = 0

# Create video_name to ID mapping
all_video_names = [v[0] if isinstance(v, tuple) else v for v in grouped.groups.keys()]
video_name_to_id = {video: idx for idx, video in enumerate(sorted(all_video_names))}

for i, (video, group) in enumerate(grouped):
    if (i + 1) % 100 == 0:
        print(f"  Processing {i+1}/{total_groups} videos...")

    subject = group['subject_id'].iloc[0]
    label = group['label'].iloc[0]
    video_name = video[0] if isinstance(video, tuple) else video
    video_id = video_name_to_id[video_name]

    # Sort by timestamp
    group = group.sort_values('TimeStamp')

    # Extract sensor data (12 channels)
    sensor_data = group[SENSOR_COLS].values  # [T, 12]

    # Skip if too short
    if len(sensor_data) < WINDOW_SIZE:
        skipped += 1
        continue

    # Sliding window
    num_windows = (len(sensor_data) - WINDOW_SIZE) // STRIDE + 1

    for w in range(num_windows):
        start = w * STRIDE
        end = start + WINDOW_SIZE
        window = sensor_data[start:end]  # [330, 12]

        segments.append(window.T)  # Transpose to [12, 330]
        labels.append(label)
        subjects.append(subject)
        video_ids.append(video_id)

print(f"  Created {len(segments)} windows from {total_groups - skipped} videos")
print(f"  Skipped {skipped} videos (too short)")
print()

# Convert to numpy arrays
X = np.array(segments, dtype=np.float32)  # [N, 12, 330]
y = np.array(labels, dtype=np.int64)
subjects = np.array(subjects)
video_ids = np.array(video_ids, dtype=np.int64)

print(f"  Final shape: X={X.shape}, y={y.shape}")
print(f"  Video IDs: {len(np.unique(video_ids))} unique videos")
print()

# Split subjects into train/val/test
print("[4/9] Splitting subjects into train/val/test (5/4/4)...")
unique_subjects = sorted(np.unique(subjects))
n_subjects = len(unique_subjects)

print(f"  Total subjects: {n_subjects}")

if n_subjects < 13:
    print(f"  [Warning] Only {n_subjects} subjects available. Adjusting split...")
    n_train = max(5, n_subjects // 2)
    n_val = (n_subjects - n_train) // 2
    n_test = n_subjects - n_train - n_val
else:
    n_train = 5
    n_val = 4
    n_test = 4

# Shuffle and split subjects
shuffled_subjects = unique_subjects.copy()
np.random.shuffle(shuffled_subjects)

train_subjects = shuffled_subjects[:n_train]
val_subjects = shuffled_subjects[n_train:n_train + n_val]
test_subjects = shuffled_subjects[n_train + n_val:n_train + n_val + n_test]

print(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
print(f"  Val subjects ({len(val_subjects)}):   {sorted(val_subjects)}")
print(f"  Test subjects ({len(test_subjects)}):  {sorted(test_subjects)}")
print()

# Create masks
train_mask = np.isin(subjects, train_subjects)
val_mask = np.isin(subjects, val_subjects)
test_mask = np.isin(subjects, test_subjects)

# Split data
X_train_full, y_train_full, video_train_full = X[train_mask], y[train_mask], video_ids[train_mask]
X_val, y_val, video_val = X[val_mask], y[val_mask], video_ids[val_mask]
X_test, y_test, video_test = X[test_mask], y[test_mask], video_ids[test_mask]

print("[5/9] Dataset statistics after split...")
print(f"  Train: {len(X_train_full)} windows, {len(np.unique(video_train_full))} videos")
print(f"  Val:   {len(X_val)} windows, {len(np.unique(video_val))} videos")
print(f"  Test:  {len(X_test)} windows, {len(np.unique(video_test))} videos")
print()

# Normalize (z-score per channel) using ONLY training data
print("[5.5/9] Normalizing data...")
mean = X_train_full.mean(axis=(0, 2), keepdims=True)  # [1, 12, 1]
std = X_train_full.std(axis=(0, 2), keepdims=True) + 1e-8

X_train_full = (X_train_full - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std
print(f"  Normalized using training statistics")
print()

# Create video-level mapping for train set
print("[6/9] Creating video-level mapping for few-shot sampling...")

# Map: video_id -> (subject, label, window_indices)
video_info = {}
for idx, vid in enumerate(video_train_full):
    if vid not in video_info:
        video_info[vid] = {
            'subject': subjects[train_mask][idx],
            'label': y_train_full[idx],
            'indices': []
        }
    video_info[vid]['indices'].append(idx)

# Group videos by (subject, label)
subject_class_videos = defaultdict(list)
for vid, info in video_info.items():
    key = (info['subject'], info['label'])
    subject_class_videos[key].append(vid)

print(f"  Total videos in train set: {len(video_info)}")
print(f"  Videos per (subject, class): (showing first 20)")
for i, ((subj, cls), vids) in enumerate(sorted(subject_class_videos.items())):
    if i < 20:
        print(f"    Subject {subj}, Class {cls}: {len(vids)} videos")
    elif i == 20:
        print(f"    ... ({len(subject_class_videos) - 20} more combinations)")
        break
print()

# Create 1-shot: 1 subject, 1 video per class
print("[7/9] Creating 1-shot split (1 subject, 1 video per class)...")

# Select 1 subject from train_subjects
selected_1shot_subject = np.random.choice(train_subjects, size=1)[0]
print(f"  Selected subject: {selected_1shot_subject}")

shot1_indices = []
shot1_videos = []

for class_id in range(num_classes):
    key = (selected_1shot_subject, class_id)
    if key in subject_class_videos and len(subject_class_videos[key]) > 0:
        # Select 1 random video
        selected_video = np.random.choice(subject_class_videos[key], size=1)[0]
        shot1_videos.append(selected_video)
        shot1_indices.extend(video_info[selected_video]['indices'])
    else:
        print(f"    [Warning] Subject {selected_1shot_subject} has no videos for class {class_id}")

X_train_1shot = X_train_full[shot1_indices]
y_train_1shot = y_train_full[shot1_indices]
video_train_1shot = video_train_full[shot1_indices]

print(f"  1-shot: {len(X_train_1shot)} windows, {len(np.unique(video_train_1shot))} videos")
print(f"  Class distribution: {dict(sorted(Counter(y_train_1shot).items()))}")
print()

# Create 5-shot: 5 subjects, 1 video per subject per class
print("[8/9] Creating 5-shot split (5 subjects, 1 video per subject per class)...")

if len(train_subjects) < 5:
    print(f"  [Warning] Only {len(train_subjects)} train subjects available, using all")
    selected_5shot_subjects = train_subjects
else:
    selected_5shot_subjects = train_subjects[:5]

print(f"  Selected subjects: {sorted(selected_5shot_subjects)}")

shot5_indices = []
shot5_videos = []

for class_id in range(num_classes):
    for subj in selected_5shot_subjects:
        key = (subj, class_id)
        if key in subject_class_videos and len(subject_class_videos[key]) > 0:
            # Select 1 random video
            selected_video = np.random.choice(subject_class_videos[key], size=1)[0]
            shot5_videos.append(selected_video)
            shot5_indices.extend(video_info[selected_video]['indices'])
        else:
            print(f"    [Warning] Subject {subj} has no videos for class {class_id}")

X_train_5shot = X_train_full[shot5_indices]
y_train_5shot = y_train_full[shot5_indices]
video_train_5shot = video_train_full[shot5_indices]

print(f"  5-shot: {len(X_train_5shot)} windows, {len(np.unique(video_train_5shot))} videos")
print(f"  Class distribution: {dict(sorted(Counter(y_train_5shot).items()))}")
print()

# Create pretrain set (train + val for self-supervised learning)
print("[9/10] Creating pretrain set (train + val)...")
X_pretrain = np.concatenate([X_train_full, X_val], axis=0)
y_pretrain = np.concatenate([y_train_full, y_val], axis=0)
video_pretrain = np.concatenate([video_train_full, video_val], axis=0)

print(f"  Pretrain (Train + Val): {len(X_pretrain)} windows, {len(np.unique(video_pretrain))} videos")
print(f"    This will be used for self-supervised pretraining")
print(f"    (Test set is NEVER used in pretraining!)")
print()

# Save all splits (PyTorch format for CA-TCC)
print("[10/10] Saving data splits...")

def save_dataset(X, y, vid_ids, path):
    """Save in CA-TCC PyTorch format"""
    data = {
        'samples': torch.FloatTensor(X),
        'labels': torch.LongTensor(y),
        'video_ids': torch.LongTensor(vid_ids)
    }
    torch.save(data, path)
    print(f"  Saved: {path}")

save_dataset(X_pretrain, y_pretrain, video_pretrain,
             os.path.join(OUTPUT_DIR, 'pretrain.pt'))
save_dataset(X_train_full, y_train_full, video_train_full,
             os.path.join(OUTPUT_DIR, 'train.pt'))
save_dataset(X_train_1shot, y_train_1shot, video_train_1shot,
             os.path.join(OUTPUT_DIR, 'train_1shot.pt'))
save_dataset(X_train_5shot, y_train_5shot, video_train_5shot,
             os.path.join(OUTPUT_DIR, 'train_5shot.pt'))
save_dataset(X_val, y_val, video_val,
             os.path.join(OUTPUT_DIR, 'val.pt'))
save_dataset(X_test, y_test, video_test,
             os.path.join(OUTPUT_DIR, 'test.pt'))

# Save split info
split_info = {
    'window_size': WINDOW_SIZE,
    'stride': STRIDE,
    'sampling_rate': SAMPLING_RATE,
    'n_channels': len(SENSOR_COLS),
    'sensor_columns': SENSOR_COLS,
    'seed': SEED,
    'n_classes': num_classes,
    'train_subjects': sorted([int(s) for s in train_subjects]),
    'val_subjects': sorted([int(s) for s in val_subjects]),
    'test_subjects': sorted([int(s) for s in test_subjects]),
    '1shot_subject': int(selected_1shot_subject),
    '5shot_subjects': sorted([int(s) for s in selected_5shot_subjects]),
    'splits': {
        'pretrain': {'windows': len(X_pretrain), 'videos': len(np.unique(video_pretrain)), 'note': 'train+val for self-supervised pretraining'},
        'train': {'windows': len(X_train_full), 'videos': len(np.unique(video_train_full))},
        'train_1shot': {'windows': len(X_train_1shot), 'videos': len(np.unique(video_train_1shot))},
        'train_5shot': {'windows': len(X_train_5shot), 'videos': len(np.unique(video_train_5shot))},
        'val': {'windows': len(X_val), 'videos': len(np.unique(video_val)), 'note': 'early stopping in fine-tuning'},
        'test': {'windows': len(X_test), 'videos': len(np.unique(video_test)), 'note': 'NEVER used in pretraining'}
    }
}

with open(os.path.join(OUTPUT_DIR, 'split_info.json'), 'w') as f:
    json.dump(split_info, f, indent=2)

print(f"  Saved: split_info.json")
print()

print("=" * 80)
print("✓ CA-TCC Dataset preparation complete!")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print()
print("Files created:")
print(f"  - pretrain.pt       : {len(X_pretrain)} windows, {len(np.unique(video_pretrain))} videos (train+val)")
print(f"  - train.pt          : {len(X_train_full)} windows, {len(np.unique(video_train_full))} videos")
print(f"  - train_1shot.pt    : {len(X_train_1shot)} windows, {len(np.unique(video_train_1shot))} videos")
print(f"  - train_5shot.pt    : {len(X_train_5shot)} windows, {len(np.unique(video_train_5shot))} videos")
print(f"  - val.pt            : {len(X_val)} windows, {len(np.unique(video_val))} videos")
print(f"  - test.pt           : {len(X_test)} windows, {len(np.unique(video_test))} videos")
print(f"  - label_mapping.json")
print(f"  - split_info.json")
print()
print("Usage:")
print(f"  - Pretraining:  Use pretrain.pt (train+val subjects, NO test!)")
print(f"  - Fine-tuning:  Use train_1shot.pt or train_5shot.pt")
print(f"  - Early stop:   Use val.pt")
print(f"  - Evaluation:   Use test.pt (ONLY for final evaluation!)")
print()
print("Subject split:")
print(f"  Train: {sorted([int(s) for s in train_subjects])}")
print(f"  Val:   {sorted([int(s) for s in val_subjects])}")
print(f"  Test:  {sorted([int(s) for s in test_subjects])}")
print()
print("Few-shot subjects:")
print(f"  1-shot: Subject {selected_1shot_subject}")
print(f"  5-shot: Subjects {sorted([int(s) for s in selected_5shot_subjects])}")
print("=" * 80)
