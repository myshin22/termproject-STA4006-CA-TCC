"""
NEW: Prepare Exercise IMU dataset with VIDEO-LEVEL few-shot learning.

This is a NEW file. The original prepare_exercise_data.py remains unchanged.

Key differences from prepare_exercise_data.py:
1. 1-shot = 1 video per exercise class (not 1% of windows)
2. 5-shot = 5 videos per exercise class from DIFFERENT subjects
3. Ensures balanced sampling across exercises and subjects
4. Reports video-level statistics clearly

This script:
1. Loads the filtered CSV
2. Segments data by video/rep
3. Creates sliding windows (5 sec window, 2 sec stride @ 66Hz)
4. Splits into train/val/test
5. Creates VIDEO-LEVEL few-shot splits (1-shot, 5-shot)
"""

import pandas as pd
import numpy as np
import torch
import os
import json
from sklearn.model_selection import train_test_split
from collections import Counter

# Configuration
INPUT_FILE = '/home/user1/MY_project/proj_ca_tcc/data/merged_dataset_imu_filtered.csv'
OUTPUT_DIR = '/home/user1/MY_project/proj_ca_tcc/CA-TCC/data/ExerciseIMU'
SAMPLING_RATE = 66  # Hz
WINDOW_SIZE = 5 * SAMPLING_RATE  # 5 seconds = 330 frames
STRIDE = 2 * SAMPLING_RATE       # 2 seconds = 132 frames
SEED = 0

# Set random seed for reproducible few-shot sampling
np.random.seed(SEED)

# Sensor columns (12 channels)
SENSOR_COLS = [
    'right_ax', 'right_ay', 'right_az',
    'right_gx', 'right_gy', 'right_gz',
    'left_ax', 'left_ay', 'left_az',
    'left_gx', 'left_gy', 'left_gz'
]

print("="*70)
print("Exercise IMU Dataset Preparation for CA-TCC (VIDEO-LEVEL Few-Shot)")
print("="*70)
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Sampling rate: {SAMPLING_RATE} Hz")
print(f"Window: {WINDOW_SIZE} frames (5 sec)")
print(f"Stride: {STRIDE} frames (2 sec)")
print()

# Load data
print("[1/7] Loading filtered data...")
df = pd.read_csv(INPUT_FILE)
print(f"  Shape: {df.shape}")
print(f"  Exercises: {sorted(df['exercise'].unique())}")
print(f"  Classes: {df['exercise'].nunique()}")
print(f"  Subjects: {sorted(df['subject_id'].unique())}")
print(f"  Total subjects: {df['subject_id'].nunique()}")
print()

# Create label encoding
print("[2/7] Creating label encoding...")
label_encoder = {ex: idx for idx, ex in enumerate(sorted(df['exercise'].unique()))}
label_decoder = {idx: ex for ex, idx in label_encoder.items()}
df['label'] = df['exercise'].map(label_encoder)

print(f"  Label mapping:")
for ex, idx in sorted(label_encoder.items()):
    print(f"    {idx}: {ex}")
print()

# Save label mapping
label_mapping = {
    'encoder': label_encoder,
    'decoder': {str(k): v for k, v in label_decoder.items()}
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'label_mapping.json'), 'w') as f:
    json.dump(label_mapping, f, indent=2)

# Segment data by video (each video is one trial)
print("[3/7] Segmenting data with sliding window...")
print(f"  Grouping by: video_name (each video = one trial)")

segments = []
labels = []
subjects = []
video_ids = []  # Track which video each window belongs to
exercises = []  # Track exercise for each window

# Group by video_name only (video is the key)
grouped = df.groupby(['video_name'])
total_groups = len(grouped)
skipped = 0

# Create video_name to ID mapping
all_video_names = [v[0] if isinstance(v, tuple) else v for v in grouped.groups.keys()]
video_name_to_id = {video: idx for idx, video in enumerate(sorted(all_video_names))}

for i, (video, group) in enumerate(grouped):
    if (i + 1) % 100 == 0:
        print(f"  Processing {i+1}/{total_groups} videos...")

    # Get subject, label, and exercise for this video (should be consistent within video)
    subject = group['subject_id'].iloc[0]
    label = group['label'].iloc[0]
    exercise = group['exercise'].iloc[0]
    video_name = video[0] if isinstance(video, tuple) else video
    video_id = video_name_to_id[video_name]

    # Sort by timestamp to ensure correct order
    group = group.sort_values('TimeStamp')

    # Extract sensor data for entire video
    sensor_data = group[SENSOR_COLS].values  # Shape: [T, 12]

    # Skip if video is too short
    if len(sensor_data) < WINDOW_SIZE:
        skipped += 1
        continue

    # Apply sliding window
    num_windows = (len(sensor_data) - WINDOW_SIZE) // STRIDE + 1

    for w in range(num_windows):
        start = w * STRIDE
        end = start + WINDOW_SIZE
        window = sensor_data[start:end]  # [330, 12]

        segments.append(window.T)  # Transpose to [12, 330]
        labels.append(label)
        subjects.append(subject)
        video_ids.append(video_id)
        exercises.append(exercise)

print(f"  Created {len(segments)} windows from {total_groups - skipped} videos")
print(f"  Skipped {skipped} videos (too short)")
print()

# Convert to numpy arrays
X = np.array(segments, dtype=np.float32)  # [N, 12, 330]
y = np.array(labels, dtype=np.int64)
subjects = np.array(subjects)
video_ids = np.array(video_ids, dtype=np.int64)
exercises = np.array(exercises)

print(f"  Final shape: X={X.shape}, y={y.shape}")
print(f"  Video IDs: {len(np.unique(video_ids))} unique videos")
print()

# Split into train/test by subject (80/20)
print("[4/7] Splitting into train/test by subject...")
unique_subjects = np.unique(subjects)
train_subjects, test_subjects = train_test_split(
    unique_subjects, test_size=0.2, random_state=SEED
)

print(f"  Total subjects: {len(unique_subjects)}")
print(f"  Train subjects: {len(train_subjects)}")
print(f"  Test subjects:  {len(test_subjects)}")

train_mask = np.isin(subjects, train_subjects)
test_mask = np.isin(subjects, test_subjects)

X_train, y_train, video_train, subject_train, exercise_train = \
    X[train_mask], y[train_mask], video_ids[train_mask], subjects[train_mask], exercises[train_mask]
X_test, y_test, video_test, subject_test, exercise_test = \
    X[test_mask], y[test_mask], video_ids[test_mask], subjects[test_mask], exercises[test_mask]

# Print window-level statistics
print(f"\n  Train: {len(X_train)} windows")
print(f"    Distribution: {dict(sorted(Counter(y_train).items()))}")
print(f"  Test:  {len(X_test)} windows")
print(f"    Distribution: {dict(sorted(Counter(y_test).items()))}")

# Print VIDEO-level statistics
unique_train_videos = np.unique(video_train)
unique_test_videos = np.unique(video_test)

# Get video-to-label mapping for train
train_video_labels = {}
for vid in unique_train_videos:
    vid_mask = video_train == vid
    train_video_labels[vid] = y_train[vid_mask][0]

train_video_label_counts = Counter(train_video_labels.values())

# Get video-to-label mapping for test
test_video_labels = {}
for vid in unique_test_videos:
    vid_mask = video_test == vid
    test_video_labels[vid] = y_test[vid_mask][0]

test_video_label_counts = Counter(test_video_labels.values())

print(f"\n  Train VIDEOS: {len(unique_train_videos)} videos")
print(f"    Videos per class: {dict(sorted(train_video_label_counts.items()))}")
print(f"  Test VIDEOS:  {len(unique_test_videos)} videos")
print(f"    Videos per class: {dict(sorted(test_video_label_counts.items()))}")
print()

# Normalize (z-score per channel)
print("[5/7] Normalizing data...")
mean = X_train.mean(axis=(0, 2), keepdims=True)  # [1, 12, 1]
std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
print(f"  Normalized using training statistics")
print()

# Save full datasets
print("[6/7] Saving full train/test datasets...")

def save_dataset(X, y, vid_ids, path):
    data = {
        'samples': torch.FloatTensor(X),
        'labels': torch.LongTensor(y),
        'video_ids': torch.LongTensor(vid_ids)
    }
    torch.save(data, path)
    print(f"  Saved: {path}")

save_dataset(X_train, y_train, video_train, os.path.join(OUTPUT_DIR, 'train.pt'))
save_dataset(X_test, y_test, video_test, os.path.join(OUTPUT_DIR, 'test.pt'))

# Also create val.pt (same as test for now, or could be a separate split)
save_dataset(X_test, y_test, video_test, os.path.join(OUTPUT_DIR, 'val.pt'))

# ============================================================================
# [7/7] Create VIDEO-LEVEL few-shot splits
# ============================================================================
print(f"\n[7/7] Creating VIDEO-LEVEL few-shot splits...")
print("="*70)

# Build video-to-subject and video-to-label mappings
video_to_label = {}
video_to_subject = {}

for vid in unique_train_videos:
    vid_mask = video_train == vid
    label = y_train[vid_mask][0]
    subject = subject_train[vid_mask][0]
    video_to_label[vid] = label
    video_to_subject[vid] = subject

# Group videos by class
videos_by_class = {}
for vid, label in video_to_label.items():
    if label not in videos_by_class:
        videos_by_class[label] = []
    videos_by_class[label].append(vid)

print(f"Total training videos: {len(unique_train_videos)}")
print(f"Videos per class:")
for cls in sorted(videos_by_class.keys()):
    class_name = label_decoder[int(cls)]
    n_vids = len(videos_by_class[cls])

    # Count unique subjects for this class
    class_subjects = set([video_to_subject[v] for v in videos_by_class[cls]])
    n_subjects = len(class_subjects)

    print(f"  Class {cls} ({class_name}): {n_vids} videos from {n_subjects} subjects")

print()

# Define few-shot splits: 1-shot and 5-shot
for n_videos_per_class in [1, 5]:
    print("="*70)
    print(f"Creating {n_videos_per_class}-shot split (VIDEO-LEVEL)")
    print("="*70)

    selected_videos = []

    for cls in sorted(videos_by_class.keys()):
        class_name = label_decoder[int(cls)]
        class_videos = videos_by_class[cls]

        if n_videos_per_class == 1:
            # For 1-shot: randomly select 1 video from this class
            sampled_vid = np.random.choice(class_videos, 1)[0]
            selected_videos.append(sampled_vid)

            sampled_subject = video_to_subject[sampled_vid]
            print(f"  Class {cls} ({class_name}): selected 1 video from subject {sampled_subject}")

        elif n_videos_per_class == 5:
            # For 5-shot: try to select videos from DIFFERENT subjects
            videos_by_subject = {}
            for vid in class_videos:
                subj = video_to_subject[vid]
                if subj not in videos_by_subject:
                    videos_by_subject[subj] = []
                videos_by_subject[subj].append(vid)

            available_subjects = list(videos_by_subject.keys())

            if len(available_subjects) >= n_videos_per_class:
                # Enough subjects: sample one video from each of 5 different subjects
                selected_subjects = np.random.choice(available_subjects, n_videos_per_class, replace=False)
                sampled_vids = []
                for subj in selected_subjects:
                    # Randomly select one video from this subject
                    vid = np.random.choice(videos_by_subject[subj])
                    sampled_vids.append(vid)

                selected_videos.extend(sampled_vids)
                n_unique_subjects = len(set([video_to_subject[v] for v in sampled_vids]))
                print(f"  Class {cls} ({class_name}): selected {len(sampled_vids)} videos from {n_unique_subjects} different subjects")
            else:
                # Not enough subjects: sample randomly with warning
                print(f"  ⚠ Warning: Class {cls} ({class_name}) only has {len(available_subjects)} subjects")
                sampled_vids = np.random.choice(class_videos, min(n_videos_per_class, len(class_videos)), replace=False).tolist()
                selected_videos.extend(sampled_vids)
                n_unique_subjects = len(set([video_to_subject[v] for v in sampled_vids]))
                print(f"  Class {cls} ({class_name}): selected {len(sampled_vids)} videos from {n_unique_subjects} subjects (limited)")

    # Get all windows belonging to selected videos
    selected_video_mask = np.isin(video_train, selected_videos)

    X_fewshot = X_train[selected_video_mask]
    y_fewshot = y_train[selected_video_mask]
    video_fewshot = video_train[selected_video_mask]

    # Statistics
    unique_selected_videos = np.unique(video_fewshot)
    fewshot_video_labels = {}
    for vid in unique_selected_videos:
        vid_mask = video_fewshot == vid
        fewshot_video_labels[vid] = y_fewshot[vid_mask][0]

    fewshot_video_counts = Counter(fewshot_video_labels.values())

    print(f"\nFinal {n_videos_per_class}-shot statistics:")
    print(f"  Total windows: {len(X_fewshot)}")
    print(f"  Total videos: {len(unique_selected_videos)}")
    print(f"  Videos per class: {dict(sorted(fewshot_video_counts.items()))}")
    print(f"  Expected: {n_videos_per_class} videos × {len(videos_by_class)} classes = {n_videos_per_class * len(videos_by_class)} videos")

    # Save few-shot split
    output_name = f"train_{n_videos_per_class}shot.pt"
    save_dataset(X_fewshot, y_fewshot, video_fewshot, os.path.join(OUTPUT_DIR, output_name))
    print()

print("="*70)
print("✓ All datasets created successfully!")
print("="*70)
print()
print("Summary of created files:")
print(f"  {OUTPUT_DIR}/train.pt         - Full training set")
print(f"  {OUTPUT_DIR}/val.pt           - Validation set")
print(f"  {OUTPUT_DIR}/test.pt          - Test set")
print(f"  {OUTPUT_DIR}/train_1shot.pt   - 1 video per class")
print(f"  {OUTPUT_DIR}/train_5shot.pt   - 5 videos per class (from different subjects)")
print(f"  {OUTPUT_DIR}/label_mapping.json - Label encoding/decoding")
print()
print("Next steps:")
print("  1. Run experiments:")
print("     ./run_experiments_video.sh ExerciseIMU 0 4")
print()
