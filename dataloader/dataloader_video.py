"""
Enhanced dataloader that reports video-level statistics.

This is a NEW file that extends the original dataloader.py functionality.
The original dataloader.py remains unchanged.

Key differences:
1. Reports number of unique videos in train/val/test splits
2. Shows video distribution per class
3. Designed for video-level few-shot learning
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter

from .augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        # Store video_ids if available
        self.video_ids = dataset.get("video_ids", None)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised" or training_mode == "SupCon":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def print_video_statistics(dataset_dict, split_name):
    """
    Print video-level statistics for a dataset split.

    Args:
        dataset_dict: Dictionary containing 'samples', 'labels', 'video_ids'
        split_name: Name of the split (e.g., 'Train', 'Val', 'Test')
    """
    if 'video_ids' not in dataset_dict:
        print(f"  {split_name}: No video_ids found, showing window-level stats only")
        print(f"    Total windows: {len(dataset_dict['labels'])}")
        return

    video_ids = dataset_dict['video_ids'].numpy() if torch.is_tensor(dataset_dict['video_ids']) else dataset_dict['video_ids']
    labels = dataset_dict['labels'].numpy() if torch.is_tensor(dataset_dict['labels']) else dataset_dict['labels']

    # Get unique videos
    unique_videos = np.unique(video_ids)
    n_videos = len(unique_videos)
    n_windows = len(labels)

    # Get video-level labels (assuming all windows in same video have same label)
    video_to_label = {}
    for vid in unique_videos:
        vid_mask = video_ids == vid
        video_label = labels[vid_mask][0]  # Take first window's label
        video_to_label[vid] = video_label

    # Count videos per class
    video_labels = list(video_to_label.values())
    class_counts = Counter(video_labels)

    print(f"  {split_name}:")
    print(f"    Total windows: {n_windows}")
    print(f"    Total videos:  {n_videos}")
    print(f"    Videos per class: {dict(sorted(class_counts.items()))}")


def data_generator(data_path, configs, training_mode, logger=None):
    """
    Enhanced data generator that reports video-level statistics.

    Args:
        data_path: Path to dataset directory
        configs: Configuration object
        training_mode: Training mode string
        logger: Optional logger for output

    Returns:
        train_loader, test_loader
    """
    batch_size = configs.batch_size

    # Load appropriate training dataset based on mode
    if "_1shot" in training_mode or "_1p" in training_mode or training_mode == "supervised_1shot":
        train_dataset = torch.load(os.path.join(data_path, "train_1shot.pt"))
        if training_mode == "supervised_1shot":
            dataset_type = "1-shot (supervised baseline, no pretraining)"
        else:
            dataset_type = "1-shot (video-level)"
    elif "_5shot" in training_mode or "_5p" in training_mode or training_mode == "supervised_5shot":
        train_dataset = torch.load(os.path.join(data_path, "train_5shot.pt"))
        if training_mode == "supervised_5shot":
            dataset_type = "5-shot (supervised baseline, no pretraining)"
        else:
            dataset_type = "5-shot (video-level)"
    elif "_10p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_10perc.pt"))
        dataset_type = "10% (percentage-based)"
    elif "_50p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_50perc.pt"))
        dataset_type = "50% (percentage-based)"
    elif "_75p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_75perc.pt"))
        dataset_type = "75% (percentage-based)"
    elif training_mode == "SupCon":
        train_dataset = torch.load(os.path.join(data_path, "pseudo_train_data.pt"))
        dataset_type = "Pseudo-labeled"
    elif training_mode == "self_supervised":
        # Use pretrain.pt (train+val) for self-supervised pretraining (NO test!)
        train_dataset = torch.load(os.path.join(data_path, "pretrain.pt"))
        dataset_type = "Pretrain set (train+val, NO test!)"
    elif training_mode == "0shot" or training_mode == "eval_pretrained":
        # For 0-shot, we don't actually need train data, but load it anyway for consistency
        train_dataset = torch.load(os.path.join(data_path, "train.pt"))
        dataset_type = "0-shot (evaluate pretrained model, NO training)"
    else:
        train_dataset = torch.load(os.path.join(data_path, "train.pt"))
        dataset_type = "Full training set"

    # Load validation and test datasets
    val_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    # Print statistics
    print("\n" + "="*70)
    print(f"DATASET STATISTICS - {dataset_type}")
    print("="*70)
    print_video_statistics(train_dataset, "TRAIN")
    print_video_statistics(val_dataset, "VAL")
    print_video_statistics(test_dataset, "TEST")
    print("="*70 + "\n")

    if logger:
        logger.debug("\n" + "="*70)
        logger.debug(f"DATASET STATISTICS - {dataset_type}")
        logger.debug("="*70)
        # Log to file as well
        if 'video_ids' in train_dataset:
            train_videos = len(np.unique(train_dataset['video_ids'].numpy()))
            logger.debug(f"  TRAIN: {len(train_dataset['labels'])} windows, {train_videos} videos")
        if 'video_ids' in test_dataset:
            test_videos = len(np.unique(test_dataset['video_ids'].numpy()))
            logger.debug(f"  TEST: {len(test_dataset['labels'])} windows, {test_videos} videos")
        logger.debug("="*70 + "\n")

    # Create Dataset objects
    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    val_dataset = Load_Dataset(val_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    # Adjust batch size if needed
    if train_dataset.__len__() < batch_size:
        batch_size = 16

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                             shuffle=False, drop_last=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)

    return train_loader, val_loader, test_loader
