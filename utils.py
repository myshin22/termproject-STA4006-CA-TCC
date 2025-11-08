import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from torch import nn


def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        # if name=='weight':
        #     nn.init.kaiming_uniform_(param.data)
        # else:
        #     torch.nn.init.zeros_(param.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path, video_ids=None):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)
    if video_ids is not None:
        np.save(os.path.join(labels_save_path, "video_ids.npy"), np.array(video_ids))

    # Window-level metrics
    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save window-level classification report (CSV only for easier parsing)
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)

    # Save as CSV only (Excel removed to save space)
    csv_file_name = f"{exp_name}_{training_mode}_classification_report.csv"
    csv_save_path = os.path.join(home_path, log_dir, csv_file_name)
    df.to_csv(csv_save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)

    # Trial-level metrics (if video_ids provided)
    if video_ids is not None:
        trial_pred, trial_true = _aggregate_to_trial_level(pred_labels, true_labels, video_ids)

        # Calculate trial-level metrics
        trial_r = classification_report(trial_true, trial_pred, digits=6, output_dict=True)
        trial_cm = confusion_matrix(trial_true, trial_pred)
        trial_df = pd.DataFrame(trial_r)
        trial_df["cohen"] = cohen_kappa_score(trial_true, trial_pred)
        trial_df["accuracy"] = accuracy_score(trial_true, trial_pred)
        trial_df = trial_df * 100

        # Save trial-level results (CSV only for easier parsing)
        trial_csv_name = f"{exp_name}_{training_mode}_TRIAL_classification_report.csv"
        trial_csv_path = os.path.join(home_path, log_dir, trial_csv_name)
        trial_df.to_csv(trial_csv_path)

        trial_cm_name = f"{exp_name}_{training_mode}_TRIAL_confusion_matrix.torch"
        trial_cm_path = os.path.join(home_path, log_dir, trial_cm_name)
        torch.save(trial_cm, trial_cm_path)


def _aggregate_to_trial_level(pred_labels, true_labels, video_ids):
    """
    Aggregate window-level predictions to trial-level using majority voting.

    Args:
        pred_labels: Window-level predictions [N]
        true_labels: Window-level true labels [N]
        video_ids: Video ID for each window [N]

    Returns:
        trial_pred, trial_true: Trial-level predictions and labels
    """
    from collections import Counter

    unique_videos = np.unique(video_ids)

    trial_preds = []
    trial_trues = []

    for video_id in unique_videos:
        # Get all windows for this video
        mask = video_ids == video_id

        video_preds = pred_labels[mask]
        video_trues = true_labels[mask]

        # Majority voting for prediction
        vote_counts = Counter(video_preds)
        trial_pred = vote_counts.most_common(1)[0][0]

        # True label should be consistent across all windows
        trial_true = Counter(video_trues).most_common(1)[0][0]

        trial_preds.append(trial_pred)
        trial_trues.append(trial_true)

    return np.array(trial_preds), np.array(trial_trues)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='w')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


from shutil import copy


def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    # copy("args.py", os.path.join(destination_dir, "args.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))
