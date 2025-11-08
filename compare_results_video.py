"""
NEW: Compare results for video-level few-shot experiments.

This is a NEW script for the video-level pipeline.
The original compare_results.py remains unchanged.

Supports the new naming convention:
- Baseline_1shot/supervised_1shot
- Baseline_5shot/supervised_5shot
- CATCC/ft_1shot
- CATCC/ft_5shot
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from glob import glob
from scipy import stats
from pathlib import Path

def collect_results(experiment_dir, run_name, training_mode_pattern):
    """
    Collect classification reports for a specific run and training mode.

    Returns:
        List of tuples: [(seed, window_acc, video_acc, window_df, video_df), ...]
    """
    results = []

    # Find all seed directories
    pattern = os.path.join(experiment_dir, run_name, f"{training_mode_pattern}_seed_*")
    seed_dirs = sorted(glob(pattern))

    for seed_dir in seed_dirs:
        # Extract seed number
        seed_num = seed_dir.split('_seed_')[-1]

        # Look for classification report files
        csv_files = glob(os.path.join(seed_dir, "*_classification_report.csv"))
        trial_csv_files = glob(os.path.join(seed_dir, "*_TRIAL_classification_report.csv"))

        if not csv_files:
            continue

        # Read window-level results
        window_df = pd.read_csv(csv_files[0], index_col=0)
        # Accuracy is in the 'support' row, 'accuracy' column
        window_acc = window_df.loc['support', 'accuracy'] if 'accuracy' in window_df.columns and 'support' in window_df.index else None

        # Read video-level results (if exists)
        video_acc = None
        video_df = None
        if trial_csv_files:
            video_df = pd.read_csv(trial_csv_files[0], index_col=0)
            video_acc = video_df.loc['support', 'accuracy'] if 'accuracy' in video_df.columns and 'support' in video_df.index else None

        results.append({
            'seed': int(seed_num),
            'window_acc': window_acc,
            'video_acc': video_acc,
            'window_df': window_df,
            'video_df': video_df
        })

    return results


def compute_statistics(results):
    """Compute mean and std for accuracies."""
    if not results:
        return None

    window_accs = [r['window_acc'] for r in results if r['window_acc'] is not None]
    video_accs = [r['video_acc'] for r in results if r['video_acc'] is not None]

    stats_dict = {}

    if window_accs:
        stats_dict['window_mean'] = np.mean(window_accs)
        stats_dict['window_std'] = np.std(window_accs)
        stats_dict['window_values'] = window_accs

    if video_accs:
        stats_dict['video_mean'] = np.mean(video_accs)
        stats_dict['video_std'] = np.std(video_accs)
        stats_dict['video_values'] = video_accs

    stats_dict['n_seeds'] = len(results)

    return stats_dict


def compute_pvalue(baseline_values, comparison_values):
    """Compute p-value using t-test."""
    if len(baseline_values) < 2 or len(comparison_values) < 2:
        return None

    # Two-tailed t-test
    t_stat, p_val = stats.ttest_ind(comparison_values, baseline_values)
    return p_val


def print_comparison_table(all_stats):
    """Print a formatted comparison table."""

    print("\n" + "="*100)
    print("WINDOW-LEVEL ACCURACY (Standard Metric)")
    print("="*100)

    print(f"{'Method':<40} {'Seeds':<8} {'Mean Acc':<12} {'Std':<10} {'vs Baseline 1-shot':<20} {'vs Baseline 5-shot':<20}")
    print("-"*100)

    # Get baseline values
    baseline_1shot_window = all_stats.get('Baseline_1shot', {}).get('window_values', [])
    baseline_5shot_window = all_stats.get('Baseline_5shot', {}).get('window_values', [])

    for method_name in ['Baseline_1shot', 'Baseline_5shot', 'Baseline_100p', 'CATCC_ft_1shot', 'CATCC_ft_5shot']:
        if method_name not in all_stats:
            continue

        stats_dict = all_stats[method_name]

        if 'window_mean' not in stats_dict:
            continue

        mean_acc = stats_dict['window_mean']
        std_acc = stats_dict['window_std']
        n_seeds = stats_dict['n_seeds']

        # Compute p-values
        p_vs_1shot = ""
        p_vs_5shot = ""

        if method_name != 'Baseline_1shot' and baseline_1shot_window:
            p_val = compute_pvalue(baseline_1shot_window, stats_dict['window_values'])
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                p_vs_1shot = f"p={p_val:.4f} {sig}"

        if method_name not in ['Baseline_1shot', 'Baseline_5shot'] and baseline_5shot_window:
            p_val = compute_pvalue(baseline_5shot_window, stats_dict['window_values'])
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                p_vs_5shot = f"p={p_val:.4f} {sig}"

        print(f"{method_name:<40} {n_seeds:<8} {mean_acc:<12.2f} {std_acc:<10.4f} {p_vs_1shot:<20} {p_vs_5shot:<20}")

    print("\n" + "="*100)
    print("VIDEO-LEVEL ACCURACY (Majority Voting - More Important!)")
    print("="*100)

    print(f"{'Method':<40} {'Seeds':<8} {'Mean Acc':<12} {'Std':<10} {'vs Baseline 1-shot':<20} {'vs Baseline 5-shot':<20}")
    print("-"*100)

    # Get baseline values
    baseline_1shot_video = all_stats.get('Baseline_1shot', {}).get('video_values', [])
    baseline_5shot_video = all_stats.get('Baseline_5shot', {}).get('video_values', [])

    for method_name in ['Baseline_1shot', 'Baseline_5shot', 'Baseline_100p', 'CATCC_ft_1shot', 'CATCC_ft_5shot']:
        if method_name not in all_stats:
            continue

        stats_dict = all_stats[method_name]

        if 'video_mean' not in stats_dict:
            continue

        mean_acc = stats_dict['video_mean']
        std_acc = stats_dict['video_std']
        n_seeds = stats_dict['n_seeds']

        # Compute p-values
        p_vs_1shot = ""
        p_vs_5shot = ""

        if method_name != 'Baseline_1shot' and baseline_1shot_video:
            p_val = compute_pvalue(baseline_1shot_video, stats_dict['video_values'])
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                p_vs_1shot = f"p={p_val:.4f} {sig}"

        if method_name not in ['Baseline_1shot', 'Baseline_5shot'] and baseline_5shot_video:
            p_val = compute_pvalue(baseline_5shot_video, stats_dict['video_values'])
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                p_vs_5shot = f"p={p_val:.4f} {sig}"

        print(f"{method_name:<40} {n_seeds:<8} {mean_acc:<12.2f} {std_acc:<10.4f} {p_vs_1shot:<20} {p_vs_5shot:<20}")

    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Compare video-level few-shot experiment results')
    parser.add_argument('--experiment_name', type=str, default='CA_TCC_VideoLevel',
                       help='Experiment name (default: CA_TCC_VideoLevel)')
    parser.add_argument('--dataset', type=str, default='ExerciseIMU',
                       help='Dataset name (not used but kept for compatibility)')
    parser.add_argument('--base_dir', type=str, default='experiments_logs',
                       help='Base directory for experiments')

    args = parser.parse_args()

    experiment_dir = os.path.join(args.base_dir, args.experiment_name)

    print("="*100)
    print("Video-Level Few-Shot Learning Results Comparison")
    print("="*100)
    print(f"Experiment: {args.experiment_name}")
    print(f"Directory: {experiment_dir}")
    print("="*100)

    if not os.path.exists(experiment_dir):
        print(f"\n⚠ Error: Experiment directory not found: {experiment_dir}")
        print("Please run experiments first using: ./run_experiments_video.sh ExerciseIMU 0 4")
        return

    print("\nCollecting results...")

    # Define experiments to collect
    experiments = {
        'Baseline_1shot': ('Baseline_1shot', 'supervised_1shot'),
        'Baseline_5shot': ('Baseline_5shot', 'supervised_5shot'),
        'Baseline_100p': ('Baseline_100p', 'supervised'),
        'CATCC_ft_1shot': ('CATCC', 'ft_1shot'),
        'CATCC_ft_5shot': ('CATCC', 'ft_5shot'),
    }

    all_stats = {}

    for method_name, (run_name, training_mode) in experiments.items():
        print(f"  - {method_name}...", end=' ')
        results = collect_results(experiment_dir, run_name, training_mode)

        if results:
            stats_dict = compute_statistics(results)
            all_stats[method_name] = stats_dict
            print(f"✓ Found {len(results)} seeds")
        else:
            print(f"⚠ No results found")

    if not all_stats:
        print("\n⚠ No results found. Please run experiments first.")
        return

    # Print comparison table
    print_comparison_table(all_stats)

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    if 'CATCC_ft_1shot' in all_stats and 'Baseline_1shot' in all_stats:
        catcc_1shot_video = all_stats['CATCC_ft_1shot'].get('video_mean', 0)
        baseline_1shot_video = all_stats['Baseline_1shot'].get('video_mean', 0)
        improvement_1shot = catcc_1shot_video - baseline_1shot_video

        print(f"\n1-shot Learning (Video-Level):")
        print(f"  Baseline:     {baseline_1shot_video:.2f}%")
        print(f"  CA-TCC:       {catcc_1shot_video:.2f}%")
        print(f"  Improvement:  {improvement_1shot:+.2f}% ({improvement_1shot/baseline_1shot_video*100:+.1f}% relative)")

    if 'CATCC_ft_5shot' in all_stats and 'Baseline_5shot' in all_stats:
        catcc_5shot_video = all_stats['CATCC_ft_5shot'].get('video_mean', 0)
        baseline_5shot_video = all_stats['Baseline_5shot'].get('video_mean', 0)
        improvement_5shot = catcc_5shot_video - baseline_5shot_video

        print(f"\n5-shot Learning (Video-Level):")
        print(f"  Baseline:     {baseline_5shot_video:.2f}%")
        print(f"  CA-TCC:       {catcc_5shot_video:.2f}%")
        print(f"  Improvement:  {improvement_5shot:+.2f}% ({improvement_5shot/baseline_5shot_video*100:+.1f}% relative)")

    print("\n" + "="*100)
    print("✓ Analysis complete!")
    print("="*100)


if __name__ == '__main__':
    main()
