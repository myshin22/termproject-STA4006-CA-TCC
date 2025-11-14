"""
Compare results for video-level few-shot experiments.

Updated to support the new experiment structure:
experiments_logs/FewShot_ExerciseIMU/seed_X/{training_mode}_seed_X/

Supports all training modes:
- 0shot: Pretrained model evaluation (no fine-tuning)
- ft_1shot, ft_5shot: Pretrained + fine-tuning
- supervised_1shot, supervised_5shot: Train from scratch (baselines)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from glob import glob
from scipy import stats
from pathlib import Path

def collect_results(experiment_dir, training_mode):
    """
    Collect classification reports for a specific training mode across all seeds.

    New structure: experiments_logs/FewShot_ExerciseIMU/seed_X/{training_mode}_seed_X/

    Returns:
        List of dictionaries with results for each seed
    """
    results = []

    # Find all seed directories
    seed_dirs = sorted(glob(os.path.join(experiment_dir, "seed_*")))

    for seed_dir in seed_dirs:
        # Extract seed number
        seed_num = os.path.basename(seed_dir).split('_')[-1]

        # Look for training mode directory
        mode_dir = os.path.join(seed_dir, f"{training_mode}_seed_{seed_num}")

        if not os.path.exists(mode_dir):
            continue

        # Look for classification report files
        csv_files = glob(os.path.join(mode_dir, "*_classification_report.csv"))
        trial_csv_files = glob(os.path.join(mode_dir, "*_TRIAL_classification_report.csv"))

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

    print("\n" + "="*110)
    print("WINDOW-LEVEL ACCURACY (Standard Metric)")
    print("="*110)

    print(f"{'Method':<35} {'Seeds':<8} {'Mean Acc':<12} {'Std':<10} {'vs Supervised 1-shot':<25} {'vs Supervised 5-shot':<25}")
    print("-"*110)

    # Get baseline values
    baseline_1shot_window = all_stats.get('Supervised_1shot', {}).get('window_values', [])
    baseline_5shot_window = all_stats.get('Supervised_5shot', {}).get('window_values', [])

    for method_name in ['0shot', 'Supervised_1shot', 'CATCC_1shot', 'Supervised_5shot', 'CATCC_5shot', 'Supervised_100%']:
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

        if method_name not in ['Supervised_1shot'] and baseline_1shot_window:
            p_val = compute_pvalue(baseline_1shot_window, stats_dict['window_values'])
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                p_vs_1shot = f"p={p_val:.4f} {sig}"

        if method_name not in ['Supervised_1shot', 'Supervised_5shot'] and baseline_5shot_window:
            p_val = compute_pvalue(baseline_5shot_window, stats_dict['window_values'])
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                p_vs_5shot = f"p={p_val:.4f} {sig}"

        print(f"{method_name:<35} {n_seeds:<8} {mean_acc:<12.2f} {std_acc:<10.4f} {p_vs_1shot:<25} {p_vs_5shot:<25}")

    print("\n" + "="*110)
    print("VIDEO-LEVEL ACCURACY (Majority Voting - More Important!)")
    print("="*110)

    print(f"{'Method':<35} {'Seeds':<8} {'Mean Acc':<12} {'Std':<10} {'vs Supervised 1-shot':<25} {'vs Supervised 5-shot':<25}")
    print("-"*110)

    # Get baseline values
    baseline_1shot_video = all_stats.get('Supervised_1shot', {}).get('video_values', [])
    baseline_5shot_video = all_stats.get('Supervised_5shot', {}).get('video_values', [])

    for method_name in ['0shot', 'Supervised_1shot', 'CATCC_1shot', 'Supervised_5shot', 'CATCC_5shot', 'Supervised_100%']:
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

        if method_name not in ['Supervised_1shot'] and baseline_1shot_video:
            p_val = compute_pvalue(baseline_1shot_video, stats_dict['video_values'])
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                p_vs_1shot = f"p={p_val:.4f} {sig}"

        if method_name not in ['Supervised_1shot', 'Supervised_5shot'] and baseline_5shot_video:
            p_val = compute_pvalue(baseline_5shot_video, stats_dict['video_values'])
            if p_val is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                p_vs_5shot = f"p={p_val:.4f} {sig}"

        print(f"{method_name:<35} {n_seeds:<8} {mean_acc:<12.2f} {std_acc:<10.4f} {p_vs_1shot:<25} {p_vs_5shot:<25}")

    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("="*110)


def main():
    parser = argparse.ArgumentParser(description='Compare video-level few-shot experiment results')
    parser.add_argument('--experiment_name', type=str, default='FewShot_ExerciseIMU',
                       help='Experiment name (default: FewShot_ExerciseIMU)')
    parser.add_argument('--base_dir', type=str, default='experiments_logs',
                       help='Base directory for experiments')
    parser.add_argument('--output_file', type=str, default='comparison_results.txt',
                       help='Output file to save results (default: comparison_results.txt)')

    args = parser.parse_args()

    experiment_dir = os.path.join(args.base_dir, args.experiment_name)

    print("="*110)
    print("CA-TCC Few-Shot Learning Results Comparison")
    print("="*110)
    print(f"Experiment: {args.experiment_name}")
    print(f"Directory: {experiment_dir}")
    print("="*110)

    if not os.path.exists(experiment_dir):
        print(f"\n⚠ Error: Experiment directory not found: {experiment_dir}")
        print("Please run experiments first using: bash run_fewshot_experiments.sh")
        return

    print("\nCollecting results...")

    # Define experiments to collect
    experiments = {
        '0shot': '0shot',
        'Supervised_1shot': 'supervised_1shot',
        'CATCC_1shot': 'ft_1shot',
        'Supervised_5shot': 'supervised_5shot',
        'CATCC_5shot': 'ft_5shot',
        'Supervised_100%': 'supervised',
    }

    all_stats = {}

    for method_name, training_mode in experiments.items():
        print(f"  - {method_name} ({training_mode})...", end=' ')
        results = collect_results(experiment_dir, training_mode)

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
    print("\n" + "="*110)
    print("SUMMARY")
    print("="*110)

    # 0-shot
    if '0shot' in all_stats:
        zeroshot_video = all_stats['0shot'].get('video_mean', 0)
        print(f"\n0-shot (Pretrained, NO fine-tuning):")
        print(f"  Video-level:  {zeroshot_video:.2f}%")

    # 1-shot
    if 'CATCC_1shot' in all_stats and 'Supervised_1shot' in all_stats:
        catcc_1shot_video = all_stats['CATCC_1shot'].get('video_mean', 0)
        baseline_1shot_video = all_stats['Supervised_1shot'].get('video_mean', 0)
        improvement_1shot = catcc_1shot_video - baseline_1shot_video

        print(f"\n1-shot Learning (Video-Level):")
        print(f"  Supervised Baseline:   {baseline_1shot_video:.2f}%")
        print(f"  CA-TCC (Pretrain+FT):  {catcc_1shot_video:.2f}%")
        print(f"  Improvement:           {improvement_1shot:+.2f}% ({improvement_1shot/baseline_1shot_video*100:+.1f}% relative)")

    # 5-shot
    if 'CATCC_5shot' in all_stats and 'Supervised_5shot' in all_stats:
        catcc_5shot_video = all_stats['CATCC_5shot'].get('video_mean', 0)
        baseline_5shot_video = all_stats['Supervised_5shot'].get('video_mean', 0)
        improvement_5shot = catcc_5shot_video - baseline_5shot_video

        print(f"\n5-shot Learning (Video-Level):")
        print(f"  Supervised Baseline:   {baseline_5shot_video:.2f}%")
        print(f"  CA-TCC (Pretrain+FT):  {catcc_5shot_video:.2f}%")
        print(f"  Improvement:           {improvement_5shot:+.2f}% ({improvement_5shot/baseline_5shot_video*100:+.1f}% relative)")

    print("\n" + "="*110)
    print("✓ Analysis complete!")
    print("="*110)

    # Save results to file
    output_path = os.path.join(experiment_dir, args.output_file)
    with open(output_path, 'w') as f:
        f.write("="*110 + "\n")
        f.write("CA-TCC Few-Shot Learning Results Comparison\n")
        f.write("="*110 + "\n")
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Directory: {experiment_dir}\n")
        f.write("="*110 + "\n\n")

        # Write statistics for each method
        f.write("DETAILED RESULTS BY METHOD\n")
        f.write("="*110 + "\n\n")

        for method_name in ['0shot', 'Supervised_1shot', 'CATCC_1shot', 'Supervised_5shot', 'CATCC_5shot', 'Supervised_100%']:
            if method_name not in all_stats:
                continue

            stats_dict = all_stats[method_name]
            f.write(f"{method_name}:\n")
            f.write(f"  Number of seeds: {stats_dict['n_seeds']}\n")

            if 'window_mean' in stats_dict:
                f.write(f"  Window-level accuracy: {stats_dict['window_mean']:.2f}% ± {stats_dict['window_std']:.2f}\n")
                f.write(f"    Individual seeds: {[f'{v:.2f}' for v in stats_dict['window_values']]}\n")

            if 'video_mean' in stats_dict:
                f.write(f"  Video-level accuracy:  {stats_dict['video_mean']:.2f}% ± {stats_dict['video_std']:.2f}\n")
                f.write(f"    Individual seeds: {[f'{v:.2f}' for v in stats_dict['video_values']]}\n")

            f.write("\n")

        # Write summary
        f.write("="*110 + "\n")
        f.write("SUMMARY\n")
        f.write("="*110 + "\n\n")

        if '0shot' in all_stats:
            zeroshot_video = all_stats['0shot'].get('video_mean', 0)
            f.write(f"0-shot (Pretrained, NO fine-tuning):\n")
            f.write(f"  Video-level:  {zeroshot_video:.2f}%\n\n")

        if 'CATCC_1shot' in all_stats and 'Supervised_1shot' in all_stats:
            catcc_1shot_video = all_stats['CATCC_1shot'].get('video_mean', 0)
            baseline_1shot_video = all_stats['Supervised_1shot'].get('video_mean', 0)
            improvement_1shot = catcc_1shot_video - baseline_1shot_video
            f.write(f"1-shot Learning (Video-Level):\n")
            f.write(f"  Supervised Baseline:   {baseline_1shot_video:.2f}%\n")
            f.write(f"  CA-TCC (Pretrain+FT):  {catcc_1shot_video:.2f}%\n")
            f.write(f"  Improvement:           {improvement_1shot:+.2f}% ({improvement_1shot/baseline_1shot_video*100:+.1f}% relative)\n\n")

        if 'CATCC_5shot' in all_stats and 'Supervised_5shot' in all_stats:
            catcc_5shot_video = all_stats['CATCC_5shot'].get('video_mean', 0)
            baseline_5shot_video = all_stats['Supervised_5shot'].get('video_mean', 0)
            improvement_5shot = catcc_5shot_video - baseline_5shot_video
            f.write(f"5-shot Learning (Video-Level):\n")
            f.write(f"  Supervised Baseline:   {baseline_5shot_video:.2f}%\n")
            f.write(f"  CA-TCC (Pretrain+FT):  {catcc_5shot_video:.2f}%\n")
            f.write(f"  Improvement:           {improvement_5shot:+.2f}% ({improvement_5shot/baseline_5shot_video*100:+.1f}% relative)\n\n")

        if 'Supervised_100%' in all_stats:
            supervised_100_video = all_stats['Supervised_100%'].get('video_mean', 0)
            f.write(f"Upper Bound (100% Supervised):\n")
            f.write(f"  Video-level:  {supervised_100_video:.2f}%\n\n")

        f.write("="*110 + "\n")

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
