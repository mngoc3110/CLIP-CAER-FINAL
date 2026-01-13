#!/usr/bin/env python3
"""
Compare results from multiple experiments.
Reads log.txt files from different experiment outputs and generates comparison table.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

def parse_log_file(log_path):
    """Parse log.txt to extract configuration and final metrics."""
    config = {}
    epochs = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Parse config (first few lines)
    for line in lines[:100]:
        if '=' in line and not line.startswith('='):
            parts = line.strip().split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                config[key] = value

    # Parse epoch summaries
    current_epoch = None
    for i, line in enumerate(lines):
        if '--- Epoch' in line and 'Summary ---' in line:
            # Extract epoch number
            match = re.search(r'Epoch (\d+)', line)
            if match:
                current_epoch = int(match.group(1))
                epoch_data = {'epoch': current_epoch}

                # Look ahead for metrics
                for j in range(i+1, min(i+10, len(lines))):
                    if 'Train WAR:' in lines[j]:
                        match = re.search(r'Train WAR: ([\d.]+)% \| Train UAR: ([\d.]+)%', lines[j])
                        if match:
                            epoch_data['train_war'] = float(match.group(1))
                            epoch_data['train_uar'] = float(match.group(2))
                    elif 'Valid WAR:' in lines[j]:
                        match = re.search(r'Valid WAR: ([\d.]+)% \| Valid UAR: ([\d.]+)%', lines[j])
                        if match:
                            epoch_data['val_war'] = float(match.group(1))
                            epoch_data['val_uar'] = float(match.group(2))
                    elif 'Best Valid UAR so far:' in lines[j]:
                        match = re.search(r'Best Valid UAR so far: ([\d.]+)%', lines[j])
                        if match:
                            epoch_data['best_val_uar'] = float(match.group(1))

                epochs.append(epoch_data)

    return config, epochs

def find_experiment_logs(base_dir='outputs'):
    """Find all log.txt files in experiment outputs."""
    base_path = Path(base_dir)
    log_files = list(base_path.glob('*/log.txt'))
    return log_files

def main():
    print("=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    print()

    # Find all experiments
    log_files = find_experiment_logs()

    if not log_files:
        print("No experiment logs found in outputs/ directory.")
        sys.exit(1)

    print(f"Found {len(log_files)} experiments:\n")

    experiments = {}
    for log_file in log_files:
        exp_name = log_file.parent.name
        config, epochs = parse_log_file(log_file)

        if not epochs:
            print(f"  ⚠ {exp_name}: No epoch data found")
            continue

        # Get best epoch
        best_epoch = max(epochs, key=lambda x: x.get('best_val_uar', 0))

        experiments[exp_name] = {
            'config': config,
            'epochs': epochs,
            'best_epoch': best_epoch,
            'log_path': log_file
        }

        print(f"  ✓ {exp_name}: {len(epochs)} epochs, Best UAR: {best_epoch.get('best_val_uar', 0):.2f}%")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()

    # Sort by best UAR
    sorted_exps = sorted(experiments.items(), key=lambda x: x[1]['best_epoch'].get('best_val_uar', 0), reverse=True)

    # Print header
    print(f"{'Experiment':<40} | {'Best Val UAR':>12} | {'Train UAR':>10} | {'Gap':>6} | {'Epoch':>5}")
    print("-" * 80)

    for exp_name, data in sorted_exps:
        best = data['best_epoch']
        val_uar = best.get('best_val_uar', 0)
        train_uar = best.get('train_uar', 0)
        gap = train_uar - best.get('val_uar', 0)
        epoch = best.get('epoch', 0)

        # Truncate long experiment names
        short_name = exp_name[:38] if len(exp_name) > 38 else exp_name

        print(f"{short_name:<40} | {val_uar:>11.2f}% | {train_uar:>9.2f}% | {gap:>5.1f}% | {epoch:>5}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if len(sorted_exps) >= 2:
        best_exp = sorted_exps[0]
        baseline = next((exp for name, exp in sorted_exps if 'baseline' in name.lower()), None)

        if baseline and best_exp[0] != baseline[0]:
            improvement = best_exp[1]['best_epoch']['best_val_uar'] - baseline['best_epoch']['best_val_uar']
            print(f"\n✓ Best experiment: {best_exp[0]}")
            print(f"  UAR: {best_exp[1]['best_epoch']['best_val_uar']:.2f}%")
            print(f"  Improvement over baseline: +{improvement:.2f}%")

    # Check for class 2 recall in logs
    print("\n" + "=" * 80)
    print("CLASS 2 (CONFUSION) RECALL CHECK")
    print("=" * 80)

    for exp_name, data in sorted_exps[:3]:  # Top 3 experiments
        log_path = data['log_path']
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Find last validation per-class recall
        for line in reversed(lines):
            if 'Per-class Recall:' in line and 'Valid' in lines[lines.index(line) - 1]:
                # Extract recalls
                recalls = re.findall(r'([\d.]+)', line)
                if len(recalls) >= 5:
                    class_recalls = [float(r) for r in recalls[:5]]
                    print(f"\n{exp_name}:")
                    class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
                    for i, (name, recall) in enumerate(zip(class_names, class_recalls)):
                        marker = " ✓" if recall >= 40 else " ⚠" if recall >= 30 else " ✗"
                        print(f"  Class {i} ({name:12s}): {recall:5.1f}%{marker}")
                break

    print("\n" + "=" * 80)
    print("To view detailed logs, check:")
    for exp_name, _ in sorted_exps[:3]:
        print(f"  - outputs/{exp_name}/log.txt")
    print()

if __name__ == '__main__':
    main()
