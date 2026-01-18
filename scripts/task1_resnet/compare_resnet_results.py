import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

def get_latest_run(task_dir, exp_name):
    # Check checks for direct path first
    if Path(exp_name).exists() and (Path(exp_name) / "metrics.json").exists():
        return Path(exp_name)

    exp_dir = task_dir / exp_name
    if not exp_dir.exists():
        return None
    runs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    if not runs:
        return None
    return runs[-1]

def main():
    parser = argparse.ArgumentParser(description="Compare training results between baseline and modified ResNet runs.")
    args = parser.parse_args()

    # Hardcoded configuration
    baseline_exp = "resnet152_head_only_cifar10"
    modified_exp = "resnet152_modified_cifar10"
    runs_dir_str = "runs/task1/resnet"
    output_file = "comparison_results.png"

    # Determine runs root
    runs_root = Path(runs_dir_str)
    if not runs_root.exists():
        # Attempt to resolve relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        runs_root = project_root / "runs" / "task1" / "resnet"
    
    print(f"Looking for runs in: {runs_root}")

    baseline_run = get_latest_run(runs_root, baseline_exp)
    modified_run = get_latest_run(runs_root, modified_exp)

    runs_to_plot = []
    if baseline_run: 
        print(f"Found Baseline run: {baseline_run}")
        runs_to_plot.append(("Baseline", baseline_run))
    else:
        print(f"Warning: Baseline run '{baseline_exp}' not found.")

    if modified_run: 
        print(f"Found Modified run: {modified_run}")
        runs_to_plot.append(("Modified", modified_run))
    else:
        print(f"Warning: Modified run '{modified_exp}' not found.")

    if not runs_to_plot:
        print("No runs found to plot.")
        return

    plt.figure(figsize=(15, 6))

    # --- Plot 1: Training Loss (Dynamics) ---
    plt.subplot(1, 2, 1)
    for label, run_path in runs_to_plot:
        metrics_file = run_path / "metrics.json"
        if not metrics_file.exists():
            print(f"metrics.json missing in {run_path}")
            continue
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            epochs = [e["epoch"] for e in data["epochs"]]
            tr_loss = [e["train_loss"] for e in data["epochs"]]
            plt.plot(epochs, tr_loss, label=f"{label} Train Loss", linestyle='--' if label == "Baseline" else '-')
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Dynamics (Loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot 2: Validation Accuracy ---
    plt.subplot(1, 2, 2)
    for label, run_path in runs_to_plot:
        metrics_file = run_path / "metrics.json"
        if not metrics_file.exists():
            continue
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            epochs = [e["epoch"] for e in data["epochs"]]
            val_acc = [e["val_acc"] for e in data["epochs"]]
            plt.plot(epochs, val_acc, label=f"{label} Val Acc", linestyle='--' if label == "Baseline" else '-')
        except Exception as e:
            pass

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved comparison plot to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
