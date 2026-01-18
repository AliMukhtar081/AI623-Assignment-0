import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def get_subdirs(path: Path):
    if not path.exists():
        return []
    return sorted([d for d in path.iterdir() if d.is_dir()])

def select_from_list(options, name):
    print(f"\nAvailable {name}s:")
    for i, opt in enumerate(options):
        print(f"{i+1}: {opt.name}")
    
    while True:
        try:
            choice = input(f"Select {name} (1-{len(options)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("Invalid selection. Please try again.")

def main():
    root = Path("runs")
    if not root.exists():
        print("No 'runs' directory found.")
        return

    # Select Task
    tasks = get_subdirs(root)
    if not tasks:
        print("No tasks found in runs/.")
        return
    task_dir = select_from_list(tasks, "Task")

    # Select Model
    models = get_subdirs(task_dir)
    if not models:
        print(f"No models found in {task_dir}.")
        return
    model_dir = select_from_list(models, "Model")

    # Select Run
    runs_list = get_subdirs(model_dir)
    if not runs_list:
        print(f"No runs found in {model_dir}.")
        return
    run_dir = select_from_list(runs_list, "Run")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"No metrics.json found in {run_dir}")
        return

    print(f"Plotting metrics from {metrics_path}")

    with open(metrics_path, "r") as f:
        m = json.load(f)

    epochs = [e["epoch"] for e in m["epochs"]]
    tr_loss = [e["train_loss"] for e in m["epochs"]]
    va_loss = [e["val_loss"] for e in m["epochs"]]
    tr_acc  = [e["train_acc"] for e in m["epochs"]]
    va_acc  = [e["val_acc"] for e in m["epochs"]]

    # Output directory mirroring runs structure
    out_dir = Path("figs") / task_dir.name / model_dir.name / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / "curves.png"

    plt.figure()
    plt.plot(epochs, tr_loss, label="train_loss")
    plt.plot(epochs, va_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(str(out_base).replace(".png", "_loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, tr_acc, label="train_acc")
    plt.plot(epochs, va_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(str(out_base).replace(".png", "_acc.png"), dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plots to {out_dir}")

if __name__ == "__main__":
    main()
