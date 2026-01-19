
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

def load_metrics(metrics_path):
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} does not exist.")
        return None
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {metrics_path}: {e}")
        return None

def plot_comparisons(runs_data, output_path=None):
    if not runs_data:
        print("No data to plot.")
        return

    is_comparison = len(runs_data) > 1
    title_suffix = " Comparison" if is_comparison else ""

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, m in runs_data:
        if "epochs" not in m: continue
        epochs = [e["epoch"] for e in m["epochs"]]
        tr_loss = [e["train_loss"] for e in m["epochs"]]
        va_loss = [e.get("val_loss", None) for e in m["epochs"]]
        
        plt.plot(epochs, tr_loss, label=f"{label} (Train)", linestyle='-')
        if any(v is not None for v in va_loss):
            plt.plot(epochs, va_loss, label=f"{label} (Val)", linestyle='--')
            
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss{title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for label, m in runs_data:
        if "epochs" not in m: continue
        epochs = [e["epoch"] for e in m["epochs"]]
        tr_acc = [e["train_acc"] for e in m["epochs"]]
        va_acc = [e.get("val_acc", None) for e in m["epochs"]]

        plt.plot(epochs, tr_acc, label=f"{label} (Train)", linestyle='-')
        if any(v is not None for v in va_acc):
            plt.plot(epochs, va_acc, label=f"{label} (Val)", linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy{title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()

def interactive_selection(root_dir):
    selected_runs = []
    
    while True:
        tasks = get_subdirs(root_dir)
        if not tasks:
            print("No tasks found in runs/.")
            break
            
        task_dir = select_from_list(tasks, "Task")

        models = get_subdirs(task_dir)
        if not models:
            print(f"No models found in {task_dir}.")
            continue
            
        model_dir = select_from_list(models, "Model")

        runs_list = get_subdirs(model_dir)
        if not runs_list:
            print(f"No runs found in {model_dir}.")
            continue
            
        run_dir = select_from_list(runs_list, "Run")
        
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            label = f"{task_dir.name}/{model_dir.name}/{run_dir.name}"
            custom_label = input(f"Enter label for this run (default: {label}): ").strip()
            if custom_label:
                label = custom_label
                
            selected_runs.append((label, metrics_path))
            print(f"-> Added {label}")
        else:
            print(f"No metrics.json found in {run_dir}")

        cont = input("\nAdd another run to compare? (y/n): ").lower().strip()
        if cont != 'y':
            break
            
    return selected_runs

def main():
    root = Path("runs")
    if not root.exists():
        print("No 'runs' directory found.")
        return

    print("--- Interactive Model Comparison ---")
    selections = interactive_selection(root)
    
    data_to_plot = []
    for label, path in selections:
        m = load_metrics(path)
        if m:
            data_to_plot.append((label, m))

    if not data_to_plot:
        print("No valid metrics found to plot.")
        return

    if len(data_to_plot) == 1:
        label = data_to_plot[0][0]
        safe_label = label.strip().replace(" ", "_")
        default_out = f"figs/{safe_label}_metrics.png"
    else:
        labels = [d[0] for d in data_to_plot]
        joined = "_vs_".join([l.replace("/", "_").replace(" ", "_") for l in labels])
        if len(joined) > 50:
            joined = joined[:47] + "..."
        default_out = f"figs/comparison_{joined}.png"

    out_name = input(f"Enter output filename (default: {default_out}): ").strip()
    if not out_name:
        out_name = default_out
    
    out_path = Path(out_name)
    plot_comparisons(data_to_plot, output_path=out_path)

if __name__ == "__main__":
    main()
