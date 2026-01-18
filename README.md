# AI623-Assignment-0
## Setup
Create environment (conda or venv) and install dependencies.

## Task 1 Baseline (ResNet-152 head-only on CIFAR-10)
Run:
python scripts/task1_resnet/train_resnet_baseline.py --epochs 2

Then plot:
python scripts/plot_metrics.py --metrics <path_to_runs/.../metrics.json>

Figures are saved under figs/.
