# Task 1: ResNet Analysis & Transfer Learning

## Run All Experiments

Run the complete Task 1 pipeline with unified config:
```bash
python scripts/task1_resnet/task1_complete.py
```

This executes all experiments step-by-step:
- Baseline ResNet-152 (CIFAR-10)
- Modified ResNet-152 with disabled skip connections (CIFAR-10)
- Transfer learning: pretrained × 3 freeze modes (CIFAR-100)
- Transfer learning: random init × 3 freeze modes (CIFAR-100)

**Configuration**: Edit `configs/task1_config.py` to modify hyperparameters.


All scripts load configuration from `configs/task1_config.py`:

```bash
python scripts/task1_resnet/train_resnet_baseline.py
python scripts/task1_resnet/train_resnet_modified.py
python scripts/task1_resnet/train_transfer_pretrained.py
python scripts/task1_resnet/train_transfer_random.py
```

## Analysis & Visualization

### Feature Visualization
```bash
python scripts/task1_resnet/visualize_feature_hierarchies.py --model_path runs/task1/resnet/<run_name>/model_best.pt
```

### Transfer Learning Comparison
```bash
python scripts/task1_resnet/generate_transfer_plot.py
```
