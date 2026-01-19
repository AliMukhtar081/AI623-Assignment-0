
# Task 1: ResNet Analysis & Transfer Learning

## Training Scripts

### Baseline (CIFAR-10)
Trains a standard ResNet-152 on CIFAR-10 (linear probe).
```bash
python scripts/task1_resnet/train_resnet_baseline.py --epochs 100
```

### Modified Architecture (CIFAR-10)
Trains ResNet-152 with skip connections disabled in layers 3.0 and 4.1.
```bash
python scripts/task1_resnet/train_resnet_modified.py --epochs 100
```

### Transfer Learning (CIFAR-100)
Fine-tunes ImageNet-pretrained ResNet-152 on CIFAR-100.
```bash
# Options: head_only, final_block, full
python scripts/task1_resnet/train_transfer_pretrained.py --freeze_mode head_only
```

### Random Initialization (CIFAR-100)
Trains ResNet-152 from scratch on CIFAR-100 for comparison.
```bash
# Options: head_only, final_block, full
python scripts/task1_resnet/train_transfer_random.py --freeze_mode head_only
```

## Analysis & Visualization

### Feature Visualization
Generates t-SNE/UMAP plots of feature representations at different network depths.
```bash
python scripts/task1_resnet/visualize_feature_hierarchies.py --model_path runs/task1/resnet/<run_name>/model_best.pt
```

### Transfer Learning Comparison
Generates comparison plots of validation accuracy for all transfer learning runs.
```bash
python scripts/task1_resnet/generate_transfer_plot.py
```
