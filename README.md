# AI623-Assignment-0

## Setup
Create environment (conda or venv) and install dependencies.

```bash
conda env create -f environment.yml
conda activate ai623-dvlm
```

## Task 1: ResNet-152 on CIFAR-10

### 1. Train Baseline (Head Only)
This script trains the head of a pre-trained ResNet-152 on CIFAR-10.

```bash
python scripts/task1_resnet/train_resnet_baseline.py --epochs 10
```

### 2. Train with Disabled Skip Connections
This script trains the model with skip connections explicitly disabled in `layer3.0` and `layer4.1`. It also runs a model analysis (params & FLOPs) before training.

```bash
python scripts/task1_resnet/train_resnet_disabled_skip.py --epochs 10
```

### 3. Compare Results
You can compare the training curves of different runs using the interactive plotting tool.

```bash
python scripts/plot_metrics.py
```
Follow the interactive prompts to select the runs you want to compare.

Figures are saved under `figs/`.

---

## Transfer Learning and Generalization (Task 1.3)

This section explores transfer learning by fine-tuning ResNet-152 on **CIFAR-100** (a different dataset from ImageNet). We compare:
- **(a)** Using ImageNet-pretrained weights vs. training from random initialization
- **(b)** Fine-tuning only the final block vs. the full backbone vs. head-only

### Dataset: CIFAR-100
CIFAR-100 contains 60,000 32×32 color images across 100 classes (600 images per class: 500 train, 100 test). This is a good transfer learning benchmark as it differs from ImageNet in resolution, domain, and number of classes.

### Experiments

#### (a) Pretrained vs Random Initialization

**1. Fine-tune with ImageNet-pretrained weights (head-only):**
```bash
python scripts/task1_resnet/train_transfer_pretrained.py --epochs 30 --freeze_mode head_only
```

**2. Train from random initialization (head-only):**
```bash
python scripts/task1_resnet/train_transfer_random.py --epochs 30 --freeze_mode head_only
```

#### (c) Fine-tuning Strategies

**3. Fine-tune only the final block (layer4) - Pretrained:**
```bash
python scripts/task1_resnet/train_transfer_pretrained.py --epochs 30 --freeze_mode final_block
```

**4. Fine-tune only the final block (layer4) - Random:**
```bash
python scripts/task1_resnet/train_transfer_random.py --epochs 30 --freeze_mode final_block
```

**5. Fine-tune the full backbone - Pretrained:**
```bash
python scripts/task1_resnet/train_transfer_pretrained.py --epochs 30 --freeze_mode full --lr 1e-4
```

**6. Fine-tune the full backbone - Random:**
```bash
python scripts/task1_resnet/train_transfer_random.py --epochs 30 --freeze_mode full --lr 1e-4
```

### Compare Transfer Learning Results

After running multiple experiments, compare them using:

```bash
python scripts/task1_resnet/compare_transfer_learning.py \
  --run_dirs runs/task1/transfer_learning/resnet152_pretrained_head_only_cifar100_* \
             runs/task1/transfer_learning/resnet152_random_head_only_cifar100_* \
             runs/task1/transfer_learning/resnet152_pretrained_final_block_cifar100_* \
             runs/task1/transfer_learning/resnet152_random_final_block_cifar100_* \
  --save_dir figs/transfer_learning
```

This generates:
- Comparison plots for validation/training accuracy and loss
- A detailed report analyzing:
  - Pretrained vs random initialization performance
  - Head-only vs final-block vs full backbone trade-offs
  - Compute efficiency (accuracy per trainable parameter)
  - Best configuration for accuracy vs compute trade-off

### Key Questions Addressed

1. **Which setting provides the best trade-off between compute and accuracy?**
   - The comparison report includes an efficiency score (accuracy per million trainable parameters)
   - Typically, fine-tuning the final block offers a good balance

2. **Which layers seem most transferable across datasets, and why?**
   - Early layers (conv1, layer1, layer2) learn general features (edges, textures)
   - Later layers (layer3, layer4) learn task-specific features
   - The report shows performance differences when freezing different layers
