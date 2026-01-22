import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import types
from torchvision.models import resnet152, ResNet152_Weights

from src.data.cifar10 import get_cifar10_loaders
from src.data.cifar100 import get_cifar100_loaders
from src.utils.repro import set_seed
from src.utils.logger import make_run_dir
from src.utils.model_analysis import get_model_summary
from src.utils.train_utils import fit
from src.utils.constants import NUM_CLASSES_CIFAR10, NUM_CLASSES_CIFAR100, CIFAR10_IMAGE_SIZE, IMAGENET_IMAGE_SIZE

from configs.task1_config import (
    BASELINE, MODIFIED,
    TRANSFER_PRETRAINED_HEAD_ONLY, TRANSFER_PRETRAINED_FINAL_BLOCK, TRANSFER_PRETRAINED_FULL,
    TRANSFER_RANDOM_HEAD_ONLY, TRANSFER_RANDOM_FINAL_BLOCK, TRANSFER_RANDOM_FULL
)

def no_skip_forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)
    out = self.relu(out)
    return out

def run_baseline():
    print("\n" + "="*80)
    print("EXPERIMENT 1: Baseline ResNet-152 (CIFAR-10, Head Only)")
    print("="*80)
    
    cfg = BASELINE
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_cifar10_loaders(batch_size=cfg.batch_size)
    
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES_CIFAR10)
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=cfg.lr)
    model_stats = get_model_summary(model, input_size=(1, 3, CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE), device=device)
    
    run_dir = make_run_dir("task1/resnet", cfg.exp)
    config = {"epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr, "seed": cfg.seed}
    config.update(model_stats)
    
    fit(model=model, train_loader=train_loader, val_loader=val_loader, 
        optimizer=optimizer, device=device, epochs=cfg.epochs, 
        run_dir=run_dir, config=config, save_final_name="model_head_only.pt")

def run_modified():
    print("\n" + "="*80)
    print("EXPERIMENT 2: Modified ResNet-152 (CIFAR-10, Skip Connections Disabled)")
    print("="*80)
    
    cfg = MODIFIED
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_cifar10_loaders(batch_size=cfg.batch_size)
    
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    
    for block_name in cfg.blocks_to_disable:
        parts = block_name.strip().split(".")
        layer_name, idx_str = parts
        if not hasattr(model, layer_name):
            continue
        layer_module = getattr(model, layer_name)
        block_idx = int(idx_str)
        if block_idx >= len(layer_module):
            continue
        block = layer_module[block_idx]
        block.forward = types.MethodType(no_skip_forward, block)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES_CIFAR10)
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=cfg.lr)
    model_stats = get_model_summary(model, input_size=(1, 3, CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE), device=device)
    
    run_dir = make_run_dir("task1/resnet", cfg.exp)
    with open(run_dir / "modified_blocks.txt", "w") as f:
        f.write(",".join(cfg.blocks_to_disable))
    
    config = {"epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr, "seed": cfg.seed}
    config.update(model_stats)
    
    fit(model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device, epochs=cfg.epochs,
        run_dir=run_dir, config=config, save_final_name="model_modified.pt")

def run_transfer(cfg):
    prefix = "Pretrained" if cfg.pretrained else "Random Init"
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {prefix} ResNet-152 (CIFAR-100, {cfg.freeze_mode})")
    print("="*80)
    
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_cifar100_loaders(batch_size=cfg.batch_size)
    
    if cfg.pretrained:
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)
    else:
        model = resnet152(weights=None)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES_CIFAR100)
    
    if cfg.freeze_mode == "head_only":
        for name, param in model.named_parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        trainable_params = model.fc.parameters()
    elif cfg.freeze_mode == "final_block":
        for name, param in model.named_parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        trainable_params = list(model.layer4.parameters()) + list(model.fc.parameters())
    elif cfg.freeze_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = model.parameters()
    
    model.to(device)
    
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr)
    model_stats = get_model_summary(model, input_size=(1, 3, IMAGENET_IMAGE_SIZE, IMAGENET_IMAGE_SIZE), device=device)
    
    run_dir = make_run_dir("task1/transfer_learning", cfg.get_exp_name())
    
    config = {
        "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr, "seed": cfg.seed,
        "pretrained": cfg.pretrained, "freeze_mode": cfg.freeze_mode,
        "total_params": total_params, "trainable_params": trainable_params_count
    }
    config.update(model_stats)
    
    fit(model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, device=device, epochs=cfg.epochs,
        run_dir=run_dir, config=config, save_final_name="model_final.pt")

def main():
   
    print("# TASK 1: Complete ResNet Analysis & Transfer Learning Pipeline")
    
    
    run_baseline()
    
    run_modified()
    
    run_transfer(TRANSFER_PRETRAINED_HEAD_ONLY)
    run_transfer(TRANSFER_PRETRAINED_FINAL_BLOCK)
    run_transfer(TRANSFER_PRETRAINED_FULL)
    
    run_transfer(TRANSFER_RANDOM_HEAD_ONLY)
    run_transfer(TRANSFER_RANDOM_FINAL_BLOCK)
    run_transfer(TRANSFER_RANDOM_FULL)
    
   
    print("# ALL EXPERIMENTS COMPLETE")
  

if __name__ == "__main__":
    main()
