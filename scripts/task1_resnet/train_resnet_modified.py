import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights

sys.path.append(str(Path(__file__).parents[2]))

from src.data.cifar10 import get_cifar10_loaders
from src.utils.repro import set_seed
from src.utils.logger import make_run_dir
from src.utils.model_analysis import get_model_summary
from src.utils.train_utils import fit
from src.utils.constants import NUM_CLASSES_CIFAR10, CIFAR10_IMAGE_SIZE
from configs.task1_config import MODIFIED

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

def main():
    cfg = MODIFIED
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_cifar10_loaders(batch_size=cfg.batch_size)

    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    
    for block_name in cfg.blocks_to_disable:
        try:
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
            
        except Exception as e:
            print(f"Error disabling block {block_name}: {e}")
            sys.exit(1)

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
    print(f"Run directory: {run_dir}")
    
    with open(run_dir / "modified_blocks.txt", "w") as f:
        f.write(",".join(cfg.blocks_to_disable))

    config = {"epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr, "seed": cfg.seed}
    config.update(model_stats)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=cfg.epochs,
        run_dir=run_dir,
        config=config,
        save_final_name="model_modified.pt"
    )

if __name__ == "__main__":
    main()
