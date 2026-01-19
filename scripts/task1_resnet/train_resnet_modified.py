
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights
import sys
import types
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from src.data.cifar10 import get_cifar10_loaders
from src.utils.repro import set_seed
from src.utils.logger import make_run_dir
from src.utils.model_analysis import print_model_analysis
from src.utils.train_utils import fit

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp", type=str, default="resnet152_modified_cifar10")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_cifar10_loaders(batch_size=args.batch_size)

    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)

    blocks_to_disable = ["layer3.0", "layer4.1"]
    
    for block_name in blocks_to_disable:
        try:
            parts = block_name.strip().split(".")
            layer_name, idx_str = parts
            
            if not hasattr(model, layer_name):
                continue

            layer_module = getattr(model, layer_name)
            idx = int(idx_str)
            
            if idx >= len(layer_module):
                continue

            block = layer_module[idx]
            
            block.forward = types.MethodType(no_skip_forward, block)
            
        except Exception as e:
            print(f"Error disabling block {block_name}: {e}")
            sys.exit(1)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

    for name, p in model.named_parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    model.to(device)

    optim = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    from src.utils.model_analysis import get_model_summary
    model_stats = get_model_summary(model, input_size=(1, 3, 32, 32), device=device)

    run_dir = make_run_dir("task1/resnet", args.exp)
    print(f"Run directory: {run_dir}")
    
    with open(run_dir / "modified_blocks.txt", "w") as f:
        f.write(",".join(blocks_to_disable))

    config = vars(args)
    config.update(model_stats)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        device=device,
        epochs=args.epochs,
        run_dir=run_dir,
        config=config,
        save_final_name="model_modified.pt"
    )

if __name__ == "__main__":
    main()
