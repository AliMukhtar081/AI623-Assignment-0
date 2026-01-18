import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights
from tqdm import tqdm
import sys
import os
import types

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.cifar10 import get_cifar10_loaders
from src.utils.repro import set_seed
from src.utils.logger import make_run_dir, log_json

# Define the modified forward method for Bottleneck (removes skip connection)
def no_skip_forward(self, x):
    # Regular forward path through convolutions
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)
    
    # The original Bottleneck does:
    # if self.downsample is not None:
    #     identity = self.downsample(x)
    # out += identity
    
    # MODIFIED: We intentionally skip the addition of identity.
    # We do NOT add the residual (skip) connection.
    
    out = self.relu(out)
    return out

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def run_epoch(model, loader, optim, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False, desc="Epoch" if train else "Eval"):
            x, y = x.to(device), y.to(device)

            if train:
                optim.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            if train:
                loss.backward()
                optim.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, y) * bs
            n += bs

    return total_loss / n, total_acc / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp", type=str, default="resnet152_modified_cifar10")
    ap.add_argument("--disable_blocks", type=str, default="", help="Comma separated list of blocks to disable skip connections e.g. layer3.0,layer4.1")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader = get_cifar10_loaders(batch_size=args.batch_size)

    print("Loading model...")
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)

    # Disable skip connections
    if args.disable_blocks:
        blocks_to_disable = args.disable_blocks.split(",")
        print(f"Disabling skip connections in: {blocks_to_disable}")
        for block_name in blocks_to_disable:
            try:
                # Expecting format like 'layerX.Y'
                parts = block_name.strip().split(".")
                if len(parts) != 2:
                     print(f"Skipping invalid block format: {block_name}")
                     continue
                
                layer_name, idx_str = parts
                if not hasattr(model, layer_name):
                    print(f"Model does not have layer: {layer_name}")
                    continue

                layer_module = getattr(model, layer_name)
                idx = int(idx_str)
                
                if idx >= len(layer_module):
                    print(f"Index {idx} out of range for {layer_name} (len={len(layer_module)})")
                    continue

                block = layer_module[idx]
                
                # Monkey-patch the forward method
                block.forward = types.MethodType(no_skip_forward, block)
                print(f" - Disabled skip in {layer_name}[{idx}]")
                
            except Exception as e:
                print(f"Error disabling block {block_name}: {e}")
                sys.exit(1)
    else:
        print("No blocks disabled. Running standard ResNet structure (baseline equivalent).")

    # Replace final layer: 1000 -> 10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

    # Freeze backbone
    print("Freezing backbone...")
    for name, p in model.named_parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    model.to(device)

    optim = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    run_dir = make_run_dir("task1/resnet", args.exp)
    print(f"Run directory: {run_dir}")
    
    metrics = {
        "config": vars(args),
        "device": device,
        "epochs": []
    }
    
    # Save information about modified blocks
    with open(run_dir / "modified_blocks.txt", "w") as f:
        f.write(args.disable_blocks or "None")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optim, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, optim, device, train=False)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc
        }
        metrics["epochs"].append(row)
        log_json(run_dir / "metrics.json", metrics)
        print(f"Epoch {epoch}: Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | Val Loss={va_loss:.4f} Acc={va_acc:.4f}")

    torch.save(model.state_dict(), run_dir / "model_modified.pt")
    print(f"Saved run to: {run_dir}")

if __name__ == "__main__":
    main()
