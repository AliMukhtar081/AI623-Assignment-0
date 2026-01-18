import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.cifar10 import get_cifar10_loaders
from src.utils.repro import set_seed
from src.utils.logger import make_run_dir, log_json

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
        for x, y in tqdm(loader, leave=False):
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
    ap.add_argument("--exp", type=str, default="resnet152_head_only_cifar10")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_cifar10_loaders(batch_size=args.batch_size)

    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)

    # Replace final layer: 1000 -> 10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

    # Freeze backbone
    for name, p in model.named_parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    model.to(device)

    optim = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    run_dir = make_run_dir("task1/resnet", args.exp)
    metrics = {
        "config": vars(args),
        "device": device,
        "epochs": []
    }

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
        print(row)

    torch.save(model.state_dict(), run_dir / "model_head_only.pt")
    print(f"Saved run to: {run_dir}")

if __name__ == "__main__":
    main()
