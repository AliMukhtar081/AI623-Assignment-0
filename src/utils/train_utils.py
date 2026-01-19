
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from src.utils.logger import log_json, make_run_dir
from typing import Dict, Any, Optional

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def run_epoch(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    optim: Optional[torch.optim.Optimizer], 
    device: str, 
    train: bool,
    desc: str = ""
) -> tuple[float, float]:
    
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
        pbar = tqdm(loader, leave=False, desc=desc or ("Train" if train else "Val"))
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            if train:
                if optim is None:
                    raise ValueError("Optimizer must be provided for training")
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
            
            if train:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n, total_acc / n

def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    device: str,
    epochs: int,
    run_dir: Path,
    config: Dict[str, Any] = None,
    save_final_name: str = "model_final.pt"
) -> Dict[str, Any]:
    
    metric_path = run_dir / "metrics.json"
    
    metrics = {
        "config": config or {},
        "device": str(device),
        "epochs": []
    }
    
    best_val_acc = 0.0
    
    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optim, device, train=True, desc=f"Ep {epoch} Train")
        va_loss, va_acc = run_epoch(model, val_loader, None, device, train=False, desc=f"Ep {epoch} Val")

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc
        }
        metrics["epochs"].append(row)
        log_json(metric_path, metrics)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            checkpoint_path = run_dir / "model_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved best model ({va_acc:.4f})")
    
    final_path = run_dir / save_final_name
    torch.save(model.state_dict(), final_path)
    
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Final model: {final_path}")
    
    return metrics
