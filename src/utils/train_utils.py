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
    optimizer: Optional[torch.optim.Optimizer], 
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
    num_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, leave=False, desc=desc or ("Train" if train else "Val"))
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            if train:
                if optimizer is None:
                    raise ValueError("Optimizer must be provided for training")
                optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy(logits, y) * batch_size
            num_samples += batch_size
            
            if train:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / num_samples, total_acc / num_samples

def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
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
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True, desc=f"Ep {epoch} Train")
        val_loss, val_acc = run_epoch(model, val_loader, None, device, train=False, desc=f"Ep {epoch} Val")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        metrics["epochs"].append(row)
        log_json(metric_path, metrics)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = run_dir / "model_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved best model ({val_acc:.4f})")
    
    final_path = run_dir / save_final_name
    torch.save(model.state_dict(), final_path)
    
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Final model: {final_path}")
    
    return metrics
