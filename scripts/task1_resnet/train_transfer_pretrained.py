
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from src.data.cifar100 import get_cifar100_loaders
from src.utils.repro import set_seed
from src.utils.logger import make_run_dir
from src.utils.model_analysis import print_model_analysis
from src.utils.train_utils import fit

def main():
    ap = argparse.ArgumentParser(description="Fine-tune ResNet-152 on CIFAR-100 with ImageNet-pretrained weights")
    ap.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--freeze_mode", type=str, default="head_only", 
                    choices=["head_only", "final_block", "full"],
                    help="What to train: head_only, final_block (layer4), or full backbone")
    ap.add_argument("--exp", type=str, default=None, help="Experiment name (auto-generated if not provided)")
    args = ap.parse_args()

    if args.exp is None:
        args.exp = f"resnet152_pretrained_{args.freeze_mode}_cifar100"

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_cifar100_loaders(batch_size=args.batch_size)

    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 100)

    if args.freeze_mode == "head_only":
        for name, p in model.named_parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        trainable_params = model.fc.parameters()
        
    elif args.freeze_mode == "final_block":
        for name, p in model.named_parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
        trainable_params = list(model.layer4.parameters()) + list(model.fc.parameters())
        
    elif args.freeze_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        trainable_params = model.parameters()

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optim = torch.optim.Adam(trainable_params, lr=args.lr)

    from src.utils.model_analysis import get_model_summary
    model_stats = get_model_summary(model, input_size=(1, 3, 224, 224), device=device)

    run_dir = make_run_dir("task1/transfer_learning", args.exp)
    
    config = vars(args)
    config.update({
        "pretrained": True,
        "total_params": total_params,
        "trainable_params": trainable_params_count
    })
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
        save_final_name="model_final.pt"
    )

if __name__ == "__main__":
    main()
