
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet152

sys.path.append(str(Path(__file__).parents[2]))

from src.data.cifar100 import get_cifar100_loaders
from src.utils.repro import set_seed
from src.utils.logger import make_run_dir
from src.utils.model_analysis import get_model_summary
from src.utils.train_utils import fit
from src.utils.constants import NUM_CLASSES_CIFAR100, IMAGENET_IMAGE_SIZE
from configs.task1_config import TRANSFER_RANDOM_HEAD_ONLY

def main():
    cfg = TRANSFER_RANDOM_HEAD_ONLY
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, val_loader = get_cifar100_loaders(batch_size=cfg.batch_size)

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
        "freeze_mode": cfg.freeze_mode, "pretrained": cfg.pretrained,
        "total_params": total_params, "trainable_params": trainable_params_count
    }
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
        save_final_name="model_final.pt"
    )

if __name__ == "__main__":
    main()
