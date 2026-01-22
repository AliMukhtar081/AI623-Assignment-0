from dataclasses import dataclass
from typing import Literal

@dataclass
class BaseConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42

@dataclass
class BaselineConfig(BaseConfig):
    exp: str = "resnet152_head_only_cifar10"

@dataclass
class ModifiedConfig(BaseConfig):
    exp: str = "resnet152_modified_cifar10"
    blocks_to_disable: list[str] = None
    
    def __post_init__(self):
        if self.blocks_to_disable is None:
            self.blocks_to_disable = ["layer3.0", "layer4.1"]

@dataclass
class TransferConfig(BaseConfig):
    epochs: int = 50
    freeze_mode: Literal["head_only", "final_block", "full"] = "head_only"
    pretrained: bool = True
    
    def get_exp_name(self) -> str:
        prefix = "pretrained" if self.pretrained else "random"
        return f"resnet152_{prefix}_{self.freeze_mode}_cifar100"

BASELINE = BaselineConfig()

MODIFIED = ModifiedConfig()

TRANSFER_PRETRAINED_HEAD_ONLY = TransferConfig(
    freeze_mode="head_only",
    pretrained=True
)

TRANSFER_PRETRAINED_FINAL_BLOCK = TransferConfig(
    freeze_mode="final_block", 
    pretrained=True
)

TRANSFER_PRETRAINED_FULL = TransferConfig(
    freeze_mode="full",
    pretrained=True
)

TRANSFER_RANDOM_HEAD_ONLY = TransferConfig(
    freeze_mode="head_only",
    pretrained=False
)

TRANSFER_RANDOM_FINAL_BLOCK = TransferConfig(
    freeze_mode="final_block",
    pretrained=False
)

TRANSFER_RANDOM_FULL = TransferConfig(
    freeze_mode="full",
    pretrained=False
)
