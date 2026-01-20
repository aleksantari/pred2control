from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class TrainConfig:
    # data
    L: int = 150
    batch_size: int = 64

    # optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # training
    max_steps: int = 10_000
    eval_every: int = 500
    loss_mode_train: str = "full"   # "full" or "last"

    # system
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
