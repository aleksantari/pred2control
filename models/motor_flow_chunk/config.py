#config.py
from dataclasses import dataclass
import torch
import numpy as np


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class FlowTrainConfig:
    # data
    context_size: int = 120
    chunk_size: int = 30
    batch_size: int = 64

    # optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # training
    max_steps: int = 5_000
    eval_every: int = 500

    # system
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 67

    # eval/sampling
    flow_steps: int = 10
    K_eval: int = 4
    method: str = "euler"