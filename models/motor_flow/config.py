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
    L: int = 150
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 30_000
    eval_every: int = 500
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # eval/sampling
    n_steps_sample_eval: int = 10
    K_eval: int = 1