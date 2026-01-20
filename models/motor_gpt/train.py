from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .data import sample_batch_motorgpt
from .config import TrainConfig, set_seed


@torch.no_grad()
def eval_teacher_forced_last_rmse(model, episodes, L=150, num_batches=200, batch_size=128, device="cuda"):
    model.eval()
    sqerr_sum = 0.0
    count = 0

    for _ in range(num_batches):
        X, Y = sample_batch_motorgpt(episodes, batch_size=batch_size, L=L)
        X = X.to(device)
        Y = Y.to(device)

        pred, _ = model(X, targets=None)
        e = pred[:, -1, :] - Y[:, -1, :]
        sqerr_sum += (e * e).sum().item()
        count += e.numel()

    return (sqerr_sum / count) ** 0.5


def train_motorgpt(model, train_eps, test_eps, cfg: TrainConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    model = model.to(cfg.device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_test_rmse = float("inf")

    for step in range(1, cfg.max_steps + 1):
        X, Y = sample_batch_motorgpt(train_eps, batch_size=cfg.batch_size, L=cfg.L)
        X = X.to(cfg.device)
        Y = Y.to(cfg.device)

        _, loss = model(X, targets=Y, loss_mode=cfg.loss_mode_train)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        with torch.no_grad():
            pred, _ = model(X, targets=None)
            last_loss = F.mse_loss(pred[:, -1, :], Y[:, -1, :]).item()
            full_loss = F.mse_loss(pred, Y).item()

        if step % 50 == 0:
            print(
                f"step {step:6d} | train_full {full_loss:.6f} | train_last {last_loss:.6f} "
                f"| train_mode={cfg.loss_mode_train}"
            )

        if step % cfg.eval_every == 0:
            test_rmse = eval_teacher_forced_last_rmse(
                model, test_eps, L=cfg.L, num_batches=200, batch_size=cfg.batch_size, device=cfg.device
            )
            print(f"== EVAL step {step:6d} | test_last_RMSE(norm) {test_rmse:.6f}")

            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                ckpt = {"model_state": model.state_dict(), "config": cfg.__dict__, "best_test_rmse": best_test_rmse}
                torch.save(ckpt, "motor_gpt_best.pt")
                print(f"   saved: motor_gpt_best.pt (best_test_rmse={best_test_rmse:.6f})")

    return {"best_test_rmse": best_test_rmse}
