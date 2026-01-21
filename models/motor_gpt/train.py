from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .data import sample_batch_motorgpt, make_fixed_rollout_valset
from .config import TrainConfig, set_seed

import wandb



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



@torch.no_grad()
def rollout_suffix_rmse_stage2(model, fixed_valset, H=100, device="cuda"):
    model.eval()
    sqerr_sum = 0.0
    count = 0

    for seed_prefix, gt_suffix in fixed_valset:
        seed_prefix = seed_prefix.unsqueeze(0).to(device)  # (1,L,6)

        # FIX: fixed_valset may store gt_suffix at max_H (e.g., 100).
        # For smaller H evals (25/50), slice to match.
        gt_suffix = gt_suffix[:H].to(device)               # (H,6)

        out = model.generate(seed_prefix, max_new_steps=H) # (1,L+H,6)
        pred_suffix = out[0, -H:, :]                       # (H,6)

        e = pred_suffix - gt_suffix
        sqerr_sum += (e * e).sum().item()
        count += e.numel()

    return (sqerr_sum / count) ** 0.5




def train_motorgpt(model, train_eps, test_eps, cfg: TrainConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    model = model.to(cfg.device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_test_rmse = float("inf")
    best_rollout_rmse_h100 = float("inf")

    # for roll out eval (store GT suffix up to max horizon you care about)
    max_H = 100
    fixed_valset = make_fixed_rollout_valset(test_eps, num_eps=16, L=cfg.L, H=max_H, seed=123)

    # horizons to log (add 10 if you want)
    rollout_Hs = [25, 50, 100]

    # --- init W&B (minimal) ---
    use_wandb = (wandb is not None)
    if use_wandb:
        run = wandb.init(
            project="pred2control",
            name=f"motor_gpt_L{cfg.L}_lr{cfg.lr}_seed{cfg.seed}",
            config=cfg.__dict__,
        )
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step")
        wandb.define_metric("grad/*", step_metric="step")
        wandb.define_metric("optim/*", step_metric="step")

    try:
        for step in range(1, cfg.max_steps + 1):
            X, Y = sample_batch_motorgpt(train_eps, batch_size=cfg.batch_size, L=cfg.L)
            X = X.to(cfg.device)
            Y = Y.to(cfg.device)

            _, loss = model(X, targets=Y, loss_mode=cfg.loss_mode_train)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            with torch.no_grad():
                pred, _ = model(X, targets=None)
                last_loss = F.mse_loss(pred[:, -1, :], Y[:, -1, :]).item()
                full_loss = F.mse_loss(pred, Y).item()

            # log lightweight training metrics (every step 10 steps)
            if use_wandb and (step % 10 == 0):
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/full_mse": float(full_loss),
                        "train/last_mse": float(last_loss),
                        "grad/clip_norm": float(clip_norm),
                        "optim/lr": float(opt.param_groups[0]["lr"]),
                    }
                )

            if step % 50 == 0:
                print(
                    f"step {step:6d} | train_full {full_loss:.6f} | train_last {last_loss:.6f} "
                    f"| train_mode={cfg.loss_mode_train}"
                )

            if step % cfg.eval_every == 0:
                # teacher-forced next-action RMSE
                test_rmse = eval_teacher_forced_last_rmse(
                    model, test_eps, L=cfg.L, num_batches=200, batch_size=cfg.batch_size, device=cfg.device
                )
                print(f"== EVAL step {step:6d} | test_last_RMSE(norm) {test_rmse:.6f}")

                # rollout RMSE at multiple horizons (H-sweep)
                rollout_metrics: Dict[str, float] = {}
                for H in rollout_Hs:
                    r = rollout_suffix_rmse_stage2(model, fixed_valset, H=H, device=cfg.device)
                    rollout_metrics[f"eval/rollout_suffix_rmse_h{H}"] = float(r)
                    print(f"== rollout_suffix_RMSE@H={H:<3d} {r:.6f}")

                # log eval metrics
                if use_wandb:
                    wandb.log(
                        {
                            "step": step,
                            "eval/test_last_rmse": float(test_rmse),
                            **rollout_metrics,
                        }
                    )

                # save best-by-teacher-forcing checkpoint
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    ckpt = {
                        "model_state": model.state_dict(),
                        "config": cfg.__dict__,
                        "best_test_rmse": best_test_rmse,
                    }
                    torch.save(ckpt, "motor_gpt_best.pt")
                    print(f"   saved: motor_gpt_best.pt (best_test_rmse={best_test_rmse:.6f})")
                    if use_wandb:
                        wandb.save("motor_gpt_best.pt")

                # save best-by-rollout checkpoint (use H=100 as the primary long-horizon metric)
                roll100 = rollout_metrics.get("eval/rollout_suffix_rmse_h100", None)
                if roll100 is not None and roll100 < best_rollout_rmse_h100:
                    best_rollout_rmse_h100 = roll100
                    ckpt = {
                        "model_state": model.state_dict(),
                        "config": cfg.__dict__,
                        "best_rollout_rmse_h100": best_rollout_rmse_h100,
                    }
                    torch.save(ckpt, "motor_gpt_best_rollout.pt")
                    print(
                        f"   saved: motor_gpt_best_rollout.pt "
                        f"(best_rollout_rmse_h100={best_rollout_rmse_h100:.6f})"
                    )
                    if use_wandb:
                        wandb.save("motor_gpt_best_rollout.pt")

        return {
            "best_test_rmse": best_test_rmse,
            "best_rollout_rmse_h100": best_rollout_rmse_h100,
        }

    finally:
        if use_wandb:
            wandb.finish()


