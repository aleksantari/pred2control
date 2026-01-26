#train.py
from typing import Dict

import torch
import torch.nn as nn

from .data import sample_batch_motorgpt_chunk
from .config import TrainConfig, set_seed

import wandb


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps # Added epsilon for numerical stability

    def forward(self, input, target):
        # Calculate MSE and take the square root
        loss = torch.sqrt(self.mse(input, target) + self.eps)
        return loss


@torch.no_grad()
def eval_teacher_forced_rmse(model, 
                             episodes, 
                             num_batches=200, 
                             batch_size=128, 
                             context_size=120, chunk_size=30, device="cuda"):
    
    model.eval()
    loss_sum = 0.0
    count = 0

    criterion = RMSELoss()

    for _ in range(num_batches):
        chunk, context = sample_batch_motorgpt_chunk(episodes, batch_size, context_size, chunk_size)
        X = context.to(device)
        Y = chunk.to(device)

        pred = model(X)

        loss_sum += criterion(pred, Y).item()
        count += 1
    return loss_sum / count



def train_motorgpt_chunk(model, train_eps, test_eps, cfg: TrainConfig) -> Dict[str, float]:

    C = cfg.context_size
    R = cfg.chunk_size
    B = cfg.batch_size

    lr = cfg.lr
    wd = cfg.weight_decay

    max_steps = cfg.max_steps
    eval_step = cfg.eval_every

    seed  = cfg.seed
    device = cfg.device

    set_seed(seed)
    model = model.to(device)
    model.train()

    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_test_rmse = float("inf")

    # --- init W&B (minimal) ---
    use_wandb = (wandb is not None)
    if use_wandb:
        run = wandb.init(
            project="pred2control",
            name=f"motor_gpt_chunk_{C}_lr{lr}_seed{seed}",
            config=cfg.__dict__,
        )
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step")
        wandb.define_metric("grad/*", step_metric="step")
        wandb.define_metric("optim/*", step_metric="step")

    try:
        for step in range(1, max_steps + 1):
            chunk, context = sample_batch_motorgpt_chunk(train_eps, batch_size=B, 
                                                         context_size=C, chunk_size=R)
            X = context.to(device)
            Y = chunk.to(device)

            pred = model(X)

            loss = criterion(pred, Y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            # log lightweight training metrics (every 10 steps)
            if use_wandb and (step % 10 == 0):
                wandb.log(
                    {
                        "step": step,
                        "train/loss": float(loss.item()),
                        "grad/clip_norm": float(clip_norm),
                        "optim/lr": float(opt.param_groups[0]["lr"]),
                    },
                    step=step,
                )

            if step % 25 == 0:
                print(
                    f"step {step:6d} | train_loss {loss:.6f}"
                )

            if step % eval_step == 0:
                # teacher-forced next-action RMSE
                test_rmse = eval_teacher_forced_rmse(model, test_eps, 
                                                     num_batches=200, batch_size=B, 
                                                     context_size=C, chunk_size=R,
                                                     device=device)
                print(f"== EVAL step {step:6d} | test_RMSE {test_rmse:.6f}")
                model.train()
                # log eval metrics
                if use_wandb:
                    wandb.log(
                        {
                            "step": step,
                            "eval/test_rmse": float(test_rmse),
                        },
                        step=step,
                    )

                # save best-by-teacher-forcing checkpoint
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    ckpt = {
                        "model_state": model.state_dict(),
                        "config": cfg.__dict__,
                        "best_test_rmse": best_test_rmse,
                    }
                    torch.save(ckpt, "motor_gpt_chunk_best.pt")
                    print(f"   saved: motor_gpt_chunk_best.pt (best_test_rmse={best_test_rmse:.6f})")
                    if use_wandb:
                        wandb.save("motor_gpt_chunk_best.pt")



        return {
            "best_test_rmse": best_test_rmse,
        }

    finally:
        if use_wandb:
            wandb.finish()
