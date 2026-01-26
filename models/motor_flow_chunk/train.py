#train.py
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from .data import sample_batch_motorflow_chunk
from .config import FlowTrainConfig, set_seed

import wandb

class RectifiedFlowScheduler:
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    # sample time for the batch
    # t: (B,)
    def sample_t(self, batch_size:int, device):
        # avoiding exactly 0 or 1
        t = torch.rand(batch_size, device=device) * (1 - 2*self.eps) + self.eps
        return t
    
    #sample noise for the batch
    # noise: (B, 6)
    def sample_noise(self, shape, device):
        noise = torch.randn(shape, device=device)
        return noise
    
    # target vector field
    # xt: (B, 6)
    def bridge(self, x0, noise, t):
        # x0, z: (B, 6) or (B, K, 6) later
        # t: (B,) 
        while t.dim() < x0.dim():
            t = t.unsqueeze(1)
        xt = (1-t) * x0 +t * noise
        return xt
    
    def target_velocity(self, x0, noise):
        return noise - x0
    


@torch.no_grad()
def sample_action(model, past_actions, chunk_size=30, n_steps=10, eps=1e-5, method="euler"):
    """
    Returns a sampled action chunk x0_hat: (B, R, 6)
    """
    B = past_actions.shape[0]
    device = past_actions.device
    R = chunk_size

    # start at noise ~ t≈1
    x = torch.randn(B, R, 6, device=device)

    # time grid from (1-eps) -> eps
    s = torch.linspace(1.0, 0.0, n_steps, device=device)
    ts = eps + (1 - 2 * eps) * s  # in [eps, 1-eps]

    for i in range(n_steps - 1):
        tau1 = ts[i].expand(B)             # (B,)
        dt = ts[i + 1] - ts[i]             # scalar, negative

        v1 = model(past_actions=past_actions, noisy_actions=x, tau=tau1)  # (B,R,6)

        if method == "euler":
            x = x + dt * v1

        elif method == "huen":
            x_euler = x + dt * v1
            tau2 = ts[i + 1].expand(B)
            v2 = model(past_actions=past_actions, noisy_actions=x_euler, tau=tau2)
            x = x + dt * 0.5 * (v1 + v2)

        else:
            raise ValueError(f"Unknown method: {method}")

    return x




@torch.no_grad()
def eval_sampling_rmse(model,
                       episodes,
                       cfg: FlowTrainConfig,
                       num_batches=200,
                       batch_size=128):
    """
    Sampling eval:
    sample x_hat via Euler/Heun given past_actions, compare to x0.
    If K_eval>1, report best-of-K (oracle) RMSE.
    """
    C = cfg.context_size
    R = cfg.chunk_size
    device = cfg.device
    K = cfg.K_eval
    n_steps = cfg.flow_steps
    method = cfg.method

    model.eval()
    loss_sum = 0.0
    count = 0

    for _ in range(num_batches):
        chunk, context = sample_batch_motorflow_chunk(
            episodes, batch_size=batch_size, context_size=C, chunk_size=R
        )
        x0 = chunk.to(device)
        context = context.to(device)

        if K == 1:
            pred = sample_action(model, past_actions=context, chunk_size=R, n_steps=n_steps, method=method)
            mse = (pred - x0).pow(2).mean(dim=(1, 2))                 # (B,)
            rmse = torch.sqrt(mse + 1e-6).mean().item()              # scalar
            loss_sum += rmse
        else:
            preds = torch.stack(
                [sample_action(model, past_actions=context, chunk_size=R, n_steps=n_steps, method=method)
                 for _ in range(K)],
                dim=0
            )  # (K,B,R,6)

            mse_kb = (preds - x0.unsqueeze(0)).pow(2).mean(dim=(2, 3))   # (K,B)
            rmse_kb = torch.sqrt(mse_kb + 1e-6)                          # (K,B)
            best_rmse = rmse_kb.min(dim=0).values.mean().item()          # scalar
            loss_sum += best_rmse

        count += 1

    return loss_sum / count




def train_motorflow_chunk(model, train_eps, test_eps, cfg: FlowTrainConfig):

    C = cfg.context_size
    R = cfg.chunk_size
    B = cfg.batch_size

    lr = cfg.lr
    wd = cfg.weight_decay

    max_steps = cfg.max_steps
    eval_step = cfg.eval_every
    K = cfg.K_eval
    flow_steps = cfg.flow_steps

    seed  = cfg.seed
    device = cfg.device
    clip = cfg.grad_clip


    set_seed(seed)
    model = model.to(device)
    model.train()

    criterion = nn.MSELoss()
    scheduler = RectifiedFlowScheduler(eps=1e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_test_rmse = float("inf")

    # --- init W&B ---
    use_wandb = (wandb is not None)
    if use_wandb:
        run = wandb.init(
            project="pred2control",
            name=f"motor_flow_chunk_{C}_lr{lr}_seed{cfg.seed}",
            config=cfg.__dict__,
        )
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step")
        wandb.define_metric("grad/*", step_metric="step")
        wandb.define_metric("optim/*", step_metric="step")

    try:
        for step in range(1, max_steps + 1):
            chunk, context = sample_batch_motorflow_chunk(train_eps, batch_size=B,
                                                          context_size=C, chunk_size=R)
            context = context.to(cfg.device)
            x0 = chunk.to(cfg.device)

            tau = scheduler.sample_t(B, device=device)             # (B,)
            z = scheduler.sample_noise((B, R, 6), device=device)   # (B,R,6)
            xt  = scheduler.bridge(x0, z, tau)                     # (B,R,6)

            v_target = scheduler.target_velocity(x0, z)
            v_pred = model(past_actions=context, noisy_actions=xt, tau=tau)

            loss = criterion(v_pred, v_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()


            if step % 25 == 0:
                print(f"step {step:6d} | train_loss {loss.item():.6f}")

            # --- log train metrics every log_every steps ---
            if use_wandb and (step % 10 == 0):
                wandb.log(
                    {
                        "step": step,
                        "train/vloss": float(loss.item()),
                        "grad/clip_norm": float(clip_norm),
                        "optim/lr": float(opt.param_groups[0]["lr"]),
                    },
                    step=step,
                )

            if step % eval_step == 0:
                # teacher-forced RMSE (your existing eval)
                test_rmse = eval_sampling_rmse(model, test_eps, cfg, num_batches=200, batch_size=128)
                model.train()
                print(
                    f"== EVAL step {step:6d} | motorflow_test_RMSE {test_rmse:.6f} "
                    f"(K={K}, flow_steps={flow_steps})"
                )

                # log eval metrics (use real step)
                if use_wandb:
                    wandb.log(
                        {
                            "step": step,
                            "eval/test_rmse": float(test_rmse),
                        },
                        step=step,
                    )

                # save best-by-teacher-forced checkpoint
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    ckpt = {"model_state": model.state_dict(), "config": cfg.__dict__, "best_test_rmse": best_test_rmse}
                    torch.save(ckpt, "motor_flow_chunk_best.pt")
                    print(f"   saved: motor_flow_chunk_best.pt (best_test_rmse={best_test_rmse:.6f})")
                    if use_wandb:
                        wandb.save("motor_flow_chunk_best.pt")

        return {
            "best_test_rmse": best_test_rmse,
        }

    finally:
        if use_wandb:
            wandb.finish()
