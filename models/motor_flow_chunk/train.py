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
def sample_action(model, past_actions, n_steps=10, eps=1e-5, method="euler"):
    B = past_actions.shape[0]
    device = past_actions.device

    x = torch.randn(B, 6, device=device)  # start at noise ~ t≈1

    # time grid from (1-eps) -> eps, 
    s = torch.linspace(1.0, 0.0, n_steps, device=device)

    ts = eps + (1 - 2*eps) * s  # in [eps, 1-eps]

    for i in range(n_steps - 1):
        t1 = ts[i].expand(B)
        dt = ts[i + 1] - ts[i]  # negative

        v1 = model(a_noisy=x, t=t1, past_actions=past_actions)

        if method == "euler":
            x = x + dt * v1
        if method == "huen":  
            x_euler = x + dt * v1
            t2 = ts[i + 1].expand(B)
            v2 = model(a_noisy=x_euler, t=t2, past_actions=past_actions)
            x = x + dt * 0.5 * (v1 + v2)

    return x



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
                             context_size=120, chunk_size=30, cfg: FlowTrainConfig):
    """
    Teacher-forced next-action eval:
    sample x_hat given past_actions, compare to x0.
    If K_eval>1, report best-of-K (oracle) RMSE.
    """
    model.eval()
    sqerr_sum = 0.0
    count = 0

    for _ in range(num_batches):
        x0, past = sample_batch_motorflow(episodes, batch_size=cfg.batch_size, L=cfg.L)
        x0 = x0.to(cfg.device)
        past = past.to(cfg.device)

        if cfg.K_eval == 1:
            x_hat = sample_action(model, past, n_steps=cfg.n_steps_sample_eval)
            err2 = ((x_hat - x0) ** 2).sum(dim=-1)  # (B,)
        else:
            samples = [sample_action(model, past, n_steps=cfg.n_steps_sample_eval) for _ in range(cfg.K_eval)]
            samples = torch.stack(samples, dim=0)  # (K,B,6)
            err2 = ((samples - x0.unsqueeze(0)) ** 2).sum(dim=-1).min(dim=0).values  # (B,)

        sqerr_sum += err2.sum().item()
        count += x0.numel()


    return (sqerr_sum / count) ** 0.5






def train_motorflow(model, train_eps, test_eps, cfg: FlowTrainConfig):

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

            tau = scheduler.sample_t(B, device=device)
            z = scheduler.sample_noise(B, device=device)
            xt = scheduler.bridge(x0, z, tau)

            v_target = scheduler.target_velocity(x0, z)
            v_pred = model(past_actions=context, noisy_actions=xt, tau=tau)

            loss = criterion(v_pred, v_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
                test_rmse = eval_motorflow_teacher_forced_rmse(model, test_eps, cfg, num_batches=200)
                model.train()
                print(
                    f"== EVAL step {step:6d} | motorflow_test_RMSE(norm) {test_rmse:.6f} "
                    f"(K={cfg.K_eval}, n_steps={cfg.n_steps_sample_eval})"
                )

                # rollout eval at multiple horizons
                rollout_metrics = {}
             
                    model.train()
                    rollout_metrics[f"eval/rollout_mean_rmse_h{H}"] = float(mean_rmse)
                    rollout_metrics[f"eval/rollout_bestofM_rmse_h{H}"] = float(best_rmse)
                    print(f"== rollout_suffix_RMSE@H={H:<3d} mean {mean_rmse:.6f} | bestofM {best_rmse:.6f}")

                # log eval metrics (use real step)
                if use_wandb:
                    wandb.log(
                        {
                            "step": step,
                            "eval/test_rmse": float(test_rmse),
                            **rollout_metrics,
                        },
                        step=step,
                    )

                # save best-by-teacher-forced checkpoint
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    ckpt = {"model_state": model.state_dict(), "config": cfg.__dict__, "best_test_rmse": best_test_rmse}
                    torch.save(ckpt, "motor_flow_best.pt")
                    print(f"   saved: motor_flow_best.pt (best_test_rmse={best_test_rmse:.6f})")
                    if use_wandb:
                        wandb.save("motor_flow_best.pt")

                # save best-by-rollout checkpoint (use best-of-M at H=100 as primary)
                bestof_h100 = rollout_metrics.get("eval/rollout_bestofM_rmse_h100", None)
                if bestof_h100 is not None and bestof_h100 < best_rollout_bestofM_h100:
                    best_rollout_bestofM_h100 = bestof_h100
                    ckpt = {
                        "model_state": model.state_dict(),
                        "config": cfg.__dict__,
                        "best_rollout_bestofM_rmse_h100": best_rollout_bestofM_h100,
                    }
                    torch.save(ckpt, "motor_flow_best_rollout.pt")
                    print(
                        f"   saved: motor_flow_best_rollout.pt "
                        f"(best_rollout_bestofM_rmse_h100={best_rollout_bestofM_h100:.6f})"
                    )
                    if use_wandb:
                        wandb.save("motor_flow_best_rollout.pt")

        return {
            "best_test_rmse": best_test_rmse,
            "best_rollout_bestofM_rmse_h100": best_rollout_bestofM_h100,
        }

    finally:
        if use_wandb:
            wandb.finish()
