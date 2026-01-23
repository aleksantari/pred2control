from dataclasses import dataclass
import numpy as np
import torch

from .data import sample_batch_motorflow, make_fixed_rollout_valset
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
def sample_action(model, past_actions, n_steps=10):
    # past_actions: (B,T,6) normalized
    B = past_actions.shape[0]
    device = past_actions.device

    x = torch.randn(B, 6, device=device)  # start at noise (t=1)

    # time grid from 1 -> 0
    ts = torch.linspace(1.0, 0.0, n_steps, device=device)

    for i in range(n_steps - 1):
        t = ts[i].repeat(B)               # (B,)
        dt = ts[i+1] - ts[i]              # negative
        v = model(a_noisy=x, t=t, past_actions=past_actions)  # (B,6)
        x = x + dt * v                    # Euler step

    return x  # normalized action prediction (B,6)





@torch.no_grad()
def eval_motorflow_teacher_forced_rmse(model, episodes, cfg: FlowTrainConfig, num_batches=200):
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




@torch.no_grad()
def rollout_stage5(model, seed_prefix, L=150, H=100, n_steps_sampler=10):
    """
    seed_prefix: (1,L,6) normalized
    returns: (1,L+H,6)
    """
    model.eval()
    out = seed_prefix

    for _ in range(H):
        past = out[:, -L:, :]  # (1,L,6)
        next_a = sample_action(model, past, n_steps=n_steps_sampler)  # (1,6)
        out = torch.cat([out, next_a.unsqueeze(1)], dim=1)  # append (1,1,6)
    return out



@torch.no_grad()
def rollout_suffix_rmse_stage5(
    model,
    fixed_valset,
    L=150,
    H=100,
    M=8,
    n_steps_sampler=10,
    device="cuda",
):
    """
    fixed_valset: iterable of (seed_prefix (L,6), gt_suffix (max_H,6)) or (H,6)
    Returns:
      mean_rmse over M rollouts + best_rmse (oracle) over M rollouts
    """
    model.eval()
    rmses = []
    best_rmses = []

    for seed_prefix, gt_suffix in fixed_valset:
        seed_prefix = seed_prefix.unsqueeze(0).to(device)  # (1,L,6)

        # IMPORTANT: fixed_valset may store gt_suffix at max_H (e.g., 100).
        # Slice to the requested H so shapes match.
        gt_suffix = gt_suffix[:H].to(device)  # (H,6)

        # run M stochastic rollouts
        errs = []
        for _ in range(M):
            out = rollout_stage5(model, seed_prefix, L=L, H=H, n_steps_sampler=n_steps_sampler)
            pred_suffix = out[0, -H:, :]  # (H,6)
            e = pred_suffix - gt_suffix
            errs.append((e * e).mean().item())  # MSE scalar

        # mean RMSE across rollouts
        mean_rmse = (sum(errs) / len(errs)) ** 0.5
        # best-of-M RMSE (oracle)
        best_rmse = (min(errs)) ** 0.5

        rmses.append(mean_rmse)
        best_rmses.append(best_rmse)

    return float(sum(rmses) / len(rmses)), float(sum(best_rmses) / len(best_rmses))



def train_motorflow(model, train_eps, test_eps, cfg: FlowTrainConfig):
    set_seed(cfg.seed)
    model = model.to(cfg.device)
    model.train()

    scheduler = RectifiedFlowScheduler(eps=1e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_test_rmse = float("inf")
    best_rollout_bestofM_h100 = float("inf")

    # rollout eval: store GT suffix up to max horizon you care about
    max_H = 100
    fixed_valset = make_fixed_rollout_valset(test_eps, num_eps=16, L=cfg.L, H=max_H, seed=123)

    # horizons to evaluate (add 10 if you want)
    rollout_Hs = [5, 10, 25, 50]

    # wandb logging cadence for training scalars
    log_every = getattr(cfg, "wandb_log_every", 10)  # default 10 if not in cfg

    # --- init W&B ---
    use_wandb = (wandb is not None)
    if use_wandb:
        run = wandb.init(
            project="pred2control",
            name=f"motor_flow_L{cfg.L}_lr{cfg.lr}_seed{cfg.seed}",
            config=cfg.__dict__,
        )
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("eval/*", step_metric="step")
        wandb.define_metric("grad/*", step_metric="step")
        wandb.define_metric("optim/*", step_metric="step")

    try:
        for step in range(1, cfg.max_steps + 1):
            x0, past = sample_batch_motorflow(train_eps, batch_size=cfg.batch_size, L=cfg.L)
            x0 = x0.to(cfg.device)
            past = past.to(cfg.device)

            B = x0.shape[0]
            t = scheduler.sample_t(B, device=cfg.device)
            z = scheduler.sample_noise(x0.shape, device=cfg.device)
            xt = scheduler.bridge(x0, z, t)
            v_target = scheduler.target_velocity(x0, z)

            v_pred = model(a_noisy=xt, t=t, past_actions=past)
            loss = torch.mean((v_pred - v_target) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()


            if step % 50 == 0:
                print(f"step {step:6d} | train_vloss {loss.item():.6f}")

            # --- log train metrics every log_every steps ---
            if use_wandb and (step % log_every == 0):
                wandb.log(
                    {
                        "step": step,
                        "train/vloss": float(loss.item()),
                        "grad/clip_norm": float(clip_norm),
                        "optim/lr": float(opt.param_groups[0]["lr"]),
                    },
                    step=step,
                )

            if step % cfg.eval_every == 0:
                # teacher-forced RMSE (your existing eval)
                test_rmse = eval_motorflow_teacher_forced_rmse(model, test_eps, cfg, num_batches=200)
                model.train()
                print(
                    f"== EVAL step {step:6d} | motorflow_test_RMSE(norm) {test_rmse:.6f} "
                    f"(K={cfg.K_eval}, n_steps={cfg.n_steps_sample_eval})"
                )

                # rollout eval at multiple horizons
                rollout_metrics = {}
                for H in rollout_Hs:
                    mean_rmse, best_rmse = rollout_suffix_rmse_stage5(
                        model,
                        fixed_valset,
                        L=cfg.L,
                        H=H,
                        M=cfg.K_eval if hasattr(cfg, "K_eval") else 8,
                        n_steps_sampler=cfg.n_steps_sample_eval,
                        device=cfg.device,
                    )
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
