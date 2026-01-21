import numpy as np
import torch


def sample_batch_motorflow(episodes, batch_size, L=150, T=300):
    """
    episodes: list of (T,6) tensors (already normalized)
    returns:
      x0:   (B,6)   next action
      past: (B,L,6) context actions
    """
    B = batch_size
    past = torch.empty(B, L, 6, dtype=torch.float32)
    x0 = torch.empty(B, 6, dtype=torch.float32)

    max_s = T - (L + 1)

    for b in range(B):
        ep = episodes[np.random.randint(0, len(episodes))]
        s = np.random.randint(0, max_s + 1)
        past[b] = ep[s : s + L]
        x0[b] = ep[s + L]
    return x0, past

def make_fixed_rollout_valset(test_eps, num_eps=16, L=150, H=100, seed=0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(test_eps), size=min(num_eps, len(test_eps)), replace=False)
    # each item is (seed_prefix, gt_suffix)
    valset = []
    for i in idxs:
        ep = test_eps[i]              # (300,6)
        seed_prefix = ep[:L]          # (L,6)
        gt_suffix   = ep[L:L+H]       # (H,6)
        valset.append((seed_prefix, gt_suffix))
    return valset