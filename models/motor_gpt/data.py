import numpy as np
import torch


def sample_batch_motorgpt(episodes, batch_size, L=150, T=300):
    """
    episodes: list of (T,6) tensors (already normalized)
    returns X,Y each (B,L,6) where Y is shifted by 1
    """
    B = batch_size
    X = torch.empty(B, L, 6, dtype=torch.float32)
    Y = torch.empty(B, L, 6, dtype=torch.float32)

    max_s = T - (L + 1)

    for b in range(B):
        ep = episodes[np.random.randint(0, len(episodes))]
        s = np.random.randint(0, max_s + 1)
        X[b] = ep[s : s + L]
        Y[b] = ep[s + 1 : s + L + 1]
    return X, Y



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
   