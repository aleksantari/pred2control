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
