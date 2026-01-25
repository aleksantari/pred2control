import numpy as np
import torch


def sample_batch_motorgpt_chunk(episodes, batch_size, context_size=120, chunk_size=30, device=None):
    """
    episodes: list of torch.Tensor, each (T_i, 6) float32 (already normalized)
    returns:
      chunk: (B, R, 6)
      past:  (B, C, 6)
    """
    B = batch_size
    C = context_size
    R = chunk_size

    past  = torch.empty(B, C, 6, dtype=torch.float32, device=device)
    chunk = torch.empty(B, R, 6, dtype=torch.float32, device=device)

    for b in range(B):
        ep = episodes[np.random.randint(0, len(episodes))] # sample an episode from list
        assert torch.is_tensor(ep) and ep.ndim == 2 and ep.shape[1] == 6

        T_ep = ep.shape[0]
        max_s = T_ep - (C + R)
        assert max_s >= 0, f"Episode too short: T={T_ep}, need at least C+R={C+R}"

        s = np.random.randint(0, max_s + 1) # sample a random start within valid window

        window = ep[s : s + C + R]  # (C+R, 6)
        past[b]  = window[:C]
        chunk[b] = window[C:]
    return chunk, past
