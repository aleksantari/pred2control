"""
model.py (Torch env)

Motor-GPT: a NanoGPT-style causal decoder adapted for continuous actions.
Key differences from text GPT:
- Input "embedding": Linear(action_dim=6 -> d_model)
- Output head: Linear(d_model -> action_dim=6)
- Loss: regression (MSE or SmoothL1) on next-action prediction
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import math




class SelfAttentionHead(nn.Module):
    def __init__(self, head_dim, embed_dim, traj_size, dropout=0.0):
        super().__init__()
        # linear projections for Q, V, K
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        mask = torch.tril(torch.ones(traj_size, traj_size)).view(1, traj_size, traj_size)
        self.atten_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) #(B, T, H)
        k = self.key(x)
        v = self.value(x)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, T, T)
        att = att.masked_fill(self.mask[:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)  # (B, T, T)
        att = self.atten_drop(att)

        out = att @ v  # (B, T, H)
        out = self.resid_drop(out)

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, head_dim, traj_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_dim, embed_dim, traj_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        multi_head_out = [h(x) for h in self.heads]  # list of (B, T, head_size)
        multi_head_concat = torch.cat(multi_head_out, dim=-1) # (B, T, num_heads * head_size)

        out = self.drop(self.proj(multi_head_concat))  # (B, T, embed_dim)

        return out
    

class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, expansion*embed_dim),
            nn.GELU(),
            nn.Linear(expansion*embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_dim, n_head, block_size, mlp_expansion=4, dropout=0.0):
        super().__init__()
        assert embed_dim % n_head == 0
        head_size = embed_dim // n_head
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(n_head, embed_dim, head_size, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, expansion=mlp_expansion, dropout=dropout)

    def forward(self, x):
        # TODO
        x = x + self.attn(self.ln1(x)) # skip connection
        x = x + self.mlp(self.ln2(x)) # skip connection

        return x
    




class MotorGPT(nn.Module):
    def __init__(self, action_size=6, embed_dim=192, traj_size=128, n_layer=4, n_head=4, dropout=0.0):
        super().__init__()
        self.action_size = action_size
        self.traj_size = traj_size

        # “token embedding” for continuous actions
        self.token_emb = nn.Linear(action_size, embed_dim, bias=False)
        self.pos_emb   = nn.Embedding(traj_size, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_head, traj_size, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

        # regression head back to action space
        self.head = nn.Linear(embed_dim, action_size, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # NanoGPT init is fine conceptually, but add LayerNorm explicitly
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, targets=None, loss_mode="full"):
        """
        x:       (B, T, A)  normalized actions
        targets: (B, T, A)  next-step normalized actions
        """
        B, T, A = x.shape
        assert T <= self.traj_size, "Sequence length exceeds traj_size"


        #token and position embedding
        tok = self.token_emb(x)  # (B, T, E)
        pos = self.pos_emb(torch.arange(T, device=x.device)).unsqueeze(0)  # (1, T, E)
        h = tok + pos

        #pass through Transformer blocks
        for block in self.blocks:
            h = block(h)

        # regression head 
        h = self.ln_f(h)
        pred = self.head(h)      # (B, T, A)

        loss = None
        if targets is not None:
            if loss_mode == "full":
                loss = F.mse_loss(pred, targets)
            elif loss_mode == "last":
                loss = F.mse_loss(pred[:, -1, :], targets[:, -1, :])
            else:
                raise ValueError("loss_mode must be 'full' or 'last'")

        return pred, loss

    @torch.no_grad()
    def generate(self, seed_actions, max_new_steps=100, noise_std=0.0):
        """
        seed_actions: (B, T0, A) normalized actions
        returns:      (B, T0+max_new_steps, A) normalized actions
        """
        self.eval()
        out = seed_actions
        for _ in range(max_new_steps):
            cond = out[:, -self.traj_size:, :]      # crop context, so only last traj_size steps
            pred, _ = self(cond)                    # (B, Tcond, A), this includes the whole context
            next_a = pred[:, -1, :]                 # (B, A), take only the last time step

            # optional stochasticity (since MSE tends to be “average”)
            if noise_std > 0:
                next_a = next_a + noise_std * torch.randn_like(next_a)

            out = torch.cat([out, next_a.unsqueeze(1)], dim=1)
        return out
