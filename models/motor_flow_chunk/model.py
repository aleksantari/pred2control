#model.py
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
import numpy as np


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) float in [0,1] or (B,) int steps cast to float
        returns: (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
        )
        # (B, half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb



class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)



class AdaRMSNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.to_scale = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim, bias=True),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        cond: (B, cond_dim)
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        scale = self.to_scale(cond).unsqueeze(1)  # (B, 1, D)
        return x_norm * (self.weight + scale)




class AttentionHead(nn.Module):
    """ Single Attention Head: for Self-Attention or Cross-Attention
    Args:
        head_dim: dimension of each attention head
        embed_dim: input embedding dimension
        dropout: dropout rate
        causal: whether to apply causal masking
        max_len: maximum sequence (context) length (required if causal is True) 
      
    Returns:
        out: output tensor after attention (B, Tq, head_dim)
    
     """
    def __init__(self, head_dim, embed_dim, dropout=0.0, causal=False, max_len=None):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        self.atten_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.causal = causal
        if causal:
            assert max_len is not None, "max_len required for causal mask"
            mask = torch.tril(torch.ones(max_len, max_len)).view(1, max_len, max_len)
            self.register_buffer("mask", mask)

    def forward(self, x_q, x_kv=None):
        # x_q: (B, Tq, E), x_kv: (B, Tk, E)
        if x_kv is None:
            x_kv = x_q

        B, Tq, _ = x_q.shape
        Tk = x_kv.shape[1]

        q = self.query(x_q)
        k = self.key(x_kv)
        v = self.value(x_kv)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.causal:
            att = att.masked_fill(self.mask[:, :Tq, :Tk] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.atten_drop(att)

        out = att @ v
        out = self.resid_drop(out)
        return out
    



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, head_dim, dropout=0.0, causal=False, max_len=None):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_dim, embed_dim, dropout, causal, max_len) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_kv=None):
        multi_head_out = [h(x, x_kv) for h in self.heads]  # list of (B, T, head_size)
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
    def forward(self, x): 
        return self.net(x)


# Transformer Block with Self-Attention, Cross-Attention, and Feed-Forward Network
class Block(nn.Module):
    def __init__(self, embed_dim, tau_dim, n_head, mlp_expansion=4, dropout=0.0):
        super().__init__()
        assert embed_dim % n_head == 0
        head_size = embed_dim // n_head

        self.ada_norm1 = AdaRMSNorm(embed_dim, tau_dim)
        self.ada_norm2 = AdaRMSNorm(embed_dim, tau_dim)
        self.ada_norm3 = AdaRMSNorm(embed_dim, tau_dim)
        self.rms_norm = nn.RMSNorm(embed_dim)


        self.self_attn = MultiHeadAttention(n_head, embed_dim, head_size, dropout)
        self.cross_attn = MultiHeadAttention(n_head, embed_dim, head_size, dropout)
        self.mlp = FeedForward(embed_dim, expansion=mlp_expansion, dropout=dropout)

    def forward(self, x, x_kv, tau):
      
        x = x + self.self_attn(self.ada_norm1(x, tau))
        x = x + self.cross_attn(self.ada_norm2(x, tau), self.rms_norm(x_kv))
        x = x + self.mlp(self.ada_norm3(x, tau)) 
        return x
    


class ContextBlock(nn.Module):
    """
    A standard transformer encoder block for the context sequence only.
    No timestep conditioning is needed here (context is "past actions", not noisy).
    """
    def __init__(self, embed_dim, n_head, mlp_expansion=2, dropout=0.0):
        super().__init__()
        assert embed_dim % n_head == 0
        head_dim = embed_dim // n_head

        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(n_head, embed_dim, head_dim, dropout)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, expansion=mlp_expansion, dropout=dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



def init_weights(module: nn.Module):
    # --- default linear/embedding/layernorm init ---
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    # --- RMSNorm weights to 1 (PyTorch RMSNorm has weight only) ---
    if isinstance(module, nn.RMSNorm):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)

    # --- AdaRMSNorm: keep base weight=1; make conditioning start at "no effect" ---
    if module.__class__.__name__ == "AdaRMSNorm":
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)

        # last linear in to_scale -> zeros => scale(t) starts at 0
        last = module.to_scale[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    # --- TimeMLP: last layer zeros so t-conditioning starts neutral ---
    if isinstance(module, TimeMLP):
        last = module.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)



class MotorFlow_chunk(nn.Module):
    def __init__(self, embed_dim=128, tau_dim=64, 
                 n_layer=4, n_head=4, dropout=0.0,
                 context_size=150, chunk_size=30, 
                 context_layers=2):
        super().__init__()

        self.embed_dim = embed_dim
        self.chunk_size = chunk_size

        self.action_in = nn.Linear(6, embed_dim)
        self.action_out = nn.Linear(embed_dim, 6)

        # pos emb for both sequences
        self.pos_emb = nn.Embedding(context_size + chunk_size, embed_dim)

        # tau embedding
        self.tau_embed = SinusoidalTimeEmbedding(tau_dim)
        self.tau_mlp = TimeMLP(tau_dim, embed_dim)

        # context encoder
        self.context_blocks = nn.ModuleList([
            ContextBlock(embed_dim, n_head, dropout=dropout)
            for _ in range(context_layers)
        ])
        self.context_norm = nn.RMSNorm(embed_dim)

        # decoder blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, tau_dim, n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.norm = nn.RMSNorm(embed_dim)

        self.apply(init_weights)
        nn.init.normal_(self.action_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, past_actions: torch.Tensor, noisy_actions: torch.Tensor, tau: int ) -> torch.Tensor:
        """
        past_actions: (B, C, 6)
        noisey_actions: (B, R, 6)
        returns:      (B, R, 6)
        """
        R = self.chunk_size

        B, C, _ = past_actions.shape
        assert R <= self.chunk_size, f"R={R} > model chunk_size={self.chunk_size}"
        assert (C + R) <= self.pos_emb.num_embeddings, "pos_emb too small for C+R"

        # ----- encode context -----
        ctx = self.action_in(past_actions)  # (B,C,D)
        ctx_pos = self.pos_emb(torch.arange(C, device=past_actions.device)).unsqueeze(0)  # (1,C,D)
        ctx = ctx + ctx_pos

        for blk in self.context_blocks:
            ctx = blk(ctx)
        ctx = self.context_norm(ctx)

        # ----- build query tokens -----
        q_pos = self.pos_emb(torch.arange(C, C + R, device=past_actions.device)).unsqueeze(0)  # (1,R,D)
        q = self.action_in(noisy_actions)
        q = q + q_pos  # (B,R,D)

        # tau embeddings
        tau = self.tau_mlp(self.tau_embed(tau))

        # ----- decode -----
        for blk in self.blocks:
            q = blk(q, ctx, tau)

        q = self.norm(q)
        out = self.action_out(q)  # (B,R,6)
        return out
