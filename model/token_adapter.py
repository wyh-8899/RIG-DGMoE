import torch
import torch.nn as nn


class STTokenAdapter(nn.Module):
    """
    hx_enc/he_enc: [B, N, d_in]
    -> hx_tok/he_tok: [B, N, d_llm]
    """
    def __init__(self, d_in: int, d_llm: int, dropout: float = 0.1):
        super().__init__()
        self.proj_x = nn.Sequential(
            nn.Linear(d_in, d_llm),
            nn.LayerNorm(d_llm),
            nn.Dropout(dropout),
        )
        self.proj_e = nn.Sequential(
            nn.Linear(d_in, d_llm),
            nn.LayerNorm(d_llm),
            nn.Dropout(dropout),
        )

        self.mod_x = nn.Parameter(torch.zeros(1, 1, d_llm))
        self.mod_e = nn.Parameter(torch.zeros(1, 1, d_llm))

    def forward(self, hx_enc: torch.Tensor, he_enc: torch.Tensor):
        hx_tok = self.proj_x(hx_enc) + self.mod_x
        he_tok = self.proj_e(he_enc) + self.mod_e
        return hx_tok, he_tok

class LatentQueryCompressor(nn.Module):
    """
    learned latent queries:
      [B, N, D] -> [B, K, D]
    """
    def __init__(self, d_model: int, num_latents: int = 32, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # x: [B, N, D]
        B = x.size(0)
        q = self.latents.expand(B, -1, -1)          # [B, K, D]
        out, _ = self.cross_attn(query=q, key=x, value=x)
        return self.ln(q + out)


class DualTokenCompressor(nn.Module):
    def __init__(self, d_model: int, num_latents_x: int = 32, num_latents_e: int = 32, num_heads: int = 4):
        super().__init__()
        self.comp_x = LatentQueryCompressor(d_model, num_latents_x, num_heads)
        self.comp_e = LatentQueryCompressor(d_model, num_latents_e, num_heads)

    def forward(self, hx_tok: torch.Tensor, he_tok: torch.Tensor):
        z_x = self.comp_x(hx_tok)   # [B, Kx, D]
        z_e = self.comp_e(he_tok)   # [B, Ke, D]
        return z_x, z_e