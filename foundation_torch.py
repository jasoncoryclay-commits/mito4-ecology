#!/usr/bin/env python3
"""
foundation_torch.py — reconstruct + load the REAL FOUNDATION transformers.

Built directly from the checkpoint inspection (foundation_probe.py) of the models
in /workspace/foundation_model:

ENCODER  phase2_text_to_grid/best_text_to_grid.pt  (state_dict @ 'model_state_dict', 92 tensors)
    text_proj: Linear(3072->768) -> LayerNorm -> Linear(768->768)
    pos_embed: Embedding(128, 768)
    input_norm: LayerNorm(768)
    12 x Block { norm1, attn.qkv(768->2304), attn.out_proj(768->768),
                 norm2, ffn.w1(768->3072), ffn.w2(3072->768), ffn.w3(768->3072) }  # SwiGLU
    output_norm: LayerNorm(768)
    output_proj: Linear(768->768) -> (GELU) -> Linear(768->400)   # -> 20x20 grid

DECODER  phase3_grid_to_text_v3/best_grid_to_text.pt (state_dict @ 'model_state_dict', 96 tensors)
    config: d_model=768 n_layers=12 n_heads=12 d_ff=3072 d_grid=400 d_text=3072
            max_seq=128 loss=contrastive_infonce
    grid_proj: Linear(400->768) -> LayerNorm -> Linear(768->768)
    12 x Block { ... ff.w1/w2/w3 ... }
    text_out_norm + text_out_proj: Linear(768->768) -> Linear(768->3072)   # -> text EMBEDDING

IMPORTANT: this decoder outputs a 3072-dim TEXT EMBEDDING (contrastive), not tokens.
"grid -> text" = grid -> embedding -> nearest concept in an embedding bank.

The encoder likewise consumes a 3072-dim TEXT EMBEDDING, not a raw string. The
text->3072 embedder is a SEPARATE upstream component; pass one in, or operate in
grid-space. Everything here is inference-only (no training, no weight writes).

Requires torch (present on the pod). Import-safe if torch is missing.
"""
from __future__ import annotations
import os, json
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:
    _TORCH = False

D_MODEL, N_LAYERS, N_HEADS, D_FF = 768, 12, 12, 3072
D_GRID, D_TEXT, MAX_SEQ = 400, 3072, 128


if _TORCH:
    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
            self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
            self.nh = N_HEADS

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            o = F.scaled_dot_product_attention(q, k, v)
            o = o.transpose(1, 2).reshape(B, T, C)
            return self.out_proj(o)

    class _SwiGLU(nn.Module):
        # ffn.w1,w3: 768->3072 ; ffn.w2: 3072->768   ->  w2( silu(w1 x) * w3 x )
        def __init__(self, w_prefix="ffn"):
            super().__init__()
            self.w1 = nn.Linear(D_MODEL, D_FF, bias=False)
            self.w2 = nn.Linear(D_FF, D_MODEL, bias=False)
            self.w3 = nn.Linear(D_MODEL, D_FF, bias=False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class _Block(nn.Module):
        def __init__(self, ffn_name="ffn"):
            super().__init__()
            self.norm1 = nn.LayerNorm(D_MODEL)
            self.attn = _Attn()
            self.norm2 = nn.LayerNorm(D_MODEL)
            # encoder uses .ffn, decoder uses .ff — set attribute name to match keys
            setattr(self, ffn_name, _SwiGLU())
            self._ffn_name = ffn_name

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + getattr(self, self._ffn_name)(self.norm2(x))
            return x

    class FoundationEncoder(nn.Module):
        """text_embedding(3072) -> grid(400)."""
        def __init__(self):
            super().__init__()
            self.text_proj = nn.Sequential(
                nn.Linear(D_TEXT, D_MODEL), nn.LayerNorm(D_MODEL),
                nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
            self.pos_embed = nn.Embedding(MAX_SEQ, D_MODEL)
            self.input_norm = nn.LayerNorm(D_MODEL)
            self.blocks = nn.ModuleList([_Block("ffn") for _ in range(N_LAYERS)])
            self.output_norm = nn.LayerNorm(D_MODEL)
            self.output_proj = nn.Sequential(
                nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_GRID))

        def forward(self, text_emb):            # (B, 3072)
            x = self.text_proj(text_emb).unsqueeze(1)     # (B,1,768)
            x = x + self.pos_embed(torch.zeros(x.shape[1], dtype=torch.long,
                                               device=x.device))
            x = self.input_norm(x)
            for b in self.blocks:
                x = b(x)
            x = self.output_norm(x)
            return self.output_proj(x).squeeze(1)          # (B,400)

    class FoundationDecoder(nn.Module):
        """grid(400) -> text_embedding(3072) (contrastive)."""
        def __init__(self):
            super().__init__()
            self.grid_proj = nn.Sequential(
                nn.Linear(D_GRID, D_MODEL), nn.LayerNorm(D_MODEL),
                nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
            self.pos_embed = nn.Embedding(MAX_SEQ, D_MODEL)
            self.blocks = nn.ModuleList([_Block("ff") for _ in range(N_LAYERS)])
            self.text_out_norm = nn.LayerNorm(D_MODEL)
            self.text_out_proj = nn.Sequential(
                nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_TEXT))

        def forward(self, grid):                # (B,400)
            x = self.grid_proj(grid).unsqueeze(1)
            x = x + self.pos_embed(torch.zeros(x.shape[1], dtype=torch.long,
                                               device=x.device))
            for b in self.blocks:
                x = b(x)
            x = self.text_out_norm(x)
            return self.text_out_proj(x).squeeze(1)         # (B,3072)


def _extract_sd(obj):
    """Pull the tensor state_dict out of a checkpoint dict, stripping compile prefixes."""
    sd = obj
    for k in ("model_state_dict", "model", "state_dict", "net", "weights"):
        if isinstance(obj, dict) and k in obj and hasattr(obj[k], "items"):
            sd = obj[k]; break
    # strip torch.compile "_orig_mod." prefix if present
    return { (kk[len("_orig_mod."):] if kk.startswith("_orig_mod.") else kk): vv
             for kk, vv in sd.items() }


def load_encoder(path, device="cpu"):
    obj = torch.load(path, map_location=device, weights_only=False)
    m = FoundationEncoder()
    missing, unexpected = m.load_state_dict(_extract_sd(obj), strict=False)
    m.eval().to(device)
    return m, {"missing": list(missing), "unexpected": list(unexpected)}

def load_decoder(path, device="cpu"):
    obj = torch.load(path, map_location=device, weights_only=False)
    m = FoundationDecoder()
    missing, unexpected = m.load_state_dict(_extract_sd(obj), strict=False)
    m.eval().to(device)
    return m, {"missing": list(missing), "unexpected": list(unexpected)}


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/foundation_model"
    if not _TORCH:
        print("torch not available here; run on the pod."); raise SystemExit(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc_p = os.path.join(root, "phase2_text_to_grid/best_text_to_grid.pt")
    dec_p = os.path.join(root, "phase3_grid_to_text_v3/best_grid_to_text.pt")

    print(f"device={dev}")
    enc, ei = load_encoder(enc_p, dev)
    print(f"ENCODER loaded. missing={len(ei['missing'])} unexpected={len(ei['unexpected'])}")
    if ei["missing"][:8]:   print("  missing (first 8):", ei["missing"][:8])
    if ei["unexpected"][:8]:print("  unexpected (first 8):", ei["unexpected"][:8])

    dec, di = load_decoder(dec_p, dev)
    print(f"DECODER loaded. missing={len(di['missing'])} unexpected={len(di['unexpected'])}")
    if di["missing"][:8]:   print("  missing (first 8):", di["missing"][:8])
    if di["unexpected"][:8]:print("  unexpected (first 8):", di["unexpected"][:8])

    # forward smoke test with a random 3072 text-embedding
    with torch.no_grad():
        te = torch.randn(1, D_TEXT, device=dev)
        g = enc(te)
        print(f"\nencoder: text_emb(3072) -> grid {tuple(g.shape)} "
              f"range[{g.min():.2f},{g.max():.2f}]")
        back = dec(g)
        print(f"decoder: grid(400) -> text_emb {tuple(back.shape)}")
    print("\nIf missing/unexpected are both ~0, the reconstruction MATCHES the real weights.")
    print("If not, paste the missing/unexpected lists and I'll fix the module to match.")
