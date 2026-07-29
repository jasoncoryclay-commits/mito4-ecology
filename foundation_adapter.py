#!/usr/bin/env python3
"""
foundation_adapter.py — bind the MITO-4 bridge to the REAL FOUNDATION shard.

Provides one stable contract to the rest of the system:

    fa = FoundationAdapter()          # auto-detects the real model, else stand-in
    grid = fa.text_to_grid("longing") # -> 20x20 float grid in [0,1]
    text = fa.grid_to_text(grid)      # -> str (if the real model supports decode)

It adapts to the real model discovered by foundation_probe.py, handling the common
shape/range variations (400-vector vs 20x20, logits vs sigmoid vs raw), and falls
back to a DETERMINISTIC hash-embedding stand-in when foundation_model is absent so
the pipeline is testable off-pod. Either way the output honors the bridge contract:
20x20, finite, min-max normalized to [0,1].

The adapter is READ-ONLY toward FOUNDATION: it calls encode/decode, never trains,
never writes weights (consistent with the additive/sealed rules).
"""
from __future__ import annotations
import os, sys, json, glob, hashlib, importlib.util
import numpy as np

GRID_SIDE = 20
GRID_CELLS = 400
D_TEXT_DIM = 3072      # FOUNDATION's text-embedding dim (encoder input / decoder output)

# candidate method names the real FOUNDATION model might expose for each direction
ENCODE_NAMES = ["text_to_grid", "encode", "encode_text", "text_to_state",
                "to_grid", "embed", "forward"]
DECODE_NAMES = ["grid_to_text", "decode", "decode_grid", "state_to_text",
                "to_text", "invert"]


def _normalize_to_grid(arr) -> np.ndarray:
    """Coerce ANY numeric output into a 20x20 [0,1] grid (the bridge contract)."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return np.zeros((GRID_SIDE, GRID_SIDE))
    # take/pad to exactly 400
    if a.size >= GRID_CELLS:
        a = a[:GRID_CELLS]
    else:
        a = np.pad(a, (0, GRID_CELLS - a.size))
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    return a.reshape(GRID_SIDE, GRID_SIDE)


class FoundationAdapter:
    def __init__(self, probe_json="foundation_probe.json", root=None, verbose=True,
                 text_embedder=None):
        self.verbose = verbose
        self.model = None
        self.encode_fn = None
        self.decode_fn = None
        self.mode = "standin"          # "real" once a live model binds
        self.root = root
        # text_embedder: callable(str)->np.ndarray(3072). Needed for the REAL encoder,
        # whose input is a 3072-dim text embedding (not a raw string). If None, the
        # real ENCODER cannot run from text; grid-space ops and the DECODER still work.
        self.text_embedder = text_embedder
        self._enc = self._dec = None   # torch modules once bound
        self._device = "cpu"
        self._bind(probe_json)

    def _log(self, *a):
        if self.verbose:
            print("[foundation_adapter]", *a)

    def _bind(self, probe_json):
        root = self.root
        if not root and os.path.exists(probe_json):
            try:
                root = json.load(open(probe_json)).get("root")
            except Exception:
                root = None
        # PATH 1: torch checkpoints (the real FOUNDATION shard is .pt files)
        for p in ([root] if root else []) + ["/workspace/foundation_model"]:
            if p and os.path.isdir(p) and self._try_bind_torch(p):
                self.mode = "real"
                self.root = p
                self._log(f"bound REAL FOUNDATION (torch checkpoints) at {p}")
                return
        # PATH 2: importable modules (fallback for a code-based FOUNDATION)
        for p in ([root] if root else []) + ["/workspace/foundation_model"]:
            if p and os.path.isdir(p) and self._try_bind_dir(p):
                self.mode = "real"
                self.root = p
                self._log(f"bound REAL FOUNDATION at {p}")
                return
        self._log("no live FOUNDATION model bound -> deterministic stand-in "
                  "(fine off-pod; run foundation_probe.py on the pod for the real one)")

    def _try_bind_torch(self, root):
        """Bind the real .pt checkpoints via foundation_torch (reconstructed modules)."""
        enc_p = os.path.join(root, "phase2_text_to_grid/best_text_to_grid.pt")
        dec_p = os.path.join(root, "phase3_grid_to_text_v3/best_grid_to_text.pt")
        if not (os.path.exists(enc_p) or os.path.exists(dec_p)):
            return False
        try:
            import torch
            import foundation_torch as ft
        except Exception as e:
            self._log(f"torch/foundation_torch unavailable ({e}); not binding torch path")
            return False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        bound = False
        if os.path.exists(enc_p):
            try:
                self._enc, info = ft.load_encoder(enc_p, self._device)
                self._log(f"encoder loaded (missing={len(info['missing'])} "
                          f"unexpected={len(info['unexpected'])})")
                bound = True
            except Exception as e:
                self._log(f"encoder load failed: {e}")
        if os.path.exists(dec_p):
            try:
                self._dec, info = ft.load_decoder(dec_p, self._device)
                self._log(f"decoder loaded (missing={len(info['missing'])} "
                          f"unexpected={len(info['unexpected'])})")
                bound = True
            except Exception as e:
                self._log(f"decoder load failed: {e}")
        return bound

    def _try_bind_dir(self, root):
        """Try to import a module in root and locate encode/decode callables."""
        cands = glob.glob(os.path.join(root, "*.py")) + \
                glob.glob(os.path.join(root, "**", "__init__.py"), recursive=True)
        for path in cands[:20]:
            try:
                spec = importlib.util.spec_from_file_location(
                    "fnd_" + os.path.splitext(os.path.basename(path))[0], path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception:
                continue
            # look for a class we can instantiate, or module-level functions
            enc = self._find_callable(mod, ENCODE_NAMES)
            dec = self._find_callable(mod, DECODE_NAMES)
            if enc:
                self.encode_fn, self.decode_fn = enc, dec
                return True
        return False

    def _find_callable(self, mod, names):
        # module-level function first
        for n in names:
            fn = getattr(mod, n, None)
            if callable(fn):
                return fn
        # else a class exposing one of the methods (instantiate with no args if possible)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type):
                for n in names:
                    if hasattr(obj, n):
                        try:
                            inst = obj()
                        except Exception:
                            continue
                        m = getattr(inst, n, None)
                        if callable(m):
                            self.model = inst
                            return m
        return None

    def text_to_grid_raw(self, text: str):
        """Encoder's NATIVE grid output (no normalization). This is what the DECODER
        was trained to consume. Returns (400,) float or None if no real encoder."""
        if self._enc is None:
            return None
        emb = self._embed(text)
        if emb is None:
            return None
        try:
            import torch
            with torch.no_grad():
                t = torch.as_tensor(emb, dtype=torch.float32,
                                    device=self._device).reshape(1, -1)
                return self._enc(t).squeeze(0).detach().cpu().numpy()
        except Exception as e:
            self._log(f"raw encode failed ({e})")
            return None

    # ---- the stable contract ----
    def text_to_grid(self, text: str) -> np.ndarray:
        # real torch encoder: needs a 3072-dim text embedding
        if self._enc is not None:
            emb = self._embed(text)
            if emb is not None:
                try:
                    import torch
                    with torch.no_grad():
                        t = torch.as_tensor(emb, dtype=torch.float32,
                                            device=self._device).reshape(1, -1)
                        out = self._enc(t).squeeze(0).detach().cpu().numpy()
                    return _normalize_to_grid(out)
                except Exception as e:
                    self._log(f"real encode failed ({e}); stand-in for this call")
        # legacy importable-fn encoder
        if self.mode == "real" and self.encode_fn:
            try:
                out = self.encode_fn(text)
                if hasattr(out, "detach"):
                    out = out.detach().cpu().numpy()
                return _normalize_to_grid(out)
            except Exception as e:
                self._log(f"real encode failed ({e}); stand-in for this call")
        return self._standin_grid(text)

    def grid_to_embedding(self, grid: np.ndarray):
        """REAL decoder: grid(400) -> 3072-dim text embedding (contrastive space).
        Compare against a concept bank to name it (see grid_to_text)."""
        if self._dec is None:
            return None
        try:
            import torch
            with torch.no_grad():
                g = torch.as_tensor(np.asarray(grid).ravel()[:GRID_CELLS],
                                    dtype=torch.float32, device=self._device).reshape(1, -1)
                return self._dec(g).squeeze(0).detach().cpu().numpy()
        except Exception as e:
            self._log(f"real decode failed ({e})")
            return None

    def grid_to_text(self, grid: np.ndarray, concept_bank: dict | None = None):
        """Name a grid by nearest concept in an embedding bank {name: emb(3072)}.
        Without a bank, returns the raw embedding (caller matches). None if no decoder."""
        emb = self.grid_to_embedding(grid)
        if emb is None:
            return None
        if not concept_bank:
            return emb
        # cosine nearest
        best, bestcos = None, -1e9
        for name, e in concept_bank.items():
            e = np.asarray(e).ravel()
            denom = np.linalg.norm(emb) * np.linalg.norm(e)
            cos = float(np.dot(emb, e) / denom) if denom else -1e9
            if cos > bestcos:
                best, bestcos = name, cos
        return {"nearest": best, "cosine": bestcos}

    def _embed(self, text: str):
        """text -> 3072-dim embedding via the user-supplied embedder (or None)."""
        if self.text_embedder is None:
            return None
        try:
            v = np.asarray(self.text_embedder(text), dtype=np.float64).ravel()
            if v.size != D_TEXT_DIM:
                v = np.resize(v, D_TEXT_DIM)
            return v
        except Exception as e:
            self._log(f"text_embedder failed ({e})")
            return None

    def _standin_grid(self, text: str) -> np.ndarray:
        """Deterministic hash-embedding: same text -> same grid, structured (not noise).
        Produces smooth blobs so the ecology has real structure to animate."""
        seed = int(hashlib.sha256((text or "").encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        # a few gaussian bumps -> structured activation
        g = np.zeros((GRID_SIDE, GRID_SIDE))
        ys, xs = np.mgrid[0:GRID_SIDE, 0:GRID_SIDE]
        for _ in range(rng.integers(2, 5)):
            cy, cx = rng.integers(0, GRID_SIDE, 2)
            s = rng.uniform(2, 5)
            g += np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * s * s))
        return _normalize_to_grid(g)


if __name__ == "__main__":
    fa = FoundationAdapter()
    print(f"\nmode = {fa.mode}   root = {fa.root}")
    for concept in ["longing", "betrayal", "longing"]:
        g = fa.text_to_grid(concept)
        print(f"  text_to_grid('{concept}') -> shape {g.shape} "
              f"range [{g.min():.2f},{g.max():.2f}] mean {g.mean():.2f} "
              f"hash {hashlib.sha256(g.tobytes()).hexdigest()[:8]}")
    # determinism: same concept -> identical grid
    a = fa.text_to_grid("longing"); b = fa.text_to_grid("longing")
    assert np.array_equal(a, b), "adapter non-deterministic"
    print("adapter determinism: PASS")
