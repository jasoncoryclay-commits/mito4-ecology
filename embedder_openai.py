#!/usr/bin/env python3
"""
embedder_openai.py — the text->3072 embedder FOUNDATION was trained on.

Confirmed from phase2_spi_true/summary.json:  embedding_source = text-embedding-3-large (real)
FOUNDATION's encoder input dim is 3072 (verified from weights), and OpenAI's
text-embedding-3-large is natively 4096-dim but supports the `dimensions` param;
FOUNDATION used dimensions=3072. This module reproduces exactly that.

    emb = OpenAIEmbedder()          # needs OPENAI_API_KEY in env
    v = emb("longing")             # -> np.ndarray shape (3072,)

Features:
  * dimensions=3072 (matches FOUNDATION) — override if you retrain.
  * on-disk cache (deterministic + avoids re-billing the same text).
  * optional L2 normalization (toggle; some training pipelines normalize).
  * graceful: if the SDK/key is absent, raises a clear error so the adapter
    falls back to grid-space instead of silently using wrong vectors.

Nothing here writes to Mia or FOUNDATION. Read-only embedding calls.
"""
from __future__ import annotations
import os, json, hashlib
import numpy as np

MODEL = "text-embedding-3-large"
DIM = 3072                      # FOUNDATION's trained text dim (NOT the native 4096)
CACHE_DIR = os.environ.get("MITO4_EMB_CACHE", ".emb_cache")


class OpenAIEmbedder:
    def __init__(self, model=MODEL, dimensions=DIM, l2_normalize=False,
                 cache_dir=CACHE_DIR, api_key=None):
        self.model = model
        self.dimensions = dimensions
        self.l2 = l2_normalize
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._client = None
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def _client_lazy(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError(
                    "openai SDK not installed. `pip install openai`, and set "
                    "OPENAI_API_KEY. (FOUNDATION used text-embedding-3-large.)") from e
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment.")
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _cache_path(self, text):
        h = hashlib.sha256(f"{self.model}|{self.dimensions}|{text}".encode()).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{h}.npy")

    def __call__(self, text: str) -> np.ndarray:
        cp = self._cache_path(text)
        if os.path.exists(cp):
            v = np.load(cp)
        else:
            resp = self._client_lazy().embeddings.create(
                model=self.model, input=text, dimensions=self.dimensions)
            v = np.asarray(resp.data[0].embedding, dtype=np.float32)
            np.save(cp, v)
        if self.l2:
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v


def concept_bank(concepts, embedder) -> dict:
    """Build {concept: 3072-vec} for naming grids via the decoder (semantic novelty)."""
    return {c: embedder(c) for c in concepts}


if __name__ == "__main__":
    try:
        emb = OpenAIEmbedder()
        v = emb("longing")
        print(f"text-embedding-3-large @ dim={emb.dimensions}: 'longing' -> shape {v.shape} "
              f"norm {np.linalg.norm(v):.3f}")
        v2 = emb("longing")   # cache hit
        print("cache determinism:", "PASS" if np.array_equal(v, v2) else "FAIL")
        print("Ready to feed FOUNDATION's encoder (expects 3072).")
    except Exception as e:
        print(f"embedder not runnable here: {e}")
        print("On the pod: pip install openai ; export OPENAI_API_KEY=...")
