#!/usr/bin/env python3
"""
calibrate_and_test.py — make the semantic signal TRUSTWORTHY before trusting it.

Two honest experiments, both pre-committed:

A) CALIBRATION — which grid convention does the real DECODER actually want?
   The encoder emits a native grid; MITO-4 needs [0,1] energy; but the decoder was
   trained on SOME convention. We find it empirically: for each convention, encode a
   set of concepts -> grid -> decode -> nearest-in-bank, and measure ROUND-TRIP
   ACCURACY (does 'grief' decode back nearest to 'grief'?). The convention with the
   highest self-consistency is the one the decoder expects. No guessing.

   Conventions tried: raw | minmax[0,1] | sigmoid | zscore | zscore->sigmoid

B) NULL-CONTROL EXCEED-ITSELF — is MITO-4's semantic drift real or noise?
   Pre-committed test: animate each concept's grid with transcend, decode seed vs
   transcended, and measure how often the nearest concept CHANGES. Compare against a
   NULL: apply the same MITO-4 dynamics to RANDOM grids. Exceed-itself is only
   credible if transcended-grid concept-change rate exceeds the random-grid rate by a
   pre-declared margin (default: >= 0.20 absolute), AND seed round-trip accuracy (A)
   is decent (else "nearest concept" is meaningless and the whole test is void).

Run ON THE POD with OPENAI_API_KEY set:
    python3 calibrate_and_test.py
"""
from __future__ import annotations
import argparse
import numpy as np

import foundation_adapter as fa
import mito4_mia_bridge as bridge

# ---- PRE-COMMITTED PARAMETERS (declared before looking at results) ----
BANK = ["longing", "grief", "hope", "betrayal", "awe", "silence", "fear", "joy",
        "shame", "wonder", "rage", "tenderness", "despair", "gratitude",
        "loneliness", "serenity", "envy", "pride", "curiosity", "grief"]
BANK = sorted(set(BANK))
PROBE_CONCEPTS = ["grief", "hope", "awe", "fear", "joy", "shame"]  # for round-trip acc
MARGIN = 0.20              # transcended change-rate must beat null by >= this
MIN_ROUNDTRIP_ACC = 0.34   # else "nearest concept" is too weak to trust (chance ~1/20=0.05)
TICKS, SEED = 30, 7


def apply_convention(raw, conv):
    g = np.asarray(raw, dtype=np.float64).ravel()[:400]
    g = np.nan_to_num(g)
    if conv == "raw":
        out = g
    elif conv == "minmax":
        lo, hi = g.min(), g.max(); out = (g - lo) / (hi - lo) if hi > lo else g * 0
    elif conv == "sigmoid":
        out = 1 / (1 + np.exp(-g))
    elif conv == "zscore":
        s = g.std(); out = (g - g.mean()) / s if s > 0 else g * 0
    elif conv == "zsig":
        s = g.std(); z = (g - g.mean()) / s if s > 0 else g * 0
        out = 1 / (1 + np.exp(-z))
    else:
        out = g
    return out.reshape(20, 20)


def nearest(adapter, grid, bank_emb):
    emb = adapter.grid_to_embedding(grid)
    if emb is None:
        return None, -1
    best, bc = None, -1e9
    for name, e in bank_emb.items():
        d = np.linalg.norm(emb) * np.linalg.norm(e)
        c = float(np.dot(emb, e) / d) if d else -1e9
        if c > bc:
            best, bc = name, c
    return best, bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=TICKS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from embedder_openai import OpenAIEmbedder, concept_bank
    embedder = OpenAIEmbedder()
    adapter = fa.FoundationAdapter(text_embedder=embedder, verbose=True)
    if adapter._enc is None or adapter._dec is None:
        print("Need BOTH encoder and decoder bound (mode=real). Aborting."); return
    print(f"[adapter] mode={adapter.mode} enc={adapter._enc is not None} dec={adapter._dec is not None}\n")

    bank_emb = concept_bank(BANK, embedder)

    # ---------- A) CALIBRATION ----------
    print("=" * 64)
    print("A) CALIBRATION — round-trip accuracy per grid convention")
    print("   (encode concept -> grid[conv] -> decode -> nearest; want nearest == concept)")
    print("-" * 64)
    conventions = ["raw", "minmax", "sigmoid", "zscore", "zsig"]
    raws = {c: adapter.text_to_grid_raw(c) for c in PROBE_CONCEPTS}
    acc = {}
    for conv in conventions:
        hits = 0
        for c in PROBE_CONCEPTS:
            grid = apply_convention(raws[c], conv)
            nm, _ = nearest(adapter, grid, bank_emb)
            hits += (nm == c)
        acc[conv] = hits / len(PROBE_CONCEPTS)
        print(f"   {conv:8s}: round-trip accuracy = {acc[conv]:.2f}  ({hits}/{len(PROBE_CONCEPTS)})")
    best_conv = max(acc, key=acc.get)
    print(f"\n   -> BEST convention: '{best_conv}' (acc {acc[best_conv]:.2f})")
    if acc[best_conv] < MIN_ROUNDTRIP_ACC:
        print(f"   WARNING: best acc {acc[best_conv]:.2f} < pre-committed floor {MIN_ROUNDTRIP_ACC}.")
        print("   The grid<->concept round-trip is too weak; semantic-novelty results below are")
        print("   NOT trustworthy yet. Likely the decoder wants a convention we haven't matched,")
        print("   or grid_proj expects a different scale. Reporting anyway, flagged as void.")

    # ---------- B) NULL-CONTROL EXCEED-ITSELF ----------
    print("\n" + "=" * 64)
    print("B) NULL-CONTROL exceed-itself test (using best convention)")
    print("   transcended concept-change rate must beat RANDOM-grid rate by "
          f">= {MARGIN:.2f}")
    print("-" * 64)

    def run_one(seed_grid_raw, rng):
        # MITO-4 wants [0,1] energy; decode uses best_conv on BOTH seed & transcended
        eco_in = apply_convention(seed_grid_raw, "minmax")   # energy field
        res = bridge.mito4_advisory(eco_in, concept="x", ticks=args.ticks,
                                    seed=args.seed, transcend=True)
        # decode seed vs transcended under the calibrated convention
        seed_dec = apply_convention(seed_grid_raw, best_conv)
        # transcended: return_grid is [0,1]; map back through best_conv-ish:
        # we treat the returned energy grid as-is under best_conv for consistency
        out_dec = apply_convention(res["return_grid"], best_conv)
        s_nm, _ = nearest(adapter, seed_dec, bank_emb)
        o_nm, _ = nearest(adapter, out_dec, bank_emb)
        return s_nm != o_nm

    # transcended drift on real concepts
    changed = 0
    for c in BANK:
        raw = adapter.text_to_grid_raw(c)
        if raw is None:
            continue
        changed += run_one(raw, None)
    real_rate = changed / len(BANK)

    # null: same dynamics on random grids
    rng = np.random.default_rng(123)
    nchanged = 0
    N_NULL = len(BANK)
    for _ in range(N_NULL):
        raw = rng.standard_normal(400) * (np.std([adapter.text_to_grid_raw("grief")]) or 1)
        nchanged += run_one(raw, rng)
    null_rate = nchanged / N_NULL

    print(f"   transcended concept-change rate: {real_rate:.2f}  ({changed}/{len(BANK)})")
    print(f"   null (random-grid) change rate:  {null_rate:.2f}  ({nchanged}/{N_NULL})")
    delta = real_rate - null_rate
    print(f"   delta (real - null): {delta:+.2f}   (pre-committed margin {MARGIN:.2f})")

    print("\n" + "=" * 64)
    print("VERDICT")
    void = acc[best_conv] < MIN_ROUNDTRIP_ACC
    if void:
        print("  VOID — round-trip accuracy too low; fix decode convention first.")
        print("  (Paste this whole output; I'll adjust the grid scaling to the decoder.)")
    elif delta >= MARGIN:
        print(f"  PASS — transcendence produces REAL semantic drift beyond noise "
              f"(delta {delta:+.2f} >= {MARGIN}).")
        print("  MITO-4 exceeds itself in Mia's terms, not just in threshold-space.")
    else:
        print(f"  FAIL — semantic drift not distinguishable from noise "
              f"(delta {delta:+.2f} < {MARGIN}).")
        print("  Vocabulary-novelty is real, but SEMANTIC novelty is not yet demonstrated.")
    print("=" * 64)


if __name__ == "__main__":
    main()
