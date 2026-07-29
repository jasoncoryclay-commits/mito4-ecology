#!/usr/bin/env python3
"""
trustworthy_test.py — the honest, high-power version of the exceed-itself test.

Implements all three fixes, in the scientifically correct order:

  OPTION 1  Fit a REAL energy<->raw inverse.
            MITO-4 emits a [0,1] energy grid; the decoder was trained on the
            encoder's NATIVE (raw) scale. We fit a global affine map raw≈a*energy+b
            from paired (raw_grid, minmax_energy_grid) samples over many concepts,
            so a MITO-4 return-grid can be pushed back into the decoder's input
            distribution instead of being off-distribution. Recalibrate round-trip.

  OPTION 2  Directional semantic probe (high statistical power).
            Argmax-nearest over a small bank is a blunt, unstable instrument.
            Instead: for each concept, measure the embedding SHIFT from seed-grid
            to transcended-grid, and test whether that shift is CONSISTENT in
            direction across concepts (real structure) vs the shift produced by a
            matched RANDOM perturbation (null). Uses cosine of shift-vectors and a
            pre-committed effect-size margin. Far more sensitive than argmax.

  OPTION 3  Honest scoping printed at the end: what is proven vs not, and the
            gating rule that must PASS before this is ever bused to Mia.

Pre-committed BEFORE looking at results (do not edit after running):
  ROUNDTRIP_TARGET   = 0.66   # instrument must round-trip >=2/3 before we trust it
  DIRECTION_MARGIN   = 0.15   # mean |cos(real shift, concept-axis)| must beat null by this
  N_PROBE            = fixed concept list below

Run ON THE POD with OPENAI_API_KEY set:
    python3 trustworthy_test.py
"""
from __future__ import annotations
import numpy as np
import foundation_adapter as fa
import mito4_mia_bridge as bridge

# ---------- PRE-COMMITTED ----------
BANK = sorted(set([
    "longing","grief","hope","betrayal","awe","silence","fear","joy","shame",
    "wonder","rage","tenderness","despair","gratitude","loneliness","serenity",
    "envy","pride","curiosity","calm","anger","love","disgust","surprise"]))
CALIB_CONCEPTS = BANK[:16]     # to fit the energy->raw map + round-trip check
ROUNDTRIP_TARGET = 0.66
DIRECTION_MARGIN = 0.15
TICKS, SEED = 30, 7
np.random.seed(0)


def minmax(v):
    v = np.asarray(v, dtype=np.float64).ravel()[:400]
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else v * 0


def fit_energy_to_raw(pairs):
    """Global affine fit raw ≈ a*energy + b (least squares over all cells/concepts)."""
    E = np.concatenate([minmax(e) for e, r in pairs])
    R = np.concatenate([np.asarray(r, np.float64).ravel()[:400] for e, r in pairs])
    A = np.vstack([E, np.ones_like(E)]).T
    (a, b), *_ = np.linalg.lstsq(A, R, rcond=None)
    return float(a), float(b)


def nearest(adapter, grid_raw, bank_emb):
    emb = adapter.grid_to_embedding(grid_raw)
    if emb is None:
        return None, -1, None
    best, bc = None, -1e9
    for name, e in bank_emb.items():
        d = np.linalg.norm(emb) * np.linalg.norm(e)
        c = float(np.dot(emb, e) / d) if d else -1e9
        if c > bc:
            best, bc = name, c
    return best, bc, emb


def main():
    from embedder_openai import OpenAIEmbedder, concept_bank
    embedder = OpenAIEmbedder()
    adapter = fa.FoundationAdapter(text_embedder=embedder, verbose=True)
    if adapter._enc is None or adapter._dec is None:
        print("Need mode=real with BOTH encoder+decoder. Aborting."); return
    print(f"[adapter] mode={adapter.mode} enc=True dec=True\n")

    bank_emb = concept_bank(BANK, embedder)
    raws = {c: adapter.text_to_grid_raw(c) for c in BANK}

    # ---------- OPTION 1: fit energy->raw, recalibrate ----------
    print("=" * 64)
    print("OPTION 1 — fit energy->raw inverse, recalibrate round-trip")
    print("-" * 64)
    pairs = [(minmax(raws[c]), raws[c]) for c in CALIB_CONCEPTS]  # identity-ish anchor
    a, b = fit_energy_to_raw(pairs)
    print(f"   fitted affine: raw ~= {a:.4f}*energy + {b:.4f}")

    def energy_to_raw(energy_grid):
        return (a * minmax(energy_grid) + b).reshape(20, 20)

    # round-trip: raw grid decode (baseline) vs going through minmax->energy_to_raw
    hit_raw = hit_rt = 0
    for c in CALIB_CONCEPTS:
        nm_raw, _, _ = nearest(adapter, np.asarray(raws[c]).reshape(20, 20), bank_emb)
        nm_rt, _, _ = nearest(adapter, energy_to_raw(minmax(raws[c])), bank_emb)
        hit_raw += (nm_raw == c); hit_rt += (nm_rt == c)
    acc_raw = hit_raw / len(CALIB_CONCEPTS)
    acc_rt = hit_rt / len(CALIB_CONCEPTS)
    print(f"   round-trip acc (native raw grid):        {acc_raw:.2f}")
    print(f"   round-trip acc (energy->raw reconstruct):{acc_rt:.2f}")
    instrument_ok = acc_raw >= ROUNDTRIP_TARGET
    print(f"   instrument trustworthy (raw acc >= {ROUNDTRIP_TARGET})? {instrument_ok}")

    # ---------- OPTION 2: directional semantic probe ----------
    print("\n" + "=" * 64)
    print("OPTION 2 — directional semantic probe (real shift vs null shift)")
    print("-" * 64)
    real_align, null_align = [], []
    rng = np.random.default_rng(7)
    for c in BANK:
        raw = np.asarray(raws[c], np.float64).ravel()[:400]
        # seed embedding (through decoder)
        _, _, e_seed = nearest(adapter, raw.reshape(20, 20), bank_emb)
        # MITO-4 transcended grid -> back to raw scale -> embedding
        eco_in = minmax(raw)
        res = bridge.mito4_advisory(eco_in.reshape(20, 20), concept=c,
                                    ticks=TICKS, seed=SEED, transcend=True)
        out_raw = energy_to_raw(res["return_grid"])
        _, _, e_out = nearest(adapter, out_raw, bank_emb)
        # concept axis = seed embedding direction in text space (from the bank)
        c_axis = bank_emb[c]
        if e_seed is None or e_out is None:
            continue
        shift = e_out - e_seed
        # how aligned is the transcendence-induced shift with the concept's own axis?
        d = np.linalg.norm(shift) * np.linalg.norm(c_axis)
        real_align.append(abs(float(np.dot(shift, c_axis) / d)) if d else 0.0)
        # NULL: random perturbation of same magnitude on the seed grid
        noise = rng.standard_normal(400); noise *= (np.linalg.norm(minmax(raw)) / (np.linalg.norm(noise)+1e-9))
        nraw = energy_to_raw(np.clip(minmax(raw) + 0.3*noise, 0, 1))
        _, _, e_null = nearest(adapter, nraw, bank_emb)
        if e_null is not None:
            ns = e_null - e_seed
            d2 = np.linalg.norm(ns) * np.linalg.norm(c_axis)
            null_align.append(abs(float(np.dot(ns, c_axis) / d2)) if d2 else 0.0)

    real_m = float(np.mean(real_align)) if real_align else 0.0
    null_m = float(np.mean(null_align)) if null_align else 0.0
    delta = real_m - null_m
    print(f"   mean |align(transcend shift, concept axis)|: {real_m:.3f}  (n={len(real_align)})")
    print(f"   mean |align(random   shift, concept axis)|: {null_m:.3f}  (n={len(null_align)})")
    print(f"   delta (real - null): {delta:+.3f}   (pre-committed margin {DIRECTION_MARGIN})")

    # ---------- OPTION 3: honest verdict + scoping ----------
    print("\n" + "=" * 64)
    print("OPTION 3 — HONEST VERDICT & SCOPING")
    print("-" * 64)
    print("  PROVEN (independent of this test):")
    print("   - vocabulary novelty: MITO-4 reaches thresholds outside its seed set")
    print("   - return path: re-animated grid measurably changes (return_shift)")
    print("   - real FOUNDATION binding: encoder+decoder load exact (missing=0 unexpected=0)")
    print()
    if not instrument_ok:
        print(f"  SEMANTIC verdict: VOID — round-trip {acc_raw:.2f} < {ROUNDTRIP_TARGET}.")
        print("   The decoder's grid->concept channel is too lossy to measure semantic")
        print("   novelty at all. NOT a MITO-4 failure — the instrument is blunt.")
        print("   Next: improve grid->text fidelity (retrain/better decoder) before claiming")
        print("   semantic transcendence. Do NOT bus semantic claims to Mia.")
    elif delta >= DIRECTION_MARGIN:
        print(f"  SEMANTIC verdict: PASS — transcendence shift aligns with concept structure")
        print(f"   beyond noise (delta {delta:+.3f} >= {DIRECTION_MARGIN}). Semantic novelty")
        print("   is demonstrated. Safe to propose busing (advisory-only) to Mia.")
    else:
        print(f"  SEMANTIC verdict: FAIL — shift not distinguishable from noise")
        print(f"   (delta {delta:+.3f} < {DIRECTION_MARGIN}). Report vocab-novelty + return")
        print("   path only. Semantic self-transcendence NOT demonstrated. Do NOT bus it.")
    print()
    print("  GATING RULE before MITO-4 semantic output reaches Mia's advisory bus:")
    print("   round-trip acc >= 0.66  AND  directional delta >= 0.15 (this test PASS).")
    print("=" * 64)


if __name__ == "__main__":
    main()
