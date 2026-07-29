#!/usr/bin/env python3
"""
run_mia_real.py — the FULL real-path demo, end to end, on the pod.

    text --[text-embedding-3-large @3072]--> emb
         --[FOUNDATION encoder]-------------> 20x20 grid (Mia's real grid language)
         --[MITO-4 ecology, transcend]------> animated grid + verbalized advisory
         --[FOUNDATION decoder]-------------> 3072 embedding
         --[nearest in concept bank]--------> what the self-reached state MEANS

This closes the SEMANTIC-novelty loop: not just "MITO-4 reached a new threshold"
(vocabulary novelty) but "the transcended grid decodes to a DIFFERENT concept than
the seed" (semantic novelty) — the honest test of exceeding itself in Mia's terms.

Everything is advisory / read-only. No writes to Mia or FOUNDATION.

Run ON THE POD (needs torch + openai + OPENAI_API_KEY):
    python3 run_mia_real.py "longing"  --bank longing grief hope betrayal awe silence
"""
from __future__ import annotations
import argparse
import numpy as np

import foundation_adapter as fa
import mito4_mia_bridge as bridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("concept", nargs="?", default="longing")
    ap.add_argument("--bank", nargs="*", default=["longing", "grief", "hope",
                                                  "betrayal", "awe", "silence", "fear", "joy"])
    ap.add_argument("--ticks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--l2", action="store_true", help="L2-normalize embeddings")
    args = ap.parse_args()

    # 1) real embedder
    try:
        from embedder_openai import OpenAIEmbedder, concept_bank
        embedder = OpenAIEmbedder(l2_normalize=args.l2)
        embedder("warmup")  # trigger a real call / clear error early
        print("[embedder] text-embedding-3-large @3072: OK")
    except Exception as e:
        print(f"[embedder] unavailable ({e}). Falling back to grid-space stand-in.")
        embedder = None

    # 2) adapter bound to the REAL FOUNDATION checkpoints, with the embedder plugged in
    adapter = fa.FoundationAdapter(text_embedder=embedder, verbose=True)
    print(f"[adapter] mode = {adapter.mode}  (want 'real')")
    print(f"[adapter] encoder bound = {adapter._enc is not None}  "
          f"decoder bound = {adapter._dec is not None}")

    # 3) text -> Mia's real 20x20 grid
    grid = adapter.text_to_grid(args.concept)
    print(f"\n[foundation] '{args.concept}' -> grid 20x20 "
          f"range[{grid.min():.2f},{grid.max():.2f}] mean {grid.mean():.2f}")

    # 4) MITO-4 animates it (advisory + return + exceed-itself)
    res = bridge.mito4_advisory(grid, concept=args.concept,
                                ticks=args.ticks, seed=args.seed, transcend=True)
    print(f"\n[mito4] return_shift={res['return_shift']:.3f}  "
          f"vocab_novelty={res['novelty']:.3f}  "
          f"lineages={res['stats_after']['lineages']} "
          f"(novel {res['stats_after']['novel_lineages']})  "
          f"max_gen={res['stats_after']['max_gen']}")
    print("\n[VERBALIZED — advisory bus payload]")
    print(res["verbalized"])

    # 5) SEMANTIC novelty: decode seed grid vs transcended grid, compare nearest concept
    if adapter._dec is not None and embedder is not None:
        bank = concept_bank(args.bank, embedder)
        seed_name = adapter.grid_to_text(grid, bank)
        out_name = adapter.grid_to_text(res["return_grid"], bank)
        print("\n[semantic novelty — the honest exceed-itself test]")
        print(f"  seed grid decodes to:        {seed_name}")
        print(f"  transcended grid decodes to: {out_name}")
        if isinstance(seed_name, dict) and isinstance(out_name, dict):
            moved = seed_name["nearest"] != out_name["nearest"]
            print(f"  -> concept {'CHANGED' if moved else 'held'}: "
                  f"{seed_name['nearest']} -> {out_name['nearest']}")
            print("  (CHANGED = the self-reached state is semantically new, not just a new threshold.)")
    else:
        print("\n[semantic novelty] decoder or embedder unavailable — "
              "vocabulary-novelty only this run. Provide OPENAI_API_KEY on the pod for the full test.")


if __name__ == "__main__":
    main()
