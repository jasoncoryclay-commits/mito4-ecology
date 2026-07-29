#!/usr/bin/env python3
"""
confirm_dims.py — settle the text-embedding dimension (3072 vs 4096) from GROUND TRUTH.

Reads the real FOUNDATION weights and prints, unambiguously:
  * ENCODER text-input dim  = the `text_proj` first Linear's in_features
  * DECODER text-output dim = the `text_out_proj` last Linear's out_features
  * whether 4096 appears in ANY tensor shape (would indicate the embedder dim)
  * the contents of the FOUNDATION json configs (may name the embedder/model)

This resolves whether FOUNDATION consumes the embedder's native 4096 directly,
or a 3072-dim projection of it (=> a 4096->3072 step lives upstream).

Run ON THE POD:  python3 confirm_dims.py   [/workspace/foundation_model]
"""
import os, sys, json, glob

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/foundation_model"

def sd_of(obj):
    for k in ("model_state_dict", "model", "state_dict", "net", "weights"):
        if isinstance(obj, dict) and k in obj and hasattr(obj[k], "items"):
            obj = obj[k]; break
    return { (kk[len("_orig_mod."):] if kk.startswith("_orig_mod.") else kk): vv
             for kk, vv in obj.items() }

def main():
    try:
        import torch
    except Exception as e:
        print("torch required on the pod:", e); return

    enc_p = os.path.join(ROOT, "phase2_text_to_grid/best_text_to_grid.pt")
    dec_p = os.path.join(ROOT, "phase3_grid_to_text_v3/best_grid_to_text.pt")

    print("=" * 60)
    if os.path.exists(enc_p):
        sd = sd_of(torch.load(enc_p, map_location="cpu", weights_only=False))
        # text_proj.0 is the first Linear: weight shape [out, IN] -> IN is the text dim
        for key in sd:
            if key.startswith("text_proj") and key.endswith(".weight"):
                w = sd[key]
                print(f"ENCODER  {key}: shape {list(w.shape)}  -> TEXT-INPUT DIM = {w.shape[1]}")
                break
        alldims = set()
        for v in sd.values():
            if hasattr(v, "shape"):
                alldims |= set(int(d) for d in v.shape)
        print(f"ENCODER  contains 4096? {4096 in alldims}   contains 3072? {3072 in alldims}")
        print(f"ENCODER  all large dims present: {sorted(d for d in alldims if d >= 1000)}")
    else:
        print("encoder checkpoint not found:", enc_p)

    print("-" * 60)
    if os.path.exists(dec_p):
        obj = torch.load(dec_p, map_location="cpu", weights_only=False)
        sd = sd_of(obj)
        # text_out_proj last Linear: weight [OUT, in] -> OUT is text dim
        cand = [k for k in sd if k.startswith("text_out_proj") and k.endswith(".weight")]
        if cand:
            k = sorted(cand)[-1]
            w = sd[k]
            print(f"DECODER  {k}: shape {list(w.shape)}  -> TEXT-OUTPUT DIM = {w.shape[0]}")
        alldims = set()
        for v in sd.values():
            if hasattr(v, "shape"):
                alldims |= set(int(d) for d in v.shape)
        print(f"DECODER  contains 4096? {4096 in alldims}   contains 3072? {3072 in alldims}")
        print(f"DECODER  all large dims present: {sorted(d for d in alldims if d >= 1000)}")
        if isinstance(obj, dict) and "config" in obj:
            print(f"DECODER  saved config: {obj['config']}")

    print("=" * 60)
    print("FOUNDATION json configs (may name the 4096 embedder):")
    for jf in glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True):
        try:
            data = json.load(open(jf))
        except Exception:
            continue
        # print only keys likely to name an embedder / dims
        hits = {}
        def walk(d, prefix=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    kl = str(k).lower()
                    if any(t in kl for t in ("embed", "model", "dim", "text", "encoder",
                                             "sentence", "hidden", "4096", "3072", "name")):
                        if not isinstance(v, (dict, list)):
                            hits[f"{prefix}{k}"] = v
                    walk(v, prefix + str(k) + ".")
        walk(data)
        if hits:
            print(f"\n  {os.path.relpath(jf, ROOT)}:")
            for k, v in list(hits.items())[:20]:
                print(f"     {k} = {str(v)[:120]}")
    print("\n" + "=" * 60)
    print("VERDICT GUIDE:")
    print("  If ENCODER TEXT-INPUT DIM == 4096  -> FOUNDATION takes the embedder directly; set D_TEXT=4096.")
    print("  If it == 3072 and embedder is 4096 -> a 4096->3072 projection lives upstream; I add it.")
    print("  Paste this whole output back and I'll bind it exactly.")

if __name__ == "__main__":
    main()
