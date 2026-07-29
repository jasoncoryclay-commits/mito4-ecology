#!/usr/bin/env python3
"""
foundation_probe.py — discover the REAL FOUNDATION shard's interface on the pod.

FOUNDATION (shard 02, /workspace/foundation_model) is documented as
`text <-> 400-dim grid` (== 20x20). But the exact API — module name, class,
method signatures, checkpoint file, and the numeric RANGE of grid values — is
only knowable on the pod. This script introspects it and writes a JSON report
that foundation_adapter.py consumes, so the adapter binds to what actually
exists rather than what we guessed.

Run ON THE POD:
    python3 foundation_probe.py            # auto-searches /workspace/foundation_model
    python3 foundation_probe.py /path/to/foundation_model

It does NOT modify anything (read-only inspection). Output: foundation_probe.json
"""
import os, sys, json, glob, inspect, importlib.util, traceback

DEFAULT_PATHS = [
    "/workspace/foundation_model",
    "/workspace/maw_mindmap/foundation_model",
    os.path.expanduser("~/foundation_model"),
]

def find_root(argv):
    if len(argv) > 1 and os.path.isdir(argv[1]):
        return argv[1]
    for p in DEFAULT_PATHS:
        if os.path.isdir(p):
            return p
    return None

def scan_files(root):
    exts = ("*.py", "*.pt", "*.pth", "*.bin", "*.safetensors", "*.json", "*.npy", "*.npz")
    found = {}
    for e in exts:
        hits = glob.glob(os.path.join(root, "**", e), recursive=True)
        if hits:
            found[e] = [os.path.relpath(h, root) for h in hits[:50]]
    return found

def grep_grid_hints(root):
    """Look for the grid dimension and value-range in the source."""
    hints = []
    needles = ["400", "20", "grid", "sigmoid", "tanh", "softmax", "normalize",
               "clip", "def encode", "def decode", "def text_to_grid",
               "def grid_to_text", "class Foundation", "GRID"]
    for py in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        try:
            src = open(py, errors="ignore").read()
        except Exception:
            continue
        for n in needles:
            if n in src:
                # capture the line for context
                for ln in src.splitlines():
                    if n in ln and len(ln.strip()) < 200:
                        hints.append({"file": os.path.relpath(py, root),
                                      "needle": n, "line": ln.strip()})
                        break
    # dedupe
    seen, out = set(), []
    for h in hints:
        k = (h["file"], h["needle"])
        if k not in seen:
            seen.add(k); out.append(h)
    return out[:80]

def inspect_checkpoints(root):
    """Deep-inspect the key .pt checkpoints so the adapter knows HOW to load them.
    Reports: is it a full model / state_dict / dict-with-config; tensor key shapes;
    and any grid-dim (400/20) or config clues. Requires torch; degrades if absent."""
    out = {}
    try:
        import torch
    except Exception as e:
        return {"_torch_error": f"{type(e).__name__}: {e}"}
    targets = [
        "phase2_text_to_grid/best_text_to_grid.pt",
        "phase3_grid_to_text_v3/best_grid_to_text.pt",
        "phase3_spi_grids/final_grid_to_text.pt",
        "phase2_spi_true/best_model.pt",
        "final_model.pt", "best_pretrained.pt",
    ]
    for rel in targets:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        info = {"exists": True, "bytes": os.path.getsize(path)}
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            try:
                obj = torch.load(path, map_location="cpu", weights_only=True)
                info["loaded_weights_only"] = True
            except Exception as e2:
                info["load_error"] = f"{type(e).__name__}: {e} | wo: {e2}"
                out[rel] = info; continue
        info["type"] = type(obj).__name__
        if isinstance(obj, dict):
            info["top_keys"] = list(obj.keys())[:25]
            # find the actual state_dict (common wrappers)
            sd = None
            for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
                if k in obj and hasattr(obj[k], "items"):
                    sd = obj[k]; info["state_dict_key"] = k; break
            if sd is None and all(hasattr(v, "shape") for v in list(obj.values())[:3]):
                sd = obj; info["state_dict_key"] = "<root>"
            # config-ish entries
            for k in ("config", "args", "hparams", "grid_dim", "grid_side",
                      "vocab", "vocab_size", "d_model", "hidden"):
                if k in obj:
                    v = obj[k]
                    info.setdefault("config", {})[k] = str(v)[:200]
            if sd is not None:
                items = list(sd.items())
                info["n_tensors"] = len(items)
                info["first_layers"] = [(k, list(v.shape)) for k, v in items[:6]
                                        if hasattr(v, "shape")]
                info["last_layers"] = [(k, list(v.shape)) for k, v in items[-6:]
                                       if hasattr(v, "shape")]
                # hunt for a 400 or 20x20 dimension in any tensor shape
                dims = set()
                for k, v in items:
                    if hasattr(v, "shape"):
                        for d in v.shape:
                            dims.add(int(d))
                info["has_dim_400"] = 400 in dims
                info["has_dim_20"] = 20 in dims
                info["notable_dims"] = sorted(d for d in dims if d in (20, 400, 401, 512, 768))
        elif hasattr(obj, "state_dict"):
            info["is_full_module"] = True
            info["module_repr"] = repr(obj)[:600]
        out[rel] = info
    return out

def try_import(root):
    """Best-effort: import likely entry modules and list their public API."""
    api = {}
    candidates = glob.glob(os.path.join(root, "*.py")) + \
                 glob.glob(os.path.join(root, "**", "__init__.py"), recursive=True)
    for path in candidates[:20]:
        modname = "fnd_" + os.path.splitext(os.path.basename(path))[0]
        try:
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # may fail if heavy deps; that's fine
            members = {}
            for name, obj in inspect.getmembers(mod):
                if name.startswith("_"):
                    continue
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except (ValueError, TypeError):
                        sig = "(?)"
                    members[name] = {"kind": "class" if inspect.isclass(obj) else "func",
                                     "sig": sig}
            if members:
                api[os.path.relpath(path, root)] = members
        except Exception as e:
            api[os.path.relpath(path, root)] = {"_import_error": f"{type(e).__name__}: {e}"}
    return api

def main():
    root = find_root(sys.argv)
    report = {"root": root, "found_root": bool(root)}
    if not root:
        report["error"] = ("FOUNDATION model dir not found. Pass the path: "
                            "python3 foundation_probe.py /workspace/foundation_model")
        print(json.dumps(report, indent=2))
        json.dump(report, open("foundation_probe.json", "w"), indent=2)
        return
    try:
        report["files"] = scan_files(root)
        report["grid_hints"] = grep_grid_hints(root)
        report["api"] = try_import(root)
        report["checkpoints"] = inspect_checkpoints(root)
    except Exception:
        report["probe_error"] = traceback.format_exc()

    json.dump(report, open("foundation_probe.json", "w"), indent=2)
    print(json.dumps({k: report.get(k) for k in ("root", "files")}, indent=2))
    print("\nGrid hints (dimension / range clues):")
    for h in report.get("grid_hints", [])[:25]:
        print(f"  [{h['needle']:>10}] {h['file']}: {h['line'][:90]}")
    print("\nDiscovered API (classes/functions):")
    for f, members in report.get("api", {}).items():
        print(f"  {f}:")
        for name, info in (members.items() if isinstance(members, dict) else []):
            if name == "_import_error":
                print(f"     (import error: {info})")
            else:
                print(f"     {info.get('kind','?'):5} {name}{info.get('sig','')}")
    print("\n=== CHECKPOINT INSPECTION (how to load the real model) ===")
    ck = report.get("checkpoints", {})
    if ck.get("_torch_error"):
        print("  torch not importable here:", ck["_torch_error"])
    for rel, info in ck.items():
        if rel.startswith("_"):
            continue
        print(f"\n  {rel}")
        print(f"    type={info.get('type')}  n_tensors={info.get('n_tensors')}  "
              f"has_400={info.get('has_dim_400')} has_20={info.get('has_dim_20')} "
              f"notable_dims={info.get('notable_dims')}")
        if info.get("state_dict_key"):
            print(f"    state_dict at key: {info['state_dict_key']}")
        if info.get("config"):
            print(f"    config: {info['config']}")
        if info.get("first_layers"):
            print(f"    first layers: {info['first_layers']}")
        if info.get("last_layers"):
            print(f"    last  layers: {info['last_layers']}")
        if info.get("module_repr"):
            print(f"    MODULE (loadable directly!): {info['module_repr'][:300]}")
        if info.get("load_error"):
            print(f"    load_error: {info['load_error']}")
    print("\nFull report -> foundation_probe.json")
    print("Send me the CHECKPOINT INSPECTION block and I'll bind the real encoder/decoder.")

if __name__ == "__main__":
    main()
