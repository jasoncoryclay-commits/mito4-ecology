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
    print("\nFull report -> foundation_probe.json")
    print("Next: python3 foundation_adapter.py --selftest   (adapter binds to this report)")

if __name__ == "__main__":
    main()
