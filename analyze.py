#!/usr/bin/env python3
"""
MITO-4 ECOLOGY — analysis & pre-committed hypothesis scoring.
Reads results/mito4_run_seed*.csv (produced by run.sh / mito4_kernel.cu),
produces plots and a SCORECARD.md scoring H1..H5 from EXPERIMENT_CARD.md.

matplotlib + stdlib only. Usage:  python3 analyze.py [results_dir]
"""
import sys, os, glob, csv, statistics as st

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"

def load_csv(path):
    """Parse a run log. Skips the banner line and '# ...' comment lines.
    Returns dict of lists: tick, alive, mean_thr, lineages. Also lattice size + throughput."""
    ticks, alive, mthr, lin = [], [], [], []
    lattice_cells, gcups = None, None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("MITO-4 ECOLOGY"):
                # "... | HxW = N cells | ..."
                try:
                    seg = line.split("=")[1]
                    lattice_cells = int(seg.strip().split()[0])
                except Exception:
                    pass
                continue
            if line.startswith("#"):
                if "billion cell-updates/sec" in line:
                    try:
                        gcups = float(line.split("|")[1].strip().split()[0])
                    except Exception:
                        pass
                continue
            if line.startswith("tick,"):   # header
                continue
            parts = line.split(",")
            if len(parts) != 4:
                continue
            try:
                ticks.append(int(parts[0])); alive.append(int(parts[1]))
                mthr.append(float(parts[2])); lin.append(int(parts[3]))
            except ValueError:
                continue
    return dict(tick=ticks, alive=alive, mean_thr=mthr, lineages=lin,
                cells=lattice_cells, gcups=gcups, path=path)

def tail(xs, n):
    return xs[-n:] if len(xs) >= n else xs[:]

def cov(xs):
    if len(xs) < 2:
        return 0.0
    m = st.mean(xs)
    return (st.pstdev(xs) / m) if m else 0.0

def score(runs):
    """Score the five pre-committed hypotheses across seeds."""
    S = len(runs)
    lines = []
    lines.append("# MITO-4 Ecology — Hypothesis Scorecard\n")
    lines.append(f"Seeds analyzed: {S}  |  source: `{RESULTS}/`\n")

    # per-seed derived quantities
    per = []
    for r in runs:
        if not r["tick"]:
            per.append(None); continue
        final_window_alive = tail(r["alive"], max(2, len(r["alive"])//5))  # ~final 20% of samples
        occ = (r["alive"][-1] / r["cells"]) if r["cells"] else None
        d_thr = r["mean_thr"][-1] - r["mean_thr"][0]
        min_lin_tail = min(tail(r["lineages"], max(2, len(r["lineages"])//5)))
        per.append(dict(
            extinct=(r["alive"][-1] == 0),
            occ=occ,
            cov_tail=cov(final_window_alive),
            d_thr=d_thr,
            min_lin_tail=min_lin_tail,
            gcups=r["gcups"],
        ))

    valid = [p for p in per if p]

    def frac(pred):
        return sum(1 for p in valid if pred(p))

    # H1: carrying capacity 80-95% occupancy, stable (cov<0.05) over final window, non-extinct
    h1_ok = frac(lambda p: (not p["extinct"]) and p["occ"] is not None
                 and 0.80 <= p["occ"] <= 0.98 and p["cov_tail"] < 0.05)
    # H2: mean threshold drops >=5 units
    h2_ok = frac(lambda p: p["d_thr"] <= -5)
    # H3: lineage diversity does not collapse (min tail >=10)
    h3_ok = frac(lambda p: p["min_lin_tail"] >= 10)
    # H5: throughput >=20 Gcups (any seed measured)
    gc = [p["gcups"] for p in valid if p["gcups"] is not None]
    h5_val = max(gc) if gc else None

    def verdict(passing, need, n): return "PASS" if passing >= need else "FAIL"
    need = max(1, (4 * S) // 5)  # >=4/5 seeds

    lines.append("| # | Hypothesis | Prediction | Result | Verdict |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| H1 | Stable carrying capacity | 80-98% occ, CoV<0.05, non-extinct | {h1_ok}/{S} seeds | {verdict(h1_ok,need,S)} |")
    lines.append(f"| H2 | Threshold selection downward | Δmean_thr ≤ −5 | {h2_ok}/{S} seeds | {verdict(h2_ok,need,S)} |")
    lines.append(f"| H3 | Diversity does not collapse | min tail lineages ≥ 10 | {h3_ok}/{S} seeds | {verdict(h3_ok,need,S)} |")
    lines.append(f"| H4 | Spatial pattern at scale | visible fronts/patches | see snapshot_*.png (manual) | REVIEW |")
    gtxt = f"{h5_val:.2f} Gcups" if h5_val is not None else "not measured"
    lines.append(f"| H5 | H100 throughput ≥20 Gcups | measured ≥20 | {gtxt} | {'PASS' if (h5_val or 0)>=20 else 'REVIEW'} |")
    lines.append("")

    # per-seed detail
    lines.append("## Per-seed detail\n")
    lines.append("| seed | final alive | occ % | Δmean_thr | min tail lineages | extinct | Gcups |")
    lines.append("|---|---|---|---|---|---|---|")
    for r, p in zip(runs, per):
        sd = os.path.basename(r["path"]).replace("mito4_run_seed", "").replace(".csv", "")
        if not p:
            lines.append(f"| {sd} | (no data) | | | | | |"); continue
        occ = f"{100*p['occ']:.1f}" if p["occ"] is not None else "?"
        gg = f"{p['gcups']:.2f}" if p["gcups"] is not None else "?"
        lines.append(f"| {sd} | {r['alive'][-1]} | {occ} | {p['d_thr']:+.1f} | {p['min_lin_tail']} | {p['extinct']} | {gg} |")
    lines.append("")
    return "\n".join(lines)

def plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("alive", "Population (live organisms)", "population.png"),
               ("mean_thr", "Mean division threshold (selection)", "selection.png"),
               ("lineages", "Lineage diversity (distinct thresholds)", "diversity.png")]
    for key, title, fn in metrics:
        plt.figure(figsize=(9, 5))
        for r in runs:
            if r["tick"]:
                sd = os.path.basename(r["path"]).replace("mito4_run_seed","seed ").replace(".csv","")
                plt.plot(r["tick"], r[key], label=sd, linewidth=1.6)
        plt.title(f"MITO-4 Ecology — {title}")
        plt.xlabel("tick"); plt.ylabel(key); plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout(); out = os.path.join(RESULTS, fn)
        plt.savefig(out, dpi=130); plt.close()
        print(f"  wrote {out}")

    # convert any PGM snapshots to PNG for easy viewing (H4)
    for pgm in sorted(glob.glob(os.path.join(RESULTS, "snapshot_*.pgm"))):
        try:
            with open(pgm, "rb") as f:
                assert f.readline().strip() == b"P5"
                w, h = map(int, f.readline().split())
                _ = f.readline()
                data = f.read(w*h)
            import numpy as np
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            plt.figure(figsize=(6,6)); plt.imshow(img, cmap="viridis")
            plt.title(os.path.basename(pgm)); plt.axis("off")
            outp = pgm.replace(".pgm", ".png")
            plt.tight_layout(); plt.savefig(outp, dpi=130); plt.close()
            print(f"  wrote {outp}")
        except Exception as e:
            print(f"  (snapshot {pgm} skipped: {e})")

def main():
    paths = sorted(glob.glob(os.path.join(RESULTS, "mito4_run_seed*.csv")))
    if not paths:
        print(f"No CSVs found in {RESULTS}/. Run run.sh first."); sys.exit(1)
    runs = [load_csv(p) for p in paths]
    card = score(runs)
    scp = os.path.join(RESULTS, "SCORECARD.md")
    with open(scp, "w") as f:
        f.write(card)
    print(card)
    print(f"\nwrote {scp}")
    try:
        plot(runs)
    except ImportError:
        print("matplotlib not installed; skipping plots (scorecard still written).")

if __name__ == "__main__":
    main()
