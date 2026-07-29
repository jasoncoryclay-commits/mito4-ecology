#!/usr/bin/env python3
"""
MITO-4 ECOLOGY — combined report generator.
Stitches the science scorecard (results/SCORECARD.md), the GPU-utilization summary,
and the ncu achieved-DRAM-bandwidth numbers (profile_out/ncu_mito4.csv) into a
single MITO4_REPORT.md.

Usage:  python3 combine_report.py [results_dir] [profile_dir] [out_file]
Defaults: results  profile_out  MITO4_REPORT.md

Runs analyze.py first if the scorecard is missing. Degrades gracefully if
profiling artifacts are absent (science half still reports).
"""
import sys, os, csv, glob, subprocess, collections, datetime

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
PROFILE = sys.argv[2] if len(sys.argv) > 2 else "profile_out"
OUT     = sys.argv[3] if len(sys.argv) > 3 else "MITO4_REPORT.md"

def ensure_scorecard():
    scp = os.path.join(RESULTS, "SCORECARD.md")
    if not os.path.exists(scp):
        if glob.glob(os.path.join(RESULTS, "mito4_run_seed*.csv")):
            print("[combine] SCORECARD.md missing -> running analyze.py")
            try:
                subprocess.run([sys.executable, "analyze.py", RESULTS], check=False)
            except Exception as e:
                print(f"[combine] analyze.py failed: {e}")
    return scp

def read_scorecard(scp):
    if os.path.exists(scp):
        return open(scp).read().strip()
    return "_No SCORECARD.md found — run the sweep (`make run`) first._"

def parse_ncu(path):
    """Parse ncu --csv output into {kernel: {metric: value}}. Robust to quoting."""
    if not os.path.exists(path):
        return None
    try:
        rows = list(csv.DictReader(open(path)))
    except Exception:
        return None
    by = collections.defaultdict(dict)
    for r in rows:
        # ncu csv columns vary by version; handle common ones
        k = r.get("Kernel Name") or r.get("Kernel") or ""
        m = r.get("Metric Name") or ""
        v = r.get("Metric Value") or ""
        if k and m:
            by[k.strip()][m.strip()] = v.strip()
    return by or None

def ncu_section(by):
    if not by:
        return ("## Achieved memory bandwidth (Nsight Compute)\n\n"
                "_No `profile_out/ncu_mito4.csv` found, or ncu lacked GPU-counter permission._\n"
                "_Run `make profile` (as root or with `--cap-add=SYS_ADMIN` if you hit ERR_NVGPUCTRPERM)._\n")
    out = ["## Achieved memory bandwidth (Nsight Compute)\n",
           "For a bandwidth-bound kernel, DRAM %-of-peak is the real saturation number "
           "(high DRAM%% with lower SM%% is the *good*, at-the-roofline outcome).\n",
           "| kernel | DRAM % peak | SM % peak | occupancy % | DRAM bytes |",
           "|---|---|---|---|---|"]
    for k, mm in by.items():
        kn = k.split("(")[0][:38]
        dram = mm.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", "?")
        sm   = mm.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "?")
        occ  = mm.get("sm__warps_active.avg.pct_of_peak_sustained_active", "?")
        byt  = mm.get("dram__bytes.sum", "?")
        out.append(f"| {kn} | {dram} | {sm} | {occ} | {byt} |")
    out.append("")
    return "\n".join(out)

def gpu_util_note():
    logs = sorted(glob.glob(os.path.join(RESULTS, "gpu_util_seed*.csv")))
    if not logs:
        return "_No nvidia-smi utilization logs found (H6 in scorecard will read REVIEW)._\n"
    return (f"_Per-seed `nvidia-smi` utilization logs present ({len(logs)} files); "
            "the H6 row and GPU-utilization table above are derived from them._\n")

def main():
    scp = ensure_scorecard()
    scorecard = read_scorecard(scp)
    ncu = parse_ncu(os.path.join(PROFILE, "ncu_mito4.csv"))
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    parts = []
    parts.append("# MITO-4 Ecology — Combined Run Report\n")
    parts.append(f"_Generated {ts}. Sources: `{RESULTS}/`, `{PROFILE}/`._\n")
    parts.append("This single report covers three independent saturation/behavior angles:\n"
                 "1. **Science** — did life self-organize? (scorecard H1–H4)\n"
                 "2. **Throughput & residency** — Gcups + nvidia-smi SM%%/power (H5, H6)\n"
                 "3. **Efficiency** — achieved DRAM bandwidth %% of peak (Nsight Compute)\n")
    parts.append("---\n")
    parts.append(scorecard)
    parts.append("\n" + gpu_util_note())
    parts.append("\n---\n")
    parts.append(ncu_section(ncu))
    parts.append("\n---\n")
    parts.append("## Artifact index\n")
    for pat, desc in [
        (f"{RESULTS}/mito4_run_seed*.csv", "raw trajectories"),
        (f"{RESULTS}/gpu_util_seed*.csv", "GPU utilization samples"),
        (f"{RESULTS}/*.png", "trajectory plots"),
        (f"{RESULTS}/snapshot_*.png", "spatial snapshots (H4)"),
        (f"{PROFILE}/ncu_mito4.csv", "ncu bandwidth metrics"),
        (f"{PROFILE}/nsys_report.*", "nsys timeline"),
    ]:
        files = glob.glob(pat)
        if files:
            parts.append(f"- `{pat}` — {desc} ({len(files)} file(s))")
    parts.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(parts))
    print(f"[combine] wrote {OUT}")

if __name__ == "__main__":
    main()
