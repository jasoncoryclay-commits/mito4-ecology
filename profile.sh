#!/usr/bin/env bash
# MITO-4 ECOLOGY — memory-bandwidth & timeline profiling.
# For a bandwidth-bound kernel, ACHIEVED DRAM BANDWIDTH (% of peak) is the true
# "is the GPU saturated" number — far more meaningful than nvidia-smi SM%.
#
# Usage:  bash profile.sh                 # short profiling run (small lattice, few ticks)
#         H=8192 W=8192 TICKS=50 bash profile.sh
#
# Produces (in profile_out/):
#   ncu_mito4.csv        per-kernel metrics incl. dram__throughput.avg.pct_of_peak_sustained_elapsed
#   ncu_summary.txt      human-readable per-kernel bandwidth/occupancy summary
#   nsys_report.*        timeline (if nsys present) — open in Nsight Systems GUI
set -euo pipefail

H="${H:-2048}"            # keep small: profiling replays kernels and is SLOW
W="${W:-2048}"
TICKS="${TICKS:-30}"      # a handful of ticks is plenty to profile the kernels
SEED="${SEED:-1}"
detect_arch() {
  local name; name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  case "$name" in
    *H100*|*H200*) echo sm_90 ;; *A100*) echo sm_80 ;;
    *L40*|*4090*|*L4*) echo sm_89 ;; *V100*) echo sm_70 ;; *) echo sm_90 ;;
  esac
}
ARCH="${ARCH:-$(detect_arch)}"
OUT="${OUT:-profile_out}"

command -v nvcc >/dev/null 2>&1 || export PATH=/usr/local/cuda/bin:$PATH
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found."; exit 1; }

mkdir -p "${OUT}"
echo "[build] nvcc -O3 -arch=${ARCH} (with -lineinfo for source correlation)"
nvcc -O3 -arch="${ARCH}" -lineinfo mito4_kernel.cu -o mito4_prof

APP=(./mito4_prof "${H}" "${W}" "${TICKS}" "${SEED}" 999 0)   # log_every=999 -> minimal stat overhead

# ---------- Nsight Compute: achieved DRAM bandwidth ----------
if command -v ncu >/dev/null 2>&1; then
  echo "[ncu] profiling kernels for achieved DRAM bandwidth (this replays kernels; be patient)..."
  # Key metrics:
  #   dram__throughput.avg.pct_of_peak_sustained_elapsed   -> % of peak DRAM BW achieved
  #   gpu__time_duration.sum                               -> kernel time
  #   sm__throughput.avg.pct_of_peak_sustained_elapsed     -> compute pipe utilization
  #   launch__occupancy_limit_*                            -> occupancy limiters
  METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
gpu__time_duration.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active"
  # Profile only a few launches of each kernel to keep it fast.
  ncu --target-processes all \
      --launch-count 12 \
      --metrics "${METRICS}" \
      --csv "${APP[@]}" > "${OUT}/ncu_mito4.csv" 2> "${OUT}/ncu_stderr.log" || {
        echo "[ncu] non-zero exit — see ${OUT}/ncu_stderr.log (often just permission: needs --cap-add or root)"; }
  # also a readable summary section view
  ncu --launch-count 12 --section MemoryWorkloadAnalysis --section Occupancy \
      "${APP[@]}" > "${OUT}/ncu_summary.txt" 2>>"${OUT}/ncu_stderr.log" || true
  echo "[ncu] wrote ${OUT}/ncu_mito4.csv and ${OUT}/ncu_summary.txt"
  # quick extraction of the headline bandwidth number per kernel
  python3 - "${OUT}/ncu_mito4.csv" <<'PY' 2>/dev/null || true
import csv, sys, collections
path=sys.argv[1]
try:
    rows=list(csv.DictReader(open(path)))
except Exception as e:
    print(f"(could not parse ncu csv: {e})"); sys.exit(0)
by=collections.defaultdict(dict)
for r in rows:
    k=r.get("Kernel Name","") or r.get('"Kernel Name"',"")
    m=r.get("Metric Name",""); v=r.get("Metric Value","")
    if k and m: by[k][m]=v
print("\n=== Achieved DRAM bandwidth (% of peak) per kernel ===")
for k,mm in by.items():
    bw=mm.get("dram__throughput.avg.pct_of_peak_sustained_elapsed","?")
    sm=mm.get("sm__throughput.avg.pct_of_peak_sustained_elapsed","?")
    occ=mm.get("sm__warps_active.avg.pct_of_peak_sustained_active","?")
    kn=k.split("(")[0][:40]
    print(f"  {kn:42s} DRAM {bw:>7}%  SM {sm:>7}%  occ {occ:>7}%")
print("\nInterpretation: MITO-4 is bandwidth-bound -> expect DRAM%% high, SM%% lower.")
PY
else
  echo "[ncu] Nsight Compute (ncu) not found. Install CUDA toolkit's ncu, or run on a pod that has it."
  echo "      Bandwidth cannot be measured without ncu; nsys timeline (below) still helps."
fi

# ---------- Nsight Systems: timeline ----------
if command -v nsys >/dev/null 2>&1; then
  echo "[nsys] capturing timeline..."
  nsys profile --force-overwrite true -o "${OUT}/nsys_report" \
       --stats=true "${APP[@]}" > "${OUT}/nsys_stats.txt" 2>&1 || \
       echo "[nsys] non-zero exit — see ${OUT}/nsys_stats.txt"
  echo "[nsys] wrote ${OUT}/nsys_report.* and ${OUT}/nsys_stats.txt"
else
  echo "[nsys] Nsight Systems (nsys) not found — skipping timeline."
fi

echo "=============================================="
echo " PROFILING DONE. See ${OUT}/"
echo "   ncu_mito4.csv / ncu_summary.txt  -> achieved DRAM bandwidth % (the real number)"
echo "   nsys_report.*                    -> open in Nsight Systems for the timeline"
echo "=============================================="
