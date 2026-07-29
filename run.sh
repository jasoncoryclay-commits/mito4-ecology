#!/usr/bin/env bash
# MITO-4 ECOLOGY — one-command H100 run for the 67M-organism experiment.
# Usage:   bash run.sh                # default: 8192x8192 (~67M), 5000 ticks, seeds 1..5
#          H=16384 W=16384 bash run.sh   # override lattice size (env vars)
#          DUMP=250 bash run.sh          # also write snapshot images every 250 ticks (H4)
set -euo pipefail

# ---- config (override via env) ----
H="${H:-8192}"
W="${W:-8192}"
TICKS="${TICKS:-5000}"
LOG_EVERY="${LOG_EVERY:-100}"
DUMP="${DUMP:-0}"               # 0 = no images; e.g. 250 = image every 250 ticks
SEEDS="${SEEDS:-1 2 3 4 5}"
OUTDIR="${OUTDIR:-results}"

# ---- arch: auto-detect from the GPU name unless caller pinned ARCH ----
detect_arch() {
  local name
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  case "$name" in
    *H100*|*H200*)      echo sm_90 ;;
    *A100*)             echo sm_80 ;;
    *L40*|*4090*|*L4*)  echo sm_89 ;;
    *V100*)             echo sm_70 ;;
    *)                  echo sm_90 ;;
  esac
}
ARCH="${ARCH:-$(detect_arch)}"  # sm_90=H100  sm_80=A100  sm_89=L40/4090  sm_70=V100

echo "=============================================="
echo " MITO-4 ECOLOGY RUN"
echo "   lattice : ${H} x ${W} = $((H*W)) organism slots"
echo "   ticks   : ${TICKS}   seeds: ${SEEDS}   arch: ${ARCH}"
echo "   dump    : ${DUMP}    outdir: ${OUTDIR}"
echo "=============================================="

# ---- locate nvcc ----
if ! command -v nvcc >/dev/null 2>&1; then
  export PATH=/usr/local/cuda/bin:$PATH
fi
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found. Install CUDA toolkit or fix PATH."; exit 1; }
echo "[1/4] nvcc: $(nvcc --version | grep release || true)"

# ---- preflight: GPU visible? ----
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[preflight] GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[preflight] WARNING: nvidia-smi not found; continuing (kernel will error if no GPU)."
fi

# ---- build ----
echo "[2/4] building (arch=${ARCH}) ..."
nvcc -O3 -arch="${ARCH}" mito4_kernel.cu -o mito4
echo "      built ./mito4"

# ---- GPU utilization sampler (background) ----
# Polls nvidia-smi every 0.5s while a run is in flight -> per-seed util CSV.
GPU_LOG_PID=""
start_gpu_log() {
  local out="$1"
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  echo "ts_s,util_gpu_pct,util_mem_pct,mem_used_mib,power_w,sm_clock_mhz" > "${out}"
  (
    while true; do
      line=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
             --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      printf '%s,%s\n' "$(date +%s.%N)" "${line}" >> "${out}"
      sleep 0.5
    done
  ) &
  GPU_LOG_PID=$!
}
stop_gpu_log() {
  [ -n "${GPU_LOG_PID}" ] && kill "${GPU_LOG_PID}" 2>/dev/null || true
  GPU_LOG_PID=""
}
trap stop_gpu_log EXIT

# ---- run sweep ----
mkdir -p "${OUTDIR}"
echo "[3/4] running seed sweep ..."
for s in ${SEEDS}; do
  csv="${OUTDIR}/mito4_run_seed${s}.csv"
  gpu="${OUTDIR}/gpu_util_seed${s}.csv"
  echo "  -> seed ${s}  (log -> ${csv}, gpu -> ${gpu})"
  start_gpu_log "${gpu}"
  ./mito4 "${H}" "${W}" "${TICKS}" "${s}" "${LOG_EVERY}" "${DUMP}" | tee "${csv}"
  stop_gpu_log
done
# move any snapshot images into outdir
shopt -s nullglob
for f in snapshot_seed*.pgm; do mv "$f" "${OUTDIR}/"; done

# ---- analyze (optional; needs python3 + matplotlib) ----
echo "[4/4] analyzing ..."
if command -v python3 >/dev/null 2>&1 && python3 -c "import matplotlib" 2>/dev/null; then
  python3 analyze.py "${OUTDIR}"
else
  echo "      (skipping plots: python3+matplotlib not available)"
  echo "      run later with:  python3 analyze.py ${OUTDIR}"
fi

echo "=============================================="
echo " DONE. Results in ${OUTDIR}/"
echo "   - mito4_run_seed*.csv     raw trajectories"
echo "   - *.png                    plots (if matplotlib present)"
echo "   - SCORECARD.md             hypothesis pass/fail (if analyze ran)"
echo "   - snapshot_*.pgm           spatial images (if DUMP>0)"
echo "=============================================="
