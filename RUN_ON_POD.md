# MITO-4 Ecology — Run on the New Pod (67M organisms)

Everything to go from a fresh GPU pod to scored results in one command.

## TL;DR — copy-paste this on the pod

```bash
cd /workspace
git clone https://github.com/jasoncoryclay-commits/mito4-ecology.git
cd mito4-ecology
bash run.sh              # or:  make run
```

Prefer `make`? `make quick` smoke-tests the pod in seconds; `make run` does the full sweep;
`make arch` prints the auto-detected GPU arch; `make analyze` re-scores existing results.

### One-shot: everything in a single command

```bash
make everything            # full 5-seed sweep -> profiling pass -> combined MITO4_REPORT.md
```

This runs the science sweep, then the ncu/nsys profiling pass, then stitches **one**
`MITO4_REPORT.md` covering all three angles: science (H1–H4), throughput + GPU residency
(H5, H6), and achieved DRAM bandwidth (Nsight Compute). Profiling failures (e.g. no ncu
counter permission) do **not** abort the report — the science half is always produced.
Already have results and just want the merged doc? `make report`.

That builds the kernel, runs an **8192 × 8192 = 67,108,864 organism** lattice for 5000 ticks
across seeds 1–5, captures throughput, and (if matplotlib is present) writes plots + a scored
`SCORECARD.md` into `results/`.

## What `run.sh` does, step by step

1. Finds `nvcc` (adds `/usr/local/cuda/bin` to PATH if needed).
2. Prints the GPU via `nvidia-smi` (preflight).
3. Builds: `nvcc -O3 -arch=sm_90 mito4_kernel.cu -o mito4`.
4. Runs each seed: `./mito4 8192 8192 5000 <seed> 100` → `results/mito4_run_seed<seed>.csv`.
5. Runs `analyze.py` → `results/{population,selection,diversity}.png` + `results/SCORECARD.md`.

## Knobs (environment variables)

```bash
H=16384 W=16384 bash run.sh      # 268M organisms (fits in 80GB H100, ~3.5GB)
TICKS=10000 bash run.sh          # longer run
DUMP=250 bash run.sh             # ALSO save spatial snapshot images every 250 ticks (H4)
SEEDS="1 2 3" bash run.sh        # fewer seeds
ARCH=sm_80 bash run.sh           # A100 instead of H100 (sm_89 = L40/4090)
```

**Find your ARCH if unsure:**
```bash
nvidia-smi --query-gpu=name --format=csv,noheader
# H100 -> sm_90   |   A100 -> sm_80   |   L40/L40S/4090 -> sm_89   |   V100 -> sm_70
```

## Memory footprint (so you can size the lattice)

Per organism slot: 4 B grid + 8 B (two resource buffers) + 1 B wants ≈ **13 B**.
- 8192² (67M):   ~0.87 GB
- 16384² (268M): ~3.5 GB
- 32768² (1.07B): ~14 GB  (still fits an 80 GB H100 with room to spare)

## Get results back to me

Each run also writes a **GPU-utilization log** per seed (`results/gpu_util_seed*.csv`) sampled from
`nvidia-smi` every 0.5s. `analyze.py` folds this into the scorecard as **H6 (GPU actually saturated)**
with mean/median/p95/peak SM-util, peak memory, and mean power — the hard number on how hard the
H100 was working.

Easiest — push them to the repo:
```bash
cd /workspace/mito4-ecology
git add results/ && git commit -m "run results $(date -u +%F)" && git push
```
Then tell me and I'll pull, read the SCORECARD, and interpret the trajectories.

Or just paste the contents of `results/SCORECARD.md` and the last line of any CSV
(`# elapsed ... Gcups`) into the chat.

## Preflight checklist (30 seconds)

```bash
nvidia-smi                       # GPU present + driver OK?
nvcc --version                   # CUDA toolkit present? (need >=11 for sm_80, >=12 for sm_90)
python3 -c "import matplotlib"   # optional: enables auto-plots (else scorecard-only)
```

If `python3`/matplotlib are missing and you want plots:
```bash
pip install matplotlib numpy     # numpy only needed for snapshot PNG conversion
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvcc: command not found` | `export PATH=/usr/local/cuda/bin:$PATH` then re-run |
| `cc1plus: fatal error ... .cu: No such file` | you're in the wrong dir — `cd mito4-ecology` first |
| `nvcc fatal: Unsupported gpu architecture 'sm_90'` | CUDA too old; use `ARCH=sm_80 bash run.sh` (A100) or upgrade toolkit |
| `CUDA error ... out of memory` | lower H/W (e.g. `H=8192 W=8192`) |
| `no kernel image is available for execution` | wrong ARCH for the card — set ARCH to match `nvidia-smi` |
| plots skipped | `pip install matplotlib numpy`, then `python3 analyze.py results` |

## Expected result signature (from the validated CPU reference, scaled up)

- **Population**: fast colonization boom → plateau at ~85–92% occupancy (carrying capacity, self-organized).
- **Selection**: mean threshold drifts **down** by ~10–12 units (low-threshold lineages divide sooner and win).
- **Diversity**: lineage count settles well above 1 (no monoculture collapse) under MUT_P=0.03.
- **Throughput**: H100 should clear the ≥20 Gcups pre-committed target (H5).

Pre-committed pass/fail thresholds live in `EXPERIMENT_CARD.md`; `analyze.py` scores them automatically.

## Real efficiency: achieved memory bandwidth (`make profile`)

`nvidia-smi` SM%% (H6) only tells you a kernel was *resident*, not how *efficient* it was. For a
bandwidth-bound kernel like MITO-4, the true saturation number is **achieved DRAM bandwidth as a %%
of peak**. Get it with:

```bash
make profile                       # short run (2048², 30 ticks) under the profiler
# or:  H=4096 W=4096 TICKS=50 bash profile.sh
```

Outputs land in `profile_out/`:
- `ncu_mito4.csv` / `ncu_summary.txt` — per-kernel `dram__throughput ... pct_of_peak` (the headline
  number), plus SM throughput and occupancy. The script prints a one-line-per-kernel summary.
- `nsys_report.*` — a timeline you open in the Nsight Systems GUI.

**How to read it (roofline intuition):**
- `k_diffuse` (the resource stencil) and `k_metabolize` are memory-bound → **DRAM%% should be high
  (roughly 70–90%+ on an H100)** while SM%% stays lower. That is the *good* outcome — the card is
  moving memory as fast as it can, which is the ceiling for this workload.
- If DRAM%% is low AND SM%% is low, the kernel is latency/occupancy-bound → raise the block size or
  lattice size so more warps hide memory latency.
- `k_divide` uses `atomicCAS` on contended cells; under heavy crowding its efficiency drops — expect
  it to be the least efficient kernel, which is expected and fine (it runs 4 short passes).

**Note:** `ncu` often needs elevated GPU counter permissions. On a locked-down pod you may see
`ERR_NVGPUCTRPERM`; run the pod as root or add `--cap-add=SYS_ADMIN` / set the driver's
`NVreg_RestrictProfilingToAdminUsers=0`. `profile_out/ncu_stderr.log` will tell you if that's the issue.
The `nsys` timeline usually works without special permissions and still shows kernel time breakdown.
