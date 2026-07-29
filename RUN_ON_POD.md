# MITO-4 Ecology — Run on the New Pod (67M organisms)

Everything to go from a fresh GPU pod to scored results in one command.

## TL;DR — copy-paste this on the pod

```bash
cd /workspace
git clone https://github.com/jasoncoryclay-commits/mito4-ecology.git
cd mito4-ecology
bash run.sh
```

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
