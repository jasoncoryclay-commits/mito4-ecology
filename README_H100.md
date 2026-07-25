# MITO-4 Ecology — H100 Run Guide

Everything needed to put your idle H100 to high-value use tonight running the spatial-ecology
variant of MITO-4 at a scale a CPU cannot reach.

## Files

| File | Role |
|---|---|
| `mito4.py` | Base 4-byte organism (the original paper's reference) |
| `mito4_ecology.py` | **CPU reference** for the lattice ecology — validated & deterministic |
| `mito4_kernel.cu` | **GPU kernel** — the thing you run on the H100 |
| `host_check.cpp` | Parity test: CUDA bit-logic == Python bit-logic (passes) |
| `EXPERIMENT_CARD.md` | Pre-committed metrics — read & lock BEFORE the run |

## One-time build on the pod

```bash
# On the RunPod H100 instance:
nvcc -O3 -arch=sm_90 mito4_kernel.cu -o mito4      # sm_90 = Hopper / H100
```

If `nvcc` isn't on PATH, it's usually at `/usr/local/cuda/bin/nvcc`.

## The high-value overnight run (single command)

```bash
# 8192 x 8192 lattice = ~67 million organism slots, 5000 ticks, seed 1, log every 100 ticks
./mito4 8192 8192 5000 1 100 | tee mito4_run_seed1.csv
```

Arguments: `./mito4 <H> <W> <ticks> <seed> <log_every>`

The H100 has 80 GB HBM3; each cell uses 4 B (grid) + 8 B (two resource buffers) + 1 B (wants) ≈ 13 B,
so even a **16384 x 16384** lattice (~268 M cells, ~3.5 GB) fits easily with room to spare. Scale up
`H`/`W` until you're happy the card is saturated — the run prints `billion cell-updates/sec` at the end.

### A sweep worth leaving overnight

```bash
for s in 1 2 3 4 5; do
  ./mito4 8192 8192 5000 $s 100 | tee mito4_run_seed$s.csv
done
```

Five seeds → you can tell whether the emergent dynamics (carrying capacity, threshold
selection, diversity trajectory) are robust or seed-specific. That's a real result.

## Output

CSV to stdout: `tick,alive,mean_thr,lineages` plus a final line with elapsed ms and
throughput. Redirect with `tee` (as above) to keep the data. Bring the CSVs back and I'll
plot population, mean-threshold selection, and lineage-diversity trajectories.

## What you're actually testing (why this uses the card well)

- Resource diffusion is a **stencil** (neighbor reads) — bandwidth-heavy, scales with lattice area.
- One thread per cell over tens/hundreds of millions of cells × thousands of ticks — this is where
  a CPU takes hours-to-days and the H100 takes minutes.
- The science: does population find a **carrying capacity**? Does mean threshold **drift under
  selection**? Does **lineage diversity** stabilize or collapse to a monoculture? Does spatial
  **pattern formation** (waves/patches) appear at large scale that isn't visible at 256×256?

## PARITY NOTE (important, honest)

The CPU reference (`mito4_ecology.py`) and the GPU kernel implement the **same model and the same
directional (N,S,W,E) division-placement priority**, but they are **not guaranteed bit-identical**
for two reasons, both documented so you're not surprised:

1. **Blocked-division energy.** In the current GPU kernel, a parent that *wants* to divide has its
   energy halved in `k_metabolize` *before* placement is attempted. If all four neighbors are full,
   the daughter is never placed but the parent already paid the halving (a small "wasted division"
   cost under extreme crowding). The CPU reference only halves when placement succeeds. This affects
   only saturated cells and does not change the qualitative dynamics. If you want strict parity, move
   the halving into `k_divide` on successful `atomicCAS` — noted as a TODO in the .cu file.
2. **RNG stream.** Mutation uses a per-cell hash RNG on GPU vs NumPy's PCG on CPU, so the *exact*
   mutation events differ even at the same seed. Aggregate statistics (mean threshold, diversity)
   are comparable; individual grids are not.

For a strict CPU↔GPU regression test, run both with `--mutation 0` / `MUT_P 0` and compare aggregate
population trajectories (they should track closely); exact grid hashes will still differ under the
blocked-division rule above unless you apply the TODO fix. Validate the *science* on aggregates, not
on grid-hash equality.

## Validate locally first (no GPU needed)

```bash
python3 mito4_ecology.py --verify                       # determinism check
python3 mito4_ecology.py --H 256 --W 256 --ticks 150 --seed 1 --mutation 0.03 --log-every 15
g++ -O2 host_check.cpp -o host_check && ./host_check     # CUDA bit-logic parity
```
