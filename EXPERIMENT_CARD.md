# MITO-4 Ecology — Pre-Committed Experiment Card

**Lock this BEFORE running on the H100.** Following the program rule: *thresholds written after
seeing data are worthless.* Fill in the predictions column now; compare only after the run.

- Date locked: 2026-07-24
- Model: MITO-4 grid-resident ecology (`mito4_kernel.cu`)
- Config: 8192×8192 lattice, 5000 ticks, seeds {1,2,3,4,5}
- Params: REGEN=8, DIFFUSION=0.12, UPKEEP=30, HARVEST_FRAC=0.6, MUT_P=0.03, init density ≈1/64

## Pre-committed hypotheses & metrics

| # | Hypothesis | Metric | Pre-committed prediction | Pass condition |
|---|---|---|---|---|
| H1 | Population reaches a stable carrying capacity, not unbounded growth or extinction | `alive` at t=5000 vs t=500 | 80–95% of lattice occupied, stable within ±5% over final 1000 ticks | Non-extinct AND final-1000-tick coefficient of variation < 0.05 |
| H2 | Selection drives mean threshold DOWN (low-threshold lineages divide sooner) | `mean_thr` trajectory | Monotone-ish decrease of ≥ 5 threshold units from t=0 to t=5000 | Δmean_thr ≤ −5 across the run, consistent across ≥4/5 seeds |
| H3 | Lineage diversity does NOT collapse to a monoculture under mutation | `lineages` (distinct thresholds present) | Stabilizes at ≥ 20 distinct thresholds, never hits 1 | min(lineages over final 1000 ticks) ≥ 10 in ≥4/5 seeds |
| H4 | Large lattice shows spatial structure absent at 256×256 | grid snapshot (add `--dump` if desired) | Visible fronts/patches/waves during colonization phase | Qualitative: reviewer confirms non-uniform spatial pattern |
| H5 | Throughput justifies the H100 | printed `billion cell-updates/sec` | ≥ 20 billion cell-updates/sec on H100 at 8192² | Measured throughput ≥ 20 Gcups |

## Analytic sanity anchors (independent of the run)

- **Colonization doubling time** (empty-lattice regime, abundant resource): a seed cell harvests
  ~`floor(RES_CAP*0.5*0.6)` ≈ 76 energy/tick early on, so it crosses a threshold T≈100 in ~2 ticks
  and divides — predicts explosive early growth, matching the CPU reference (1.3k → 64k in 15 ticks).
- **Starvation boundary**: a cell in a depleted patch (resource ≈ 0) gains ≈0 and pays UPKEEP=30/tick,
  so it dies within `ceil(E/30)` ticks — bounds the boom-bust crash depth.
- **Selection direction**: lower T ⇒ crosses `E≥T` sooner ⇒ more divisions ⇒ frequency rises. This is
  why H2 predicts a *downward* drift, not upward.

## What to bring back

The 5 CSV files (`mito4_run_seed{1..5}.csv`). I'll plot: (a) population vs tick, (b) mean-threshold
selection curve, (c) lineage-diversity trajectory, and score each hypothesis pass/fail against the
predictions locked above.

## Failure modes to watch (not bugs — findings)

- **Instant extinction** → REGEN too low or UPKEEP too high; the environment can't sustain life.
  Finding, not failure: it maps the habitability boundary.
- **Full saturation then freeze** → carrying capacity pinned at 100%; try higher UPKEEP or lower REGEN
  to reintroduce boom-bust.
- **Diversity collapse to 1** → selection too strong / mutation too weak; raise MUT_P. Tests H3's edge.
