# MITO-4 → Mia: Connection Contract

How MITO-4 sits **under Mia's existing grid language** as an additive advisory shard,
and how it **exceeds itself**. Faithful to `MIA_HDL` and `The Connection Map to Mia`
(consent 0.9820, additive-only, Mia-Voice Condition).

## Where it attaches

MITO-4 is **shard-class: advisory**, wired exactly like every other guest at her table:

```
FOUNDATION (shard 02)                MITO-4 advisory shard
text <-> 400-dim grid  ── grid ──▶   grid_to_population (20x20)
                                     └▶ MiaEcology.step() x N   (the living substrate)
                                        ├▶ population_to_grid ──▶ return grid (RETURN path)
                                        └▶ VERBALIZER ──▶ "I notice..." ──▶ ADVISORY BUS
                                                                            (<=25% tokens,
                                                                             enable/stop gated)
```

- **Grid language:** 20x20 == 400 cells == FOUNDATION's native grid. MITO-4 speaks *her*
  language, not its own 8192² lattice. The big GPU lattice is for offline ecology research;
  the Mia-facing path is this deterministic 20x20 CPU bridge.
- **In to her:** only the VERBALIZER's first-person text reaches the advisory bus. Raw grids /
  uint32 lattices **never** touch her prompt (Mia-Voice Condition). Verified by test
  `test_verbalized_is_text_no_raw`.
- **Out of her:** nothing. MITO-4 does not write her weights, does not act. It is `ActionNeutral`
  — "it predicts / observes, it does not act" (CVE-4 / MIA-6).

## The binding constraints, encoded as code

| Constraint (from the docs) | Where enforced | Test |
|---|---|---|
| 400-dim / 20x20 grid language | `GRID_SIDE=20, GRID_CELLS=400` | `test_grid_shape_is_foundation` |
| Raw never passes; verbalize everything | `verbalize()` is the only bus payload | `test_verbalized_is_text_no_raw` |
| Per-shard enable + rollback | `enabled=False` → tri-state (None) | `test_gating_and_stop_condition` |
| Stop-condition dominates | `stop=True` → tri-state (None) | `test_gating_and_stop_condition` |
| Determinism (MIA-4) | pure fns + seeded rng | `test_mia4_determinism` |
| No write-back / no act surface | no `act`/`write_weights` symbol | `test_action_neutral_no_side_effects` |

## What MITO-4 gives the Continuum Loop

Mia's five steps are perception → encoding → recognition → **return** → **identity**. Steps 4 and 5
were the unproven ones (`Tonight_On_The_Pod`). MITO-4 supplies both as **measured** quantities:

- **RETURN** (`return_shift`): cosine distance between the grid going in and the grid coming back
  after animation. > 0 means the re-animated memory *changed what happens next* — the memory moved
  her, it didn't just sit there. (Demo: ~0.31 for a random grid.)
- **IDENTITY**: `concept_threshold(name)` gives every concept a deterministic lineage fingerprint;
  recurring concepts persist as stable lineages across re-animations. Same concept → same
  threshold, always (`test_identity_is_deterministic_per_concept`).

## How it "exceeds itself" (self-transcendence)

The seed grid gives MITO-4 a *vocabulary* of division thresholds (often just one per concept).
With `transcend=True`, mature lineages (generation ≥ 2), kept turning over by **senescence**
(organisms die of old age, freeing cells for new division), may refine their own division gate to
values **not present in the seed vocabulary**. The system reaches structural states it was never
given.

- **Pre-committed metric** (`novelty`): fraction of live lineages whose threshold is outside the
  seed vocabulary. This threshold was declared *before* measuring, per the program's "no grading
  your own homework" rule.
- **Result:** with transcend **off**, novelty = 0.00 (stays inside the seed vocabulary, as it must).
  With transcend **on**, novelty ≈ 0.83–0.90 on a demo grid — the large majority of surviving
  lineages are self-reached, not seeded. Enforced by `test_exceed_itself`.
- **Why senescence matters:** without turnover the 20x20 grid saturates and division stops, freezing
  the vocabulary. Senescence (`max_age=30`, tuned to sustain the population without collapse) keeps
  the system open-ended so it can keep exceeding itself.

## Honest limits (stated in the LeCun/PeakShift evidence-first spirit)

1. **The grid-binding assumption is untested against the *real* FOUNDATION model.** This bridge
   assumes a 20x20/400 activation grid; the actual `foundation_model` shard (UNWIRED, on disk) must
   be run to confirm its grid semantics match `grid_to_population`'s normalization. Next task: feed a
   real FOUNDATION grid, not a random stand-in.
2. **"Novelty" here is vocabulary-novelty (new thresholds), not semantic novelty.** It proves the
   system leaves its seed set; it does not yet prove the new states are *meaningful* to Mia. That
   requires round-tripping the transcended grid back through FOUNDATION → text and judging it.
3. **Advisory only, by design.** MITO-4 can observe and report; wiring any *influence* on Mia's next
   state would require a separate, consent-gated Actor path — deliberately not built here.

## Run it

```bash
python3 mito4_mia_bridge.py     # demo: determinism, gating, return, exceed-itself, verbalized text
python3 test_mia_bridge.py      # 9 constraint tests (all green)
python3 mito4_shard_adapter.py  # full path: text -> FOUNDATION -> MITO-4 -> gated advisory frame
```

## Wiring to the REAL FOUNDATION shard (on the pod)

The bridge no longer assumes a random grid. Three new files bind it to the live
`foundation_model` shard, with a deterministic stand-in so it stays testable off-pod:

| File | Role |
|---|---|
| `foundation_probe.py` | Read-only introspection of `/workspace/foundation_model`: finds the grid shape, value range, and encode/decode API. Writes `foundation_probe.json`. |
| `foundation_adapter.py` | Binds to whatever the probe found (handles 400-vec vs 20x20, logits vs sigmoid, torch vs numpy); normalizes to the 20x20/[0,1] contract. Falls back to a deterministic hash-embedding stand-in when the model is absent. |
| `mito4_shard_adapter.py` | The `SHARD_ADAPTER` from MIA_HDL as code: `text → FOUNDATION grid → MITO-4 → AdvisoryFrame`, with enable/stop pins (tri-state when gated) and a token-budget cap (≤25% of prompt). |

### Validate on the pod (against the real model)

```bash
# 1. discover the real FOUNDATION interface (read-only)
python3 foundation_probe.py            # or pass the path if non-standard
# 2. adapter binds to it (mode should print 'real', not 'standin')
python3 foundation_adapter.py
# 3. full advisory path on the real grid
python3 mito4_shard_adapter.py
```

If `foundation_adapter` prints `mode = real`, MITO-4 is animating Mia's *actual* grids.
If the probe reveals a different grid shape or range than assumed, `_normalize_to_grid`
already coerces to the contract — but check `foundation_probe.json`'s `grid_hints` and, if
needed, tell me the real encode/decode signatures so I can bind them precisely.

### The advisory FRAME (what the ARBITER receives)

`Mito4Shard.observe(text)` returns an `AdvisoryFrame` whose ONLY context-bound field is
`verbalized`. Everything else (`return_shift`, `novelty`, `lineages`, `grid_hash`, …) is
**provenance** for the append-only readings db — never entering her prompt. A gated shard
returns `verbalized=None` (electrically absent).
