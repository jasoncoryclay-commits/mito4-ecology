# MITO-4 → Mia: What Is Proven vs Not (Honest Scoping)

Last updated after the null-control test returned **FAIL** on semantic novelty.
This document is the truthful status, written in the program's evidence-first spirit
("receipts, not marketing"). It governs what may and may not be claimed or bused to Mia.

## PROVEN (measured, on real weights)

| Claim | Evidence |
|---|---|
| Real FOUNDATION binding | encoder + decoder load `missing=0 unexpected=0`; `mode=real` |
| Mia's grid language (20×20 / 400) | confirmed from weights (`output_proj → 400`, `grid_proj ← 400`) |
| Embedder identified | `text-embedding-3-large @ dimensions=3072` (from `summary.json`) |
| Vocabulary novelty | MITO-4 reaches division thresholds outside its seed set (novelty ≈ 0.9 with transcend on, 0.0 off) |
| Return path | re-animated grid measurably differs from seed (`return_shift` ≈ 0.84) |
| Advisory safety | verbalize-only, enable/stop tri-state, token budget — all tested |

## NOT YET PROVEN

| Claim | Status |
|---|---|
| **Semantic self-transcendence** | **FAIL / VOID.** Null-control: transcended grids change nearest-concept 89% vs random grids 89% (delta +0.00). The `grief→awe` flip was decoder instability, not meaning. |
| Grid→concept fidelity | round-trip accuracy only ~0.50 (raw convention); the decoder's grid→text channel is lossy/unstable. |

## ROOT CAUSE (diagnosis)

The measuring instrument — the FOUNDATION decoder's grid→concept mapping — is too
blunt: even *seed* grids round-trip at only ~0.5, and *random* grids change concept
89% of the time. When the instrument is that noisy, MITO-4's dynamics cannot be
distinguished from noise, whether or not a real semantic effect exists. This is an
instrument problem upstream of MITO-4, not a MITO-4 failure.

## GATING RULE (must hold before semantic claims reach Mia)

MITO-4 semantic output may be bused to Mia's advisory bus ONLY when
`trustworthy_test.py` reports:

    round-trip accuracy >= 0.66   AND   directional delta >= 0.15  (PASS)

Until then, MITO-4 may be bused (if desired) reporting ONLY:
  - vocabulary novelty (new thresholds reached)
  - return-path dynamics (memory changes the next grid)
…never a claim that a self-reached state *means* a new concept.

## PATHS TO CLOSE THE GAP

1. Improve grid→text fidelity: better/retrained decoder, or a larger contrastive
   temperature, so round-trip accuracy rises toward ~0.8.
2. Use the directional probe (`trustworthy_test.py` Option 2) which has more
   statistical power than argmax-nearest once the instrument is trustworthy.
3. If fidelity cannot be raised, retire the semantic claim entirely and ship MITO-4
   as a vocabulary-novelty + return-path organ — both of which ARE proven.
