#!/usr/bin/env python3
"""Tests for mito4_mia_bridge — enforces Mia's binding constraints as assertions."""
import numpy as np
import mito4_mia_bridge as m

def _grid(seed=42):
    return np.random.default_rng(seed).random((m.GRID_SIDE, m.GRID_SIDE))

def test_grid_shape_is_foundation():
    assert m.GRID_CELLS == 400 and m.GRID_SIDE == 20  # FOUNDATION 400-dim grid

def test_roundtrip_shapes():
    g = _grid()
    pop = m.grid_to_population(g, "grief")
    assert pop.shape == (20, 20) and pop.dtype == np.uint32
    back = m.population_to_grid(pop)
    assert back.shape == (20, 20) and back.min() >= 0.0 and back.max() <= 1.0

def test_identity_is_deterministic_per_concept():
    # same concept -> same lineage threshold fingerprint (IDENTITY)
    assert m.concept_threshold("grief") == m.concept_threshold("grief")
    assert m.concept_threshold("grief") != m.concept_threshold("joy")

def test_mia4_determinism():
    # MIA-4: same grid + seed -> byte-identical
    g = _grid()
    a = m.mito4_advisory(g, "grief", seed=3)
    b = m.mito4_advisory(g, "grief", seed=3)
    assert a["grid_hash"] == b["grid_hash"]

def test_gating_and_stop_condition():
    g = _grid()
    assert m.mito4_advisory(g, "grief", enabled=False)["verbalized"] is None
    assert m.mito4_advisory(g, "grief", stop=True)["verbalized"] is None

def test_verbalized_is_text_no_raw():
    # only first-person text may reach the advisory bus; no arrays/tensors
    r = m.mito4_advisory(_grid(), "grief")
    assert isinstance(r["verbalized"], str) and r["verbalized"].startswith("I notice")

def test_return_path_moves_the_next_thought():
    r = m.mito4_advisory(_grid(), "grief", ticks=30)
    assert r["return_shift"] > 0.0   # the memory changes the returned grid

def test_exceed_itself():
    g = _grid()
    base = m.mito4_advisory(g, "grief", ticks=40, seed=3, transcend=False)
    tran = m.mito4_advisory(g, "grief", ticks=40, seed=3, transcend=True)
    assert base["novelty"] == 0.0            # without transcend, stays in seed vocab
    assert tran["novelty"] > 0.0             # with transcend, exceeds the seed vocab
    assert tran["novelty"] >= base["novelty"]

def test_action_neutral_no_side_effects():
    # the module returns data; it must not expose any 'act'/'write' surface
    assert not hasattr(m, "act") and not hasattr(m, "write_weights")

if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL  {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)
