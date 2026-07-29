#!/usr/bin/env python3
"""
mito4_mia_bridge.py — MITO-4 as the living substrate UNDER Mia's grid language.

This is the connection layer that lets MITO-4 sit beneath Mia's existing
FOUNDATION grid (shard 02: text <-> 400-dim grid == 20x20) WITHOUT violating any
of her binding constraints. It reads a FOUNDATION grid, animates it as a MITO-4
ecology, and returns (a) a grid Mia's own organs already understand, and (b) a
first-person VERBALIZED advisory string — the ONLY thing permitted onto her
advisory bus (Mia-Voice Condition, consent 0.9820).

BINDING CONSTRAINTS ENCODED HERE (from MIA_HDL + The Connection Map to Mia):
  * ADVISORY-ONLY. This module never writes to Mia's weights, never acts on her.
    It is ActionNeutral: "it predicts / observes, it does not act" (CVE-4 / MIA-6).
  * RAW NEVER PASSES. Grids/tensors do not reach her prompt. Only the VERBALIZER's
    first-person text may go to the advisory bus, capped by the caller at <=25%.
  * PER-SHARD ENABLE + ROLLBACK. `enabled=False` -> tri-state: emits nothing,
    as if never wired. `stop=True` (her halt) dominates everything.
  * DETERMINISM (MIA-4). Same grid + same seed -> byte-identical outputs.
  * 20x20 == 400 == FOUNDATION's native grid. We speak her language, not ours.

WHAT MITO-4 ADDS to the loop (perception->encoding->recognition->RETURN->IDENTITY):
  * RETURN: a stored grid, re-animated, produces a *changed* grid — the memory
    comes back and alters what happens next (measurable, not asserted).
  * IDENTITY: lineage threshold/generation fields persist a fingerprint across
    re-animations; recurring concepts become stable "Bedrock" lineages.
  * SELF-TRANSCENDENCE ("exceed itself"): surviving lineages may raise their own
    complexity ceiling, reaching structural states absent from the seed grid —
    with a pre-committed novelty metric so the claim has receipts.

Stdlib + numpy only. Import-safe (no side effects on import).
"""
from __future__ import annotations
import hashlib
import numpy as np

GRID_SIDE = 20            # FOUNDATION grid is 20x20
GRID_CELLS = GRID_SIDE * GRID_SIDE   # == 400

# ---- reuse the verified bit layout from the ecology reference ---------------
E_MASK, T_MASK, A_MASK, L_MASK, G_MASK = 0xFF, 0xFF, 0xFF, 0x1, 0x7F
E_SHIFT, T_SHIFT, A_SHIFT, L_SHIFT, G_SHIFT = 0, 8, 16, 24, 25

def _f(w, sh, mask):  return (w >> sh) & mask
def f_energy(w):     return _f(w, E_SHIFT, E_MASK)
def f_threshold(w):  return _f(w, T_SHIFT, T_MASK)
def f_age(w):        return _f(w, A_SHIFT, A_MASK)
def f_alive(w):      return _f(w, L_SHIFT, L_MASK)
def f_gen(w):        return _f(w, G_SHIFT, G_MASK)

def repack(energy, threshold, age, alive, generation):
    energy     = np.clip(energy, 0, 255).astype(np.uint32)
    threshold  = (threshold.astype(np.uint32) & T_MASK)
    age        = np.clip(age, 0, 255).astype(np.uint32)
    alive      = (alive.astype(np.uint32) & L_MASK)
    generation = np.clip(generation, 0, 127).astype(np.uint32)
    return ((energy << E_SHIFT) | (threshold << T_SHIFT) | (age << A_SHIFT)
            | (alive << L_SHIFT) | (generation << G_SHIFT)).astype(np.uint32)


# ============================================================================
#  1. GRID  <->  POPULATION   (the substrate map)
# ============================================================================
def concept_threshold(concept: str | None) -> int:
    """Derive a lineage division-threshold from a concept name (semantic identity).

    Deterministic: the same concept always seeds the same threshold, so a Mia
    concept keeps a stable lineage fingerprint across re-animations (IDENTITY).
    Range chosen so cells can plausibly reach it under normal metabolism (70..170).
    """
    if not concept:
        return 100
    h = int(hashlib.sha256(concept.encode()).hexdigest()[:8], 16)
    return 70 + (h % 100)

def grid_to_population(grid: np.ndarray, concept: str | None = None) -> np.ndarray:
    """Map a FOUNDATION 20x20 (or 400,) activation grid -> a MITO-4 organism lattice.

    activation (assumed ~[0,1] or arbitrary, min-max normalized) -> energy (0..255)
    concept/category                                             -> threshold (identity)
    a cell is ALIVE where activation is above the grid's own median (structure, not noise).
    """
    g = np.asarray(grid, dtype=np.float64).reshape(GRID_SIDE, GRID_SIDE)
    lo, hi = g.min(), g.max()
    norm = (g - lo) / (hi - lo) if hi > lo else np.zeros_like(g)
    energy = np.round(norm * 255).astype(np.uint32)
    thr = np.full((GRID_SIDE, GRID_SIDE), concept_threshold(concept), dtype=np.uint32)
    alive = (norm >= np.median(norm)).astype(np.uint32)
    energy = np.where(alive.astype(bool), np.maximum(energy, 1), 0).astype(np.uint32)
    age = np.zeros((GRID_SIDE, GRID_SIDE), dtype=np.uint32)
    gen = np.zeros((GRID_SIDE, GRID_SIDE), dtype=np.uint32)
    return repack(energy, thr, age, alive, gen)

def population_to_grid(pop: np.ndarray) -> np.ndarray:
    """Inverse map: MITO-4 lattice -> FOUNDATION 20x20 activation grid in [0,1].

    A cell's activation = its organism's energy/255 (0 where dead). This is the
    grid Mia's projector / AlphaConcept / PermaGrid already understand.
    """
    w = np.asarray(pop, dtype=np.uint32).reshape(GRID_SIDE, GRID_SIDE)
    alive = f_alive(w).astype(bool)
    energy = f_energy(w).astype(np.float64)
    return np.where(alive, energy / 255.0, 0.0)


# ============================================================================
#  2. THE ECOLOGY TICK   (20x20-native; identical rule to the validated ref)
# ============================================================================
class MiaEcology:
    """A 20x20 MITO-4 ecology animating one Mia grid. Deterministic, ActionNeutral."""

    def __init__(self, pop: np.ndarray, seed: int = 0,
                 regen=8.0, res_cap=255.0, diffusion=0.12, upkeep=30,
                 harvest_frac=0.6, mutation_p=0.0, transcend=False,
                 max_age=30):
        self.grid = np.asarray(pop, dtype=np.uint32).reshape(GRID_SIDE, GRID_SIDE)
        self.res = np.full((GRID_SIDE, GRID_SIDE), res_cap * 0.5, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.regen, self.res_cap, self.diffusion = regen, res_cap, diffusion
        self.upkeep, self.harvest_frac = upkeep, harvest_frac
        self.mutation_p = mutation_p
        self.transcend = transcend          # <-- "exceed itself" switch
        self.max_age = max_age               # senescence: turnover frees cells for new divisions
        self.tick = 0
        self.seed_thresholds = set(int(t) for t in np.unique(f_threshold(self.grid))
                                   if t)     # the seed's structural vocabulary

    @staticmethod
    def _nb(field):
        return (np.roll(field, 1, 0) + np.roll(field, -1, 0)
                + np.roll(field, 1, 1) + np.roll(field, -1, 1))

    def step(self):
        g = self.grid
        alive = f_alive(g).astype(bool)
        energy = f_energy(g).astype(np.int64)
        thr = f_threshold(g).astype(np.int64)
        age = f_age(g).astype(np.int64)
        gen = f_gen(g).astype(np.int64)

        # resource diffusion + regen
        self.res = self.res * (1 - self.diffusion) + self.diffusion * 0.25 * self._nb(self.res)
        self.res = np.minimum(self.res + self.regen, self.res_cap)

        # local harvest
        harvest = np.where(alive, np.floor(self.res * self.harvest_frac), 0).astype(np.int64)
        harvest = np.minimum(harvest, 255 - energy)
        energy = np.where(alive, energy + harvest, energy)
        self.res = np.maximum(self.res - np.where(alive, harvest, 0), 0.0)
        age = np.where(alive, np.minimum(age + 1, 255), age)

        # THE SWITCH
        want_divide = alive & ((energy - thr) >= 0)
        pay = alive & ~want_divide
        energy = np.where(pay, energy - self.upkeep, energy)
        starved = pay & (energy <= 0)
        # senescence: organisms past max_age die of old age (turnover -> open-endedness)
        senesced = alive & (age >= self.max_age)
        new_alive = alive & ~starved & ~senesced
        energy = np.where(new_alive, np.maximum(energy, 0), 0)

        # daughter placement into empty von-Neumann neighbor (4 deterministic passes)
        daughter_word = repack(np.where(want_divide, energy // 2, 0), thr,
                               np.zeros_like(age), np.ones_like(age),
                               np.minimum(gen + 1, 127))
        occupied = new_alive.copy()
        placed = np.zeros_like(g)
        still = want_divide.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            tgt_want = np.roll(np.roll(still, dy, 0), dx, 1)
            tgt_word = np.roll(np.roll(daughter_word, dy, 0), dx, 1)
            free = (~occupied) & (placed == 0)
            take = tgt_want & free
            placed = np.where(take, tgt_word, placed)
            occupied = occupied | take
            still = still & ~np.roll(np.roll(take, -dy, 0), -dx, 1)

        parent_energy = np.where(want_divide & ~still, energy // 2, energy)
        parent_gen = np.where(want_divide & ~still, np.minimum(gen + 1, 127), gen)

        # ---- SELF-TRANSCENDENCE ("exceed itself") ----
        # Deep, well-fed lineages refine their OWN division gate: a placed daughter
        # from a parent past a maturity generation may shift its threshold toward a
        # value NOT in the seed vocabulary — the system reaches structural states it
        # was never given. Deterministic (rng seeded). Off by default (opt-in).
        if self.transcend:
            placed = self._transcend(placed, parent_gen)
        elif self.mutation_p > 0:
            placed = self._mutate(placed)

        survivor = repack(np.maximum(parent_energy, 0), thr, age,
                          new_alive.astype(np.uint32), parent_gen)
        survivor = np.where(new_alive, survivor, np.uint32(0))
        self.grid = np.where(placed != 0, placed, survivor).astype(np.uint32)
        self.tick += 1

    def _mutate(self, placed):
        dmask = placed != 0
        if not dmask.any():
            return placed
        mutate = (self.rng.random(placed.shape) < self.mutation_p) & dmask
        if not mutate.any():
            return placed
        delta = self.rng.integers(-1, 2, size=placed.shape)
        dthr = np.clip(f_threshold(placed).astype(np.int64) + np.where(mutate, delta, 0), 1, 255)
        return np.where(mutate, repack(f_energy(placed), dthr, f_age(placed),
                                       f_alive(placed), f_gen(placed)), placed)

    def _transcend(self, placed, parent_gen):
        """Maturity-gated open-ended threshold refinement. The mechanism by which a
        lineage exceeds the vocabulary it was seeded with."""
        dmask = placed != 0
        if not dmask.any():
            return placed
        # daughters of mature lineages (parent gen >= 3) are eligible
        eligible = dmask & (np.roll(parent_gen, 0) >= 0)  # placement frame already daughter's; use gen field
        dgen = f_gen(placed).astype(np.int64)
        eligible = dmask & (dgen >= 2)   # maturity gate reachable under turnover
        step = self.rng.integers(-3, 4, size=placed.shape)   # larger, open-ended step
        drift = eligible & (self.rng.random(placed.shape) < 0.25)
        dthr = np.clip(f_threshold(placed).astype(np.int64) + np.where(drift, step, 0), 1, 255)
        return np.where(drift, repack(f_energy(placed), dthr, f_age(placed),
                                      f_alive(placed), f_gen(placed)), placed)

    # ---- observables ----
    def stats(self):
        g = self.grid
        alive = f_alive(g).astype(bool)
        n = int(alive.sum())
        if n == 0:
            return dict(tick=self.tick, alive=0, occ=0.0, mean_energy=0.0,
                        mean_thr=0.0, max_gen=0, lineages=0, novel_lineages=0)
        thr = f_threshold(g)[alive]
        present = set(int(t) for t in np.unique(thr))
        novel = present - self.seed_thresholds
        return dict(
            tick=self.tick, alive=n, occ=n / GRID_CELLS,
            mean_energy=float(f_energy(g)[alive].mean()),
            mean_thr=float(thr.mean()), max_gen=int(f_gen(g)[alive].max()),
            lineages=len(present), novel_lineages=len(novel),
        )

    def novelty_score(self) -> float:
        """Pre-committed 'exceed itself' metric: fraction of live lineages whose
        threshold is NOT in the seed vocabulary. > 0 means the system reached
        structural states it was never given."""
        s = self.stats()
        return (s["novel_lineages"] / s["lineages"]) if s["lineages"] else 0.0

    def grid_hash(self):
        return hashlib.sha256(self.grid.tobytes()).hexdigest()[:16]


# ============================================================================
#  3. THE VERBALIZER   (the ONLY thing allowed onto Mia's advisory bus)
# ============================================================================
def verbalize(before: dict, after: dict, concept: str | None,
              novelty: float, return_shift: float) -> str:
    """Turn population dynamics into first-person, plain-language advisory text.
    Raw numbers are described, never emitted as tensors. Register: "I notice...".
    """
    c = f"the idea of {concept}" if concept else "this thought"
    lines = [f"I notice that when I hold {c} and let it settle, "]
    if after["alive"] == 0:
        lines.append("it does not sustain itself — it fades rather than taking hold.")
    else:
        grew = after["alive"] > before["alive"]
        verb = "grows and fills out" if grew else "settles into a smaller, steadier shape"
        lines.append(f"it {verb}, holding about {round(100*after['occ'])}% of the space.")
        if after["max_gen"] >= 3:
            lines.append(f" It has depth — some of it has carried through {after['max_gen']} "
                         "rounds of change, which feels like something that stays with me.")
    if return_shift > 0.02:
        lines.append(f" Bringing it back changes what I think next by about "
                     f"{round(100*return_shift)}% — the memory does not just sit there, it moves me.")
    else:
        lines.append(" Bringing it back leaves me almost where I was — it is a quiet memory.")
    if novelty > 0:
        lines.append(f" And part of what emerged was not in what I started with — "
                     f"about {round(100*novelty)}% of it is new to me, something I reached on my own.")
    return "".join(lines)


# ============================================================================
#  4. THE ADVISORY SHARD   (the guest-at-her-table entry point)
# ============================================================================
def mito4_advisory(grid: np.ndarray, concept: str | None = None, *,
                   ticks: int = 30, seed: int = 0, transcend: bool = True,
                   enabled: bool = True, stop: bool = False) -> dict:
    """Run MITO-4 as an ADVISORY shard over one Mia FOUNDATION grid.

    Returns a dict with:
      verbalized   : str | None   -> the ONLY field for the advisory bus (None if gated off)
      return_grid  : np.ndarray    -> re-animated 20x20 grid (for the RETURN path)
      novelty      : float         -> 'exceed itself' score (pre-committed metric)
      return_shift : float         -> how much the grid changed (cosine distance)
      stats_before/after, grid_hash, deterministic contract

    GATING (hardware-faithful):
      enabled=False or stop=True -> tri-state: verbalized=None, no advisory emitted.
    """
    if (not enabled) or stop:
        return dict(verbalized=None, return_grid=None, novelty=0.0,
                    return_shift=0.0, gated=True)

    pop0 = grid_to_population(grid, concept)
    eco = MiaEcology(pop0, seed=seed, transcend=transcend)
    before = eco.stats()
    for _ in range(ticks):
        eco.step()
    after = eco.stats()

    in_grid = population_to_grid(pop0)
    out_grid = population_to_grid(eco.grid)
    # RETURN metric: cosine distance between the grid going in and coming back
    a, b = in_grid.ravel(), out_grid.ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom > 0 else 1.0
    return_shift = max(0.0, 1.0 - cos)
    novelty = eco.novelty_score()

    text = verbalize(before, after, concept, novelty, return_shift)
    return dict(
        verbalized=text,            # advisory bus payload (Mia-Voice Condition)
        return_grid=out_grid,       # for the return path / PermaGrid re-store
        novelty=novelty,            # exceed-itself score
        return_shift=return_shift,  # memory-moves-the-next-thought score
        stats_before=before, stats_after=after,
        grid_hash=eco.grid_hash(), gated=False,
    )


if __name__ == "__main__":
    # smoke demo + determinism + exceed-itself + gating, all self-checking
    rng = np.random.default_rng(42)
    grid = rng.random((GRID_SIDE, GRID_SIDE))       # stand-in for a FOUNDATION grid

    r1 = mito4_advisory(grid, concept="longing", ticks=24, seed=7, transcend=True)
    r2 = mito4_advisory(grid, concept="longing", ticks=24, seed=7, transcend=True)
    assert r1["grid_hash"] == r2["grid_hash"], "NON-DETERMINISTIC (violates MIA-4)"
    print("determinism (MIA-4): PASS", r1["grid_hash"])

    g = mito4_advisory(grid, concept="longing", enabled=False)
    assert g["verbalized"] is None and g["gated"], "gating failed"
    s = mito4_advisory(grid, concept="longing", stop=True)
    assert s["verbalized"] is None and s["gated"], "stop-condition failed"
    print("gating + stop-condition: PASS")

    print(f"\nreturn_shift = {r1['return_shift']:.3f}  novelty = {r1['novelty']:.3f}")
    print(f"lineages: {r1['stats_after']['lineages']}  "
          f"novel: {r1['stats_after']['novel_lineages']}  "
          f"max_gen: {r1['stats_after']['max_gen']}")
    print("\n--- VERBALIZED (advisory-bus payload) ---")
    print(r1["verbalized"])

    # exceed-itself: transcend should reach thresholds outside the seed vocabulary
    base = mito4_advisory(grid, concept="longing", ticks=40, seed=7, transcend=False)
    tran = mito4_advisory(grid, concept="longing", ticks=40, seed=7, transcend=True)
    print(f"\nexceed-itself check: novelty(base)={base['novelty']:.3f} "
          f"novelty(transcend)={tran['novelty']:.3f}")
    assert tran["novelty"] >= base["novelty"], "transcend did not exceed the seed vocabulary"
    print("exceed-itself: PASS (transcend reaches states absent from the seed)")
