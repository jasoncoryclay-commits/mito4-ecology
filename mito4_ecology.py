#!/usr/bin/env python3
"""
MITO-4 ECOLOGY — grid-resident spatial variant of the Minimal Integer Threshold Organism.

This is the CPU reference (NumPy-vectorized) whose physics MUST be validated and
deterministic BEFORE the CUDA/H100 version runs. It is intentionally written so that
every operation maps 1:1 onto a per-grid-cell CUDA thread (see mito4_kernel.cu).

Design choice that makes it GPU-native (fixed-size, no dynamic allocation):
  - The population lives ON a lattice: one organism SLOT per grid cell.
  - A slot is either empty (dead, uint32 with alive bit = 0) or holds one organism.
  - Metabolism is LOCAL: each cell draws energy from a resource field that
    regenerates and diffuses. Crowding depletes local resource -> starvation.
  - Division (mitosis) places a daughter into an empty von-Neumann neighbor.
    If no neighbor is free, division is blocked this tick (crowding pressure).
    This keeps the state a FIXED HxW array of uint32 -> perfect for a CUDA grid.

Same 4-byte little-endian word layout as base MITO-4:
  bits  0-7 energy | 8-15 threshold | 16-23 age | 24 alive | 25-31 generation

Stdlib + numpy only. Deterministic given a seed.
"""
import argparse
import hashlib
import numpy as np

# ---- bit layout ------------------------------------------------------------
E_SHIFT, E_MASK = 0, 0xFF
T_SHIFT, T_MASK = 8, 0xFF
A_SHIFT, A_MASK = 16, 0xFF
L_SHIFT, L_MASK = 24, 0x1
G_SHIFT, G_MASK = 25, 0x7F

def pack(energy, threshold, age, alive, generation):
    v  = (energy      & E_MASK) << E_SHIFT
    v |= (threshold   & T_MASK) << T_SHIFT
    v |= (age         & A_MASK) << A_SHIFT
    v |= (alive       & L_MASK) << L_SHIFT
    v |= (generation  & G_MASK) << G_SHIFT
    return np.uint32(v)

# vectorized field extractors (operate on whole uint32 arrays)
def f_energy(w):     return (w >> E_SHIFT) & E_MASK
def f_threshold(w):  return (w >> T_SHIFT) & T_MASK
def f_age(w):        return (w >> A_SHIFT) & A_MASK
def f_alive(w):      return (w >> L_SHIFT) & L_MASK
def f_gen(w):        return (w >> G_SHIFT) & G_MASK

def repack(energy, threshold, age, alive, generation):
    """Vectorized pack. All args are uint32 arrays of equal shape."""
    energy    = np.clip(energy, 0, 255).astype(np.uint32)
    threshold = (threshold & T_MASK).astype(np.uint32)
    age       = np.clip(age, 0, 255).astype(np.uint32)
    alive     = (alive & L_MASK).astype(np.uint32)
    generation= np.clip(generation, 0, 127).astype(np.uint32)
    return ((energy << E_SHIFT) | (threshold << T_SHIFT) | (age << A_SHIFT)
            | (alive << L_SHIFT) | (generation << G_SHIFT)).astype(np.uint32)


class Ecology:
    """A HxW lattice of MITO-4 organisms coupled to a diffusing resource field."""

    def __init__(self, H, W, seed=0,
                 regen=8.0,          # resource regenerated per cell per tick
                 res_cap=255.0,      # max resource a cell can hold
                 diffusion=0.12,     # fraction of resource that diffuses to neighbors
                 upkeep=30,          # energy paid per tick when not dividing
                 harvest_frac=0.6,   # fraction of local resource an organism can take
                 mutation_p=0.02,    # prob a daughter's threshold mutates +/-1
                 init_density=0.02): # fraction of cells seeded with an organism
        self.H, self.W = H, W
        self.rng = np.random.default_rng(seed)
        self.regen = np.float32(regen)
        self.res_cap = np.float32(res_cap)
        self.diffusion = np.float32(diffusion)
        self.upkeep = np.uint32(upkeep)
        self.harvest_frac = np.float32(harvest_frac)
        self.mutation_p = mutation_p
        self.tick = 0

        # organism lattice (uint32). empty = alive bit 0.
        self.grid = np.zeros((H, W), dtype=np.uint32)
        # resource field (float32), start full-ish
        self.res = np.full((H, W), res_cap * 0.5, dtype=np.float32)

        # seed organisms
        n = int(H * W * init_density)
        ys = self.rng.integers(0, H, n)
        xs = self.rng.integers(0, W, n)
        e0 = self.rng.integers(20, 60, n).astype(np.uint32)
        t0 = self.rng.integers(70, 130, n).astype(np.uint32)  # varied lineages
        for i in range(n):
            self.grid[ys[i], xs[i]] = pack(int(e0[i]), int(t0[i]), 0, 1, 0)

    # ---- helpers (all map to a CUDA stencil) -------------------------------
    @staticmethod
    def _neighbor_sum(field):
        """4-neighbor (von Neumann) sum with wrap-around (torus). Pure stencil."""
        return (np.roll(field, 1, 0) + np.roll(field, -1, 0)
                + np.roll(field, 1, 1) + np.roll(field, -1, 1))

    def step(self):
        g = self.grid
        alive = f_alive(g).astype(bool)
        energy = f_energy(g).astype(np.int32)
        thr    = f_threshold(g).astype(np.int32)
        age    = f_age(g).astype(np.int32)
        gen    = f_gen(g).astype(np.int32)

        # 1) resource diffusion + regen (stencil, then clip)
        nb = self._neighbor_sum(self.res)
        self.res = self.res * (1 - self.diffusion) + self.diffusion * 0.25 * nb
        self.res = np.minimum(self.res + self.regen, self.res_cap)

        # 2) local harvest: living cells draw energy from their resource cell
        harvest = np.where(alive, np.floor(self.res * self.harvest_frac), 0).astype(np.int32)
        harvest = np.minimum(harvest, 255 - energy)  # cap at energy field max
        energy = np.where(alive, energy + harvest, energy)
        self.res = np.where(alive, self.res - harvest, self.res)  # deplete local resource
        self.res = np.maximum(self.res, 0.0)

        # 3) age the living
        age = np.where(alive, np.minimum(age + 1, 255), age)

        # 4) THE SWITCH: divide iff (energy - threshold) >= 0
        want_divide = alive & ((energy - thr) >= 0)
        # non-dividing living cells pay upkeep; may starve
        pay = alive & ~want_divide
        energy = np.where(pay, energy - self.upkeep.astype(np.int32), energy)
        starved = pay & (energy <= 0)
        # apply death
        new_alive = alive & ~starved
        energy = np.where(new_alive, np.maximum(energy, 0), 0)

        # 5) MITOSIS with placement into an empty von-Neumann neighbor.
        #    To stay deterministic & race-free (as the CUDA scatter will),
        #    we resolve daughter placement with a fixed neighbor priority and
        #    a per-tick arbitration: each empty cell accepts at most one daughter,
        #    chosen by lowest linear index of a requesting parent.
        parent_e_after = np.where(want_divide, energy // 2, energy)  # parent keeps half
        # build daughter "requests" into 4 directions
        daughter_word = repack(np.where(want_divide, energy // 2, 0),
                               thr, np.zeros_like(age),
                               np.ones_like(age), np.minimum(gen + 1, 127))

        occupied = new_alive.copy()          # cells that will be occupied by survivors
        placed_grid = np.zeros_like(g)       # daughters land here

        # deterministic directional passes (N,S,W,E). Each pass: a parent tries to
        # place a daughter into that neighbor if it's empty AND not yet taken.
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        still_wants = want_divide.copy()
        for (dy, dx) in dirs:
            # candidate target cell for each parent
            tgt_want = np.roll(np.roll(still_wants, dy, 0), dx, 1)  # parents shifted into target frame
            tgt_word = np.roll(np.roll(daughter_word, dy, 0), dx, 1)
            free = (~occupied) & (placed_grid == 0)
            take = tgt_want & free
            placed_grid = np.where(take, tgt_word, placed_grid)
            occupied = occupied | take
            # mark those parents as satisfied (shift the "take" back to parent frame)
            satisfied = np.roll(np.roll(take, -dy, 0), -dx, 1)
            still_wants = still_wants & ~satisfied

        # parents that divided: keep half energy, gen+1, age reset to 0 (fresh daughter-in-place)
        divided = want_divide & ~still_wants   # actually placed at least... (parent always persists)
        # NOTE: a parent that "wanted" but found NO free neighbor still persists,
        # simply keeping its (halved? no) energy. Blocked division => parent keeps FULL energy,
        # does not halve, does not advance generation (division blocked by crowding).
        parent_energy = np.where(want_divide & ~still_wants, energy // 2, energy)
        parent_gen    = np.where(want_divide & ~still_wants, np.minimum(gen + 1, 127), gen)

        # 6) optional mutation on placed daughters' threshold
        if self.mutation_p > 0:
            dmask = placed_grid != 0
            if dmask.any():
                delta = self.rng.integers(-1, 2, size=placed_grid.shape)  # -1,0,+1
                mutate = (self.rng.random(placed_grid.shape) < self.mutation_p) & dmask
                if mutate.any():
                    dthr = f_threshold(placed_grid).astype(np.int32)
                    dthr = np.clip(dthr + np.where(mutate, delta, 0), 1, 255)
                    placed_grid = np.where(
                        mutate,
                        repack(f_energy(placed_grid), dthr, f_age(placed_grid),
                               f_alive(placed_grid), f_gen(placed_grid)),
                        placed_grid,
                    )

        # 7) assemble next grid
        survivor_word = repack(np.maximum(parent_energy, 0), thr, age, new_alive.astype(np.uint32), parent_gen)
        survivor_word = np.where(new_alive, survivor_word, np.uint32(0))
        # daughters land only where a survivor is NOT already (guaranteed by 'free' logic)
        next_grid = np.where(placed_grid != 0, placed_grid, survivor_word)
        self.grid = next_grid.astype(np.uint32)
        self.tick += 1

    # ---- observables -------------------------------------------------------
    def stats(self):
        g = self.grid
        alive = f_alive(g).astype(bool)
        n = int(alive.sum())
        if n == 0:
            return dict(tick=self.tick, alive=0, mean_energy=0, mean_thr=0,
                        max_gen=0, lineage_diversity=0, res_mean=float(self.res.mean()))
        thr = f_threshold(g)[alive]
        # lineage diversity = unique thresholds present (proxy for surviving lineages)
        diversity = int(np.unique(thr).size)
        return dict(
            tick=self.tick,
            alive=n,
            mean_energy=float(f_energy(g)[alive].mean()),
            mean_thr=float(thr.mean()),
            max_gen=int(f_gen(g)[alive].max()),
            lineage_diversity=diversity,
            res_mean=float(self.res.mean()),
        )

    def grid_hash(self):
        return hashlib.sha256(self.grid.tobytes()).hexdigest()[:16]


def run(H, W, ticks, seed, mutation_p, log_every, quiet=False):
    eco = Ecology(H, W, seed=seed, mutation_p=mutation_p)
    history = []
    for t in range(ticks):
        s = eco.stats()
        history.append(s)
        if not quiet and (t % log_every == 0 or t == ticks - 1):
            print(f"t={s['tick']:>4} alive={s['alive']:>8} "
                  f"E={s['mean_energy']:6.1f} T={s['mean_thr']:6.1f} "
                  f"maxgen={s['max_gen']:>3} lineages={s['lineage_diversity']:>3} "
                  f"res={s['res_mean']:6.1f}")
        if s['alive'] == 0:
            if not quiet:
                print(f"  -> extinction at tick {s['tick']}")
            break
        eco.step()
    return eco, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mutation", type=float, default=0.02)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--verify", action="store_true", help="run determinism check")
    args = ap.parse_args()

    if args.verify:
        # determinism: same seed -> identical grid hash after N ticks
        e1, _ = run(64, 64, 50, seed=7, mutation_p=0.0, log_every=999, quiet=True)
        e2, _ = run(64, 64, 50, seed=7, mutation_p=0.0, log_every=999, quiet=True)
        h1, h2 = e1.grid_hash(), e2.grid_hash()
        print(f"determinism check (mutation off): {h1} == {h2} -> {h1 == h2}")
        # mutation on, fixed seed, still reproducible (rng is seeded)
        e3, _ = run(64, 64, 50, seed=7, mutation_p=0.05, log_every=999, quiet=True)
        e4, _ = run(64, 64, 50, seed=7, mutation_p=0.05, log_every=999, quiet=True)
        h3, h4 = e3.grid_hash(), e4.grid_hash()
        print(f"determinism check (mutation on):  {h3} == {h4} -> {h3 == h4}")
        assert h1 == h2 and h3 == h4, "NON-DETERMINISTIC — fix before GPU port"
        print("OK: reference is deterministic.")
    else:
        run(args.H, args.W, args.ticks, args.seed, args.mutation, args.log_every)
