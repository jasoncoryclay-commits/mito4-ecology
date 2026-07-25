#!/usr/bin/env python3
"""
MITO-4 — Minimal Integer Threshold Organism (4-byte).
A single 32-bit little-endian integer encodes a "cell": a bit-packed genome+state.
A single linear-threshold switch decides each tick whether the cell divides (mitosis),
persists, or dies (starvation). Stdlib-only. Runs as a flat array of uint32.

Author: reference implementation for the MITO-4 R&D paper.
"""
import struct
from dataclasses import dataclass

# ---- Bit-field layout (little-endian uint32) -------------------------------
# bits  0-7   energy      (0..255)  current resource level        E
# bits  8-15  threshold   (0..255)  fixed lineage division gate   T
# bits 16-23  age         (0..255)  ticks lived                   A
# bit    24   alive       (0/1)     alive flag                    L
# bits 25-31  generation  (0..127)  lineage depth                 G
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
    return v & 0xFFFFFFFF

def unpack(v):
    return {
        "energy":     (v >> E_SHIFT) & E_MASK,
        "threshold":  (v >> T_SHIFT) & T_MASK,
        "age":        (v >> A_SHIFT) & A_MASK,
        "alive":      (v >> L_SHIFT) & L_MASK,
        "generation": (v >> G_SHIFT) & G_MASK,
    }

# ---- The linear-threshold switch (the single decision rule) ----------------
# decision = 1 (divide) iff (energy - threshold) >= 0, else 0.
# This is a perceptron activation / neuron firing threshold / cell-size checkpoint.
def switch(energy, threshold):
    return 1 if (energy - threshold) >= 0 else 0

# ---- Tick rule (pure function: cell -> list[cell]) -------------------------
METABOLISM = 40   # energy gained per tick from environment
UPKEEP     = 30   # energy paid per tick if not dividing

def step_cell(v, metabolism=METABOLISM, upkeep=UPKEEP):
    """Advance one cell one tick. Returns a list: [] (died), [c] (persist), or [c1,c2] (divided)."""
    s = unpack(v)
    if not s["alive"]:
        return []                      # already dead: recycled out
    energy = min(255, s["energy"] + metabolism)
    age    = min(255, s["age"] + 1)
    if switch(energy, s["threshold"]):
        # mitosis: split energy in half, same threshold (heredity), gen+1
        child_e = energy // 2
        g = min(127, s["generation"] + 1)
        c = pack(child_e, s["threshold"], 0, 1, g)
        return [c, c]                  # two daughters, identical genome
    else:
        energy -= upkeep
        if energy <= 0:
            return [pack(0, s["threshold"], age, 0, s["generation"])]  # starved -> dead
        return [pack(energy, s["threshold"], age, 1, s["generation"])]

def step_population(pop, metabolism=METABOLISM, upkeep=UPKEEP):
    nxt = []
    for v in pop:
        nxt.extend(step_cell(v, metabolism, upkeep))
    # cull dead (structure dissolves; contents recycled)
    return [v for v in nxt if unpack(v)["alive"]]

# ---- Verification ----------------------------------------------------------
if __name__ == "__main__":
    # 1) round-trip a 4-byte pack/unpack through actual little-endian bytes
    seed = pack(energy=11, threshold=90, age=0, alive=1, generation=0)
    raw  = struct.pack("<I", seed)             # little-endian, GPU/CPU memory order
    back = struct.unpack("<I", raw)[0]
    assert back == seed, "round-trip failed"
    print(f"seed uint32 = 0x{seed:08x}  raw bytes = {raw.hex()}  decodes -> {unpack(back)}")
    assert len(raw) == 4, "not 4 bytes"

    # 2) simulate exponential growth from one under-threshold cell
    pop = [seed]
    print("\ntick | alive | example cell")
    for t in range(9):
        s0 = unpack(pop[0]) if pop else {}
        print(f"{t:>4} | {len(pop):>5} | {s0}")
        pop = step_population(pop)

    # 3) starvation demo: hostile environment where upkeep > metabolism and
    #    threshold is never reachable -> the cell drains and dies.
    print("\nstarvation lineage (hostile env: metabolism=10 < upkeep=30, threshold=200 unreachable):")
    dead_seed = pack(energy=60, threshold=200, age=0, alive=1, generation=0)
    p = [dead_seed]
    for t in range(6):
        alive = len(p)
        e = unpack(p[0])["energy"] if p else 0
        print(f"  tick {t}: alive={alive} energy={e}")
        p = step_population(p, metabolism=10, upkeep=30)
    print(f"  final: {'extinct' if not p else 'survived'}")
