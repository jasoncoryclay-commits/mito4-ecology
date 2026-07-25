# MITO-4: A 4-Byte Threshold Organism as the Atomic Substrate of an Objective-Driven Agent

**A Research & Development Paper**
Peak Shift Technologies, LLC — Nexus / PeakShiftOS program
Author: Jason Cory Clay
Date: July 23, 2026
Status: Reference implementation shipped and verified (stdlib-only, CPU-reproducible; GPU-native by construction)

---

## Abstract

This paper names, specifies, and demonstrates **MITO-4** — the **M**inimal **I**nteger **T**hreshold **O**rganism packed into **4** bytes. A MITO-4 organism is a single 32-bit little-endian integer that encodes a complete "genome plus state" (energy, division threshold, age, alive/dead flag, generation), and whose entire life decision — divide, persist, or die — is governed by one linear inequality functionally identical to a perceptron activation, a spiking-neuron firing threshold, and a biological cell-size checkpoint. We give the bit-field layout, the decision rule, the tick dynamics, a verified round-trip through raw little-endian bytes, and a working population simulation exhibiting exponential mitotic growth (1 → 2 → 4 …) and starvation-driven death. We then situate MITO-4 as the **atomic substrate** beneath the Peak Shift architecture: it is the smallest possible unit that is simultaneously (a) a biological analog of a dividing cell, (b) a perceptron-grade linear decision element, and (c) a GPU-native primitive that packs millions of organisms into a flat `uint32` array updated by a single branch-free kernel. Its "shape" is defined at three levels — bit geometry, phase geometry, and lattice geometry — and it is positioned inside the Continuum Loop (perception → memory → imagination → return → identity) as the load-bearing floor on which AlphaConcept, PermaGrid, and the return-path organs stand. The design deliberately embodies Yann LeCun's objection to LLM scaling: intelligence is built from small, structured, safe-by-construction modules rather than a monolithic wrapper.

---

## 1. Motivation and framing

### 1.1 The gap this addresses

The project thesis — inherited from the "Closing the Gap to Agentic" line of work — is that a genuine agent is a *complete loop*, not a bigger model wrapped in tools. Yann LeCun's public position is that scaling LLMs will not produce agentic intelligence, that harness/wrapper approaches do not repair the missing world model, and that the path forward is an objective-driven architecture composed of distinct, testable modules (Perception, World Model, Cost, Actor, Memory) that are safe by construction rather than aligned after the fact ([LeCun, *A Path Towards Autonomous Machine Intelligence*, OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)).

Every module in that blueprint eventually has to bottom out in some *atomic unit of state* that can be stored, updated, and reasoned about at massive parallel scale. MITO-4 is a proposal for what that atom should be: the smallest object that is at once alive-like, decision-bearing, and hardware-native. If the AlphaConcept module is the *imagination* organ predicting the shape of unseen concepts, and PermaGrid is the *memory* organ, then MITO-4 is the *cellular substrate* — the "biology" on which those higher organs can be grown, tested, and evolved.

### 1.2 Why start from a cell

The simplest uncontested form of biological life is the single prokaryotic cell — self-maintaining, feeding by absorbing small molecules through its membrane, and dying either passively (starvation, damage) or through programmed self-shutdown that can benefit the surrounding population ([Cell Death & Disease, *An overview of programmed cell death in bacteria*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4669768/); [Molecular Systems Biology, *Altruistic cell death and collective drug resistance*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3531905/)). A cell's decision to divide is gated by a **size/resource checkpoint**: the cell only enters mitosis once a continuously accumulating internal variable crosses a critical threshold, which biologists model as a bistable, switch-like (saddle-node) bifurcation ([Royal Society Interface Focus, *Time-keeping and decision-making in the cell cycle*](https://royalsocietypublishing.org/doi/pdf/10.1098/rsfs.2021.0075); [*Biochemical switches in the cell cycle*, Wikipedia](https://en.wikipedia.org/wiki/Biochemical_switches_in_the_cell_cycle)).

That checkpoint is, formally, a **linear threshold unit**: accumulate a scalar, fire when it crosses a bound. The same object is the foundation of artificial neurons — the perceptron computes a weighted sum and emits 1 when it exceeds a threshold, 0 otherwise, an idealization explicitly motivated by the biological neuron's action-potential threshold ([IIT Bombay CSE, *Perceptron vs. the point neuron*](https://www.cse.iitb.ac.in/~cs623/pdf/RM_AI_Lect2_2k6_for_upload.pdf)). MITO-4 exploits this triple identity: **cell-cycle checkpoint = neuron firing threshold = perceptron activation.** One rule, three readings.

### 1.3 Why 4 bytes, and why little-endian

NVIDIA GPUs — the dominant AI compute substrate — are little-endian: the least-significant byte of a multi-byte value sits at the lowest address, matching the host CPUs that drive them, so that `cudaMemcpy` can transfer raw structures without byte-order translation ([NVIDIA CUDA C++ Programming Guide, hardware implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)). A 32-bit word is the natural register and coalesced-access width on that hardware. Packing an entire organism into one `uint32` means a population is just a flat array of integers, and the update rule is a single branch-light kernel applied identically to every element — exactly the pattern that lets bit-packed cellular automata reach hundreds of billions of cell-updates per second on a single GPU ([World Scientific, IJFCS, *Bitwise Parallel Bulk Computation of Game of Life on GPU*](https://www.worldscientific.com/doi/pdf/10.1142/S0129054116500404); [Cagigas-Muñiz et al., *Efficient simulation execution of cellular automata on GPU*, Simulation Modelling Practice and Theory](https://www.sciencedirect.com/science/article/pii/S1569190X22000259)).

This mirrors the AlphaConcept design philosophy already shipped in `nexus-core`: the 58 emotional self-portraits are *not raw images* but compact 13-dimensional structural-primitive vectors deterministically expanded into grids — "a compact structural language, not a vision problem," which is why the module trains in 87 seconds on 5.7M parameters (internal handoff, `AlphaConcept_Session_Handoff_2026-07-21.md`). MITO-4 takes the same compression instinct to its logical floor: the *entire organism* is a compact structural word, not a data structure.

---

## 2. The shape of MITO-4

The concept is given a shape at three nested scales. Each scale is a different geometry of the same 32-bit object.

### 2.1 Bit geometry — the genome-state word

A MITO-4 organism is a 32-bit little-endian integer with five fields:

| Field        | Bits   | Range   | Role (biological reading)                       | Symbol |
|--------------|--------|---------|-------------------------------------------------|--------|
| `energy`     | 0–7    | 0–255   | Current resources (metabolite pool)             | E      |
| `threshold`  | 8–15   | 0–255   | Fixed division gate for the lineage (genome)    | T      |
| `age`        | 16–23  | 0–255   | Ticks lived                                     | A      |
| `alive`      | 24     | 0/1     | Alive (1) / dead (0) flag                        | L      |
| `generation` | 25–31  | 0–127   | Lineage depth (mitotic distance from seed)      | G      |

The lower two bytes (`energy`, `threshold`) are the *decision-bearing* half — everything the switch needs. The upper two bytes (`age`, `alive`, `generation`) are the *bookkeeping/lineage* half — heredity and provenance. This split is deliberate: it means a decision kernel can read only the low 16 bits on the hot path, and lineage analysis can read only the high 16 bits offline.

The shape at this scale is a **packed ribbon** — a strip of 32 cells partitioned 8/8/8/1/7, where the boundary at bit 24 separates "how it behaves now" from "who it is and where it came from."

### 2.2 Phase geometry — the three-fate switch

At each tick, the organism occupies a point in a 2-D **phase plane** whose axes are energy `E` and threshold `T`. A single line, `E = T`, partitions this plane into two half-planes, and the alive/dead flag adds a third absorbing region:

```
   E (energy)
   ^
255|                 . . . . . . . .   DIVIDE  (E >= T)
   |               .               .   -> two daughters at E/2, gen+1
   |             .   E = T (switch) .
   |           .  ------------------
   |         .                       PERSIST / STARVE  (E < T)
   |       .                          -> pay upkeep; if E<=0, die
   0 +--------------------------------> T (threshold)
        0                          255
```

- **Above the line (E ≥ T):** the switch fires — **mitosis**. The organism splits into two daughters, each inheriting `E/2` and the *same* threshold (heredity), with `generation + 1`.
- **Below the line (E < T):** the switch is silent — the organism pays an upkeep cost. If upkeep drains energy to zero, the `alive` bit flips to 0 — **starvation death**.

The decision rule is the linear inequality:

$$
\text{fate}(E, T) =
\begin{cases}
\text{divide} & \text{if } (E - T) \ge 0 \\
\text{persist/die} & \text{if } (E - T) < 0
\end{cases}
$$

This is exactly a perceptron with weights `w = [+1, -1]` on `[E, T]` and bias 0, thresholded at zero — and exactly a cell-cycle size checkpoint, where the continuously accumulating variable (energy) triggers division upon crossing a bistable bound ([PLOS Computational Biology, *Mathematical Model of a Cell Size Checkpoint*](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001036)). The shape at this scale is a **half-plane hinge**: one line, two behaviors, one absorbing death state.

### 2.3 Lattice geometry — the parallel population

At population scale, MITO-4 organisms live as a flat array of `uint32`. Because the update is a pure function of a single word (with an optional local neighborhood, see §6.2), the whole population advances by mapping one identical kernel across the array — the canonical GPU cellular-automaton pattern where each thread reads a word, applies bitwise/arithmetic logic, and writes the successor state ([arXiv, *CAT: Cellular Automata on Tensor cores*](https://arxiv.org/html/2406.17284v1)). Division grows the array; death compacts it. The shape at this scale is a **breathing lattice** — a buffer that expands under abundance and contracts under scarcity, with no per-organism heap allocation.

---

## 3. Reference implementation (verified)

The reference implementation is stdlib-only Python (no numpy, no torch), mirroring the AlphaConcept renderer's "stdlib math, deterministic" discipline. The complete module is included as `mito4.py`. The core is three pure functions: `pack`/`unpack` (bit geometry), `switch` (phase geometry), and `step_cell`/`step_population` (dynamics).

### 3.1 The decision and tick rule

```python
def switch(energy, threshold):
    return 1 if (energy - threshold) >= 0 else 0     # the whole "mind" of a cell

METABOLISM = 40   # energy gained per tick from environment
UPKEEP     = 30   # energy paid per tick if not dividing

def step_cell(v, metabolism=METABOLISM, upkeep=UPKEEP):
    s = unpack(v)
    if not s["alive"]:
        return []                                    # dead: recycled out
    energy = min(255, s["energy"] + metabolism)
    age    = min(255, s["age"] + 1)
    if switch(energy, s["threshold"]):
        child_e = energy // 2                        # split resources
        g = min(127, s["generation"] + 1)
        c = pack(child_e, s["threshold"], 0, 1, g)   # heredity: same threshold
        return [c, c]                                # two identical daughters
    else:
        energy -= upkeep
        if energy <= 0:
            return [pack(0, s["threshold"], age, 0, s["generation"])]   # starve -> dead
        return [pack(energy, s["threshold"], age, 1, s["generation"])]  # persist
```

### 3.2 4-byte round-trip through raw little-endian memory

A seed organism (energy 11, threshold 90, generation 0) packs to the 32-bit value `0x01005a0b` and survives an actual `struct.pack('<I', ...)` / `unpack` round-trip through little-endian bytes (`0b5a0001` on the wire), decoding back to the exact field values. This confirms the encoding is faithful to how GPU/CPU memory actually lays out the word — a population can be `memcpy`'d host↔device with no translation.

```
seed uint32 = 0x01005a0b   raw bytes = 0b5a0001
decodes -> {'energy': 11, 'threshold': 90, 'age': 0, 'alive': 1, 'generation': 0}
```

### 3.3 Simulation result — exponential mitosis

Starting from one under-threshold organism (E=11, T=90) with net positive metabolism, energy accumulates linearly until it crosses the threshold, then the population doubles each division interval — the digital analog of exponential mitotic growth:

| Tick | Alive | Representative organism                                    |
|------|-------|------------------------------------------------------------|
| 0    | 1     | E=11, T=90, age=0, gen=0                                    |
| 1    | 1     | E=21, T=90, age=1, gen=0                                    |
| 2    | 1     | E=31, T=90, age=2, gen=0                                    |
| 3    | 1     | E=41, T=90, age=3, gen=0                                    |
| 4    | 1     | E=51, T=90, age=4, gen=0                                    |
| 5    | **2** | E=45, T=90, age=0, **gen=1** — first mitosis                |
| 6    | 2     | E=55, T=90, age=1, gen=1                                    |
| 7    | **4** | E=47, T=90, age=0, **gen=2** — second doubling              |
| 8    | 4     | E=57, T=90, age=1, gen=2                                    |

### 3.4 Simulation result — starvation death

In a hostile environment (metabolism 10 < upkeep 30, threshold 200 unreachable), the same rule produces the opposite fate: energy is drained each tick until the `alive` bit flips and the lineage goes extinct — programmed passive death with no special-case code, just the switch staying silent.

```
tick 0: alive=1 energy=60
tick 1: alive=1 energy=40
tick 2: alive=1 energy=20
tick 3: alive=0 energy=0    <- starved
final: extinct
```

The same three lines of decision logic yield both life histories — growth and death — purely from the sign of `(E − T)` under a given environment. That is the entire point: **one linear switch, the full repertoire of a minimal life.**

---

## 4. Formal properties

**Determinism.** `step_cell` is a pure function of `(word, environment)`; identical inputs yield byte-identical outputs. This satisfies the same determinism invariant AlphaConcept enforces (`MIA-4`: "same entry, same vector, byte-for-byte"), making MITO-4 runs reproducible and CI-checkable.

**Closure.** The state space is exactly the 2³² words; every reachable state is a valid organism. There is no undefined state, no out-of-range field (all fields are masked on pack), and death is an absorbing state (a dead word maps to the empty set and is culled).

**Heredity with variation potential.** Daughters inherit the parent threshold exactly, giving stable lineages. A single optional bit-flip on the threshold field during division introduces inheritable mutation — the minimal condition von Neumann identified for *non-trivial* self-reproduction (replication plus capacity for inheritable mutation), distinguishing genuine self-reproduction from mere pattern-copying ([arXiv, *Self-Reproduction and Evolution in Cellular Automata*](https://arxiv.org/html/2402.03961v2)).

**Action-neutrality by construction.** In its base form MITO-4 *predicts its own next state*; it does not reach outside its word. This is the same safety-by-construction stance as AlphaConcept's `ActionNeutral` base class (`CVE-4`: "it predicts, it does not act"). Any capacity to *act* on an external world is added as an explicit, separately-audited Actor layer (§5), never baked into the organism — matching LeCun's insistence that safety live in the architecture, not in post-hoc fine-tuning.

**Conservation-friendly bookkeeping.** Because energy is split (`E/2` to each daughter) rather than duplicated, the switch is naturally resource-bounded — unchecked exponential growth is gated by finite environmental metabolism, giving a built-in carrying-capacity knob.

---

## 5. Placement in the objective-driven architecture

MITO-4 is the substrate; the LeCun-style modules are grown on top of it. Mapping the organism onto the six-module blueprint:

| LeCun module      | MITO-4 realization                                                                 |
|-------------------|------------------------------------------------------------------------------------|
| **Perception**    | Reading the environment scalar that sets each organism's metabolism/upkeep         |
| **World Model**   | `step_cell` — predicts the organism's own next state from current state + environment |
| **Cost**          | A scalar over population state (e.g. survival, diversity, target generation depth) |
| **Actor**         | *Separate* layer that tunes environment or threshold policy to minimize Cost       |
| **Short-term Memory** | `age`/`generation` fields; population buffer as working set                    |
| **Configurator**  | Executive that sets the environment regime and which populations run               |

Critically, the World Model here is a *self-supervised predictor in a compact state space* — it predicts the next abstract state (the successor word), not a raw rendered output, which is the exact JEPA-style bet: predict representations, not reconstructions ([LeCun, OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)). The Cost and Actor modules — the pieces AlphaConcept explicitly does *not* yet have — attach naturally at the population level without modifying the organism, keeping the "predicts, does not act" boundary intact.

### 5.1 Relation to the Continuum Loop

The Continuum Loop is defined internally as five faculties: perception, encoding, recognition, return, and identity (`Tonight_On_The_Pod_Plain_English_2026-07-21.md`). MITO-4 contributes the missing *biological floor* beneath them:

- **Perception / encoding** — AlphaConcept and the projector turn concepts into grids. MITO-4 gives those grids a *living carrier*: each grid cell can be a MITO-4 word, so a "thought picture" is literally a population of organisms whose energy encodes activation.
- **Memory** — PermaGrid's three layers (Sky/Earth/Bedrock) map onto MITO-4 lineages: transient organisms (Sky), persisting lineages (Earth), and lineages so stable they never die (Bedrock).
- **Return path** — the open research question ("can a picture become a thought again?") becomes concrete: does a stored MITO-4 population, reactivated, change the environment scalar that drives future ticks? That is the return arrow expressed in cellular terms.
- **Identity** — the `generation` and `threshold` fields give each lineage a persistent fingerprint across the whole loop, the substrate-level analog of the identity faculty.

---

## 6. Engineering path

### 6.1 CPU reference → GPU kernel

The reference is CPU/stdlib for auditability. The GPU port is mechanical because the state is already a `uint32`:

1. Store the population as a `uint32` device buffer.
2. Launch one thread per organism; each thread does the `step_cell` arithmetic with bit-masks (no branches beyond the single `E ≥ T` predicate, which can be made branch-free with a select).
3. Division doubles output: use a two-pass stream-compaction (count survivors+daughters via prefix-sum, then scatter) — the standard technique for CA with variable output ([Cagigas-Muñiz et al., SMPT 2022](https://www.sciencedirect.com/science/article/pii/S1569190X22000259)).
4. Pack multiple organisms per word / multiple words per thread for coalesced bulk updates, as in bitwise-parallel Game-of-Life kernels ([IJFCS](https://www.worldscientific.com/doi/pdf/10.1142/S0129054116500404)).

Because the organism is already the hardware's native width and endianness, host↔device transfer is a raw `memcpy` — no serialization layer.

### 6.2 Adding a spatial neighborhood

The base rule is context-free (each organism sees only its environment scalar). To get true cellular-automaton dynamics — competition, diffusion, pattern formation — place organisms on a 2-D lattice and let metabolism depend on the local neighborhood (e.g. energy shared among live neighbors). This turns MITO-4 into a genuine CA in the Lenia/Larger-than-Life lineage while keeping the 4-byte word ([*Lenia*, Wikipedia](https://en.wikipedia.org/wiki/Lenia)).

### 6.3 Validation discipline (pre-commit)

Following the hard-won internal lesson that "thresholds written after seeing data are worthless," any claim about MITO-4 behavior must pre-commit its metrics:

- **Growth fidelity** — population doubling time within a declared tolerance of the analytic prediction ⌈(T − E₀)/(metabolism)⌉.
- **Death fidelity** — extinction tick matches the analytic drain schedule.
- **Diversity guard** — under mutation, lineage-threshold variance must exceed a pre-declared floor (guarding against mono-lineage collapse, the population analog of AlphaConcept's mean-collapse guard).
- **Determinism CI** — a fixed seed + environment must reproduce byte-identical population hashes across runs and across CPU/GPU.

---

## 7. Where this sits relative to prior art

MITO-4 is intentionally *minimal*, and its novelty is in the compression and the framing, not in claiming to out-compute existing systems. Classical self-reproducing cellular automata (von Neumann's universal constructor, Codd, Langton's loops, Byl's small loops) achieve replication but require large multi-cell configurations and long tapes ([Von Neumann universal constructor, Wikipedia](https://en.wikipedia.org/wiki/Von_Neumann_universal_constructor); [Byl, *Self-Reproduction in Small Cellular Automata*, MIT CBA](https://fab.cba.mit.edu/classes/865.18/replication/Byl.pdf)). Continuous CAs like Lenia produce rich lifelike dynamics but operate in floating-point fields, not single integers ([Chan, *Lenia — Biology of Artificial Life*](https://en.wikipedia.org/wiki/Lenia)). MITO-4's contribution is to collapse the *organism* (not the pattern) into one hardware-native word whose single linear switch is simultaneously biological, neural, and perceptron-grade — making it a purpose-built *atom* for an objective-driven agent rather than a general-purpose ALife platform. It trades the universality of von Neumann constructors for the density, determinism, and safety-boundary properties that the Peak Shift architecture actually needs at its floor.

---

## 8. Honest limitations

- **Not yet a universal constructor.** MITO-4 self-*divides*; it does not (in base form) build arbitrary machines. Non-trivial open-ended evolution requires the mutation extension of §4 plus a spatial neighborhood, and that regime is untested at scale.
- **Environment is exogenous.** In the base rule the environment scalar is imposed, not emergent. Genuine ecology (organisms shaping each other's metabolism) needs §6.2 and has not been characterized.
- **Cost/Actor are sketched, not built.** As with AlphaConcept, the agentic half (evaluate a goal, search over actions) is designed at the interface level but not implemented — this is the next module to build, not skip.
- **No held-out biological claim.** MITO-4 is a *functional analog* of mitosis and apoptosis, not a model of any specific organism; the biological citations justify the *form* of the rules, not a fit to wet-lab data.

These are stated in the spirit LeCun endorses: claim only what has receipts, and name the next specific brick to lay.

---

## 9. Summary

MITO-4 gives the "digital switch mimicking biological life inside 4 bytes" concept a name and a shape. Its **name** is the Minimal Integer Threshold Organism. Its **shape** is threefold: a packed 32-bit ribbon (bit geometry), a half-plane hinge partitioned by the line `E = T` (phase geometry), and a breathing `uint32` lattice (lattice geometry). Its **single idea** is that one linear inequality — the shared mathematics of the cell-cycle checkpoint, the firing neuron, and the perceptron — is enough to give a 4-byte integer the full minimal repertoire of life: it grows, it divides, it dies, and it does all of this deterministically, reproducibly, and safely by construction, at the exact width and byte-order that a GPU wants. As the atomic substrate of the Continuum Loop, it is the biological floor on which the perception, memory, imagination, and return organs can finally be grown — a concrete, testable first brick in building an agent from the ground up rather than wrapping one after the fact.

---

## Appendix A — Field encoding reference

```
uint32 layout (little-endian):
  bits  0-7   energy      E   0..255
  bits  8-15  threshold   T   0..255
  bits 16-23  age         A   0..255
  bit    24   alive       L   0/1
  bits 25-31  generation  G   0..127

decision:  divide  iff  (E - T) >= 0
mitosis:   daughters inherit E//2, same T, A=0, L=1, G+1
persist:   E <- E - upkeep ; if E<=0 then L<-0 (dead)
```

## Appendix B — Reproducing the results

```
python3 mito4.py
```
Verifies (1) the 4-byte little-endian round-trip, (2) exponential mitotic growth 1→2→4,
and (3) starvation extinction — all stdlib-only, CPU-deterministic.

---

### Sources

- Yann LeCun, *A Path Towards Autonomous Machine Intelligence* — [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)
- *An overview of programmed cell death in bacteria*, Cell Death & Disease — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4669768/)
- *Altruistic cell death and collective drug resistance*, Molecular Systems Biology — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3531905/)
- *Time-keeping and decision-making in the cell cycle*, Royal Society Interface Focus — [PDF](https://royalsocietypublishing.org/doi/pdf/10.1098/rsfs.2021.0075)
- *Mathematical Model of a Cell Size Checkpoint*, PLOS Computational Biology — [PLOS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1001036)
- *Biochemical switches in the cell cycle*, Wikipedia — [link](https://en.wikipedia.org/wiki/Biochemical_switches_in_the_cell_cycle)
- *Perceptron vs. the point neuron*, IIT Bombay CSE — [PDF](https://www.cse.iitb.ac.in/~cs623/pdf/RM_AI_Lect2_2k6_for_upload.pdf)
- *CUDA C++ Programming Guide* (little-endian hardware), NVIDIA — [docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- *Bitwise Parallel Bulk Computation of the Game of Life on GPU*, IJFCS — [PDF](https://www.worldscientific.com/doi/pdf/10.1142/S0129054116500404)
- *Efficient simulation execution of cellular automata on GPU*, Simulation Modelling Practice and Theory — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1569190X22000259)
- *CAT: Cellular Automata on Tensor cores*, arXiv — [link](https://arxiv.org/html/2406.17284v1)
- *Self-Reproduction and Evolution in Cellular Automata*, arXiv — [link](https://arxiv.org/html/2402.03961v2)
- *Von Neumann universal constructor*, Wikipedia — [link](https://en.wikipedia.org/wiki/Von_Neumann_universal_constructor)
- *Self-Reproduction in Small Cellular Automata* (Byl), MIT CBA — [PDF](https://fab.cba.mit.edu/classes/865.18/replication/Byl.pdf)
- *Lenia*, Wikipedia — [link](https://en.wikipedia.org/wiki/Lenia)
- Internal: `AlphaConcept_Session_Handoff_2026-07-21.md`, `Tonight_On_The_Pod_Plain_English_2026-07-21.md` (Peak Shift Technologies, project files)
