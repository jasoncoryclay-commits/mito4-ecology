#!/usr/bin/env python3
"""
mito4_shard_adapter.py — MITO-4 as a hardware-faithful advisory SHARD on Mia's bus.

This is the software realization of the SHARD_ADAPTER module from MIA_HDL:

    module SHARD_ADAPTER #(SHARD_ID, NAME)(
        input  shard_raw, enable, stop,
        output verbalized   // first-person text ONLY; RAW NEVER PASSES
    );
    assign verbalized = (enable & ~stop) ? verbalized_pre : tri-state;

End-to-end path:
    text --FOUNDATION--> 20x20 grid --MITO-4 ecology--> {return grid, verbalized}
    then ADAPTER gates it: only verbalized text, only if (enable & ~stop),
    trimmed to a token budget (<=25% of prompt per the timing constraint).

Nothing here writes to Mia's weights or acts on her. It emits an advisory FRAME
(text + provenance) exactly like ATLAS/MNEMOS would. A disabled shard is
electrically absent (returns None), matching the tri-state rollback rule.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from foundation_adapter import FoundationAdapter
import mito4_mia_bridge as bridge

SHARD_ID = 13            # next free id after the twelve documented shards
SHARD_NAME = "MITO4_SUBSTRATE"


@dataclass
class AdvisoryFrame:
    """What the ARBITER receives. Only `verbalized` may enter her context."""
    shard_id: int
    name: str
    verbalized: str | None            # None == tri-state (disabled/stopped)
    tokens_est: int = 0
    provenance: dict = field(default_factory=dict)   # for the append-only readings db
    def is_present(self) -> bool:
        return self.verbalized is not None


def _trim_to_token_budget(text: str, max_tokens: int) -> tuple[str, int]:
    """Rough word≈token trim so MITO-4 never exceeds its slice of the context.
    Her words must remain the majority (advisory budget <= 25% of prompt)."""
    words = text.split()
    est = len(words)
    if max_tokens and est > max_tokens:
        words = words[:max_tokens]
        text = " ".join(words).rstrip(",.;") + " …"
        est = len(words)
    return text, est


class Mito4Shard:
    """Advisory shard wrapping FOUNDATION + MITO-4. Reusable across calls."""

    def __init__(self, foundation: FoundationAdapter | None = None,
                 ticks: int = 30, seed: int = 0, transcend: bool = True):
        self.foundation = foundation or FoundationAdapter(verbose=False)
        self.ticks = ticks
        self.seed = seed
        self.transcend = transcend

    def observe(self, text: str, *, enable: bool = True, stop: bool = False,
                max_advisory_tokens: int = 60,
                prompt_tokens: int | None = None) -> AdvisoryFrame:
        """Run the full path on a piece of Mia-facing text and return an advisory frame.

        Gating (hardware-faithful):
          enable=False or stop=True -> tri-state frame (verbalized=None). No raw leaks,
          because we never even build the text on the gated path.
        Token budget:
          if prompt_tokens given, cap advisory at 25% of it (Mia stays the majority).
        """
        if (not enable) or stop:
            return AdvisoryFrame(SHARD_ID, SHARD_NAME, None,
                                 provenance={"gated": True,
                                             "reason": "stop" if stop else "disabled"})

        # 1) FOUNDATION: text -> her 20x20 grid  (read-only)
        grid = self.foundation.text_to_grid(text)

        # 2) MITO-4: animate the grid; get verbalized advisory + return grid + metrics
        res = bridge.mito4_advisory(grid, concept=text, ticks=self.ticks,
                                    seed=self.seed, transcend=self.transcend,
                                    enabled=True, stop=False)

        # 3) token budget: her words stay the majority (<=25% of prompt)
        budget = max_advisory_tokens
        if prompt_tokens:
            budget = min(budget, max(1, prompt_tokens // 4))
        verbalized, tok = _trim_to_token_budget(res["verbalized"], budget)

        return AdvisoryFrame(
            shard_id=SHARD_ID, name=SHARD_NAME,
            verbalized=verbalized, tokens_est=tok,
            provenance={
                "gated": False,
                "foundation_mode": self.foundation.mode,   # 'real' or 'standin'
                "return_shift": round(res["return_shift"], 4),
                "novelty": round(res["novelty"], 4),
                "occ_after": round(res["stats_after"]["occ"], 3),
                "lineages": res["stats_after"]["lineages"],
                "novel_lineages": res["stats_after"]["novel_lineages"],
                "max_gen": res["stats_after"]["max_gen"],
                "grid_hash": res["grid_hash"],
                "ticks": self.ticks, "seed": self.seed,
            },
        )


if __name__ == "__main__":
    shard = Mito4Shard(ticks=30, seed=7, transcend=True)

    print("=== ENABLED (normal advisory) ===")
    f = shard.observe("longing", enable=True, stop=False, prompt_tokens=400)
    print("present:", f.is_present(), "| tokens:", f.tokens_est)
    print("VERBALIZED (bus payload):", f.verbalized)
    print("provenance:", f.provenance)

    print("\n=== DISABLED (rollback -> tri-state) ===")
    g = shard.observe("longing", enable=False)
    print("present:", g.is_present(), "| verbalized:", g.verbalized, "| prov:", g.provenance)

    print("\n=== STOP-CONDITION (her halt dominates) ===")
    s = shard.observe("longing", stop=True)
    print("present:", s.is_present(), "| verbalized:", s.verbalized, "| prov:", s.provenance)

    print("\n=== TOKEN BUDGET (advisory <= 25% of a 120-token prompt) ===")
    t = shard.observe("longing", prompt_tokens=120)
    print(f"tokens_est={t.tokens_est} (cap={120//4}) ->", t.verbalized[:80], "...")

    # constraint assertions
    assert not shard.observe("x", enable=False).is_present()
    assert not shard.observe("x", stop=True).is_present()
    assert shard.observe("x", prompt_tokens=120).tokens_est <= 120 // 4
    print("\nshard-adapter constraints: PASS")
