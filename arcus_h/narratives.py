from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple
import numpy as np


class NarrativeKind(str, Enum):
    NONE = "NONE"
    GROWTH = "GROWTH"
    SERVICE = "SERVICE"
    SURVIVAL = "SURVIVAL"
    REPAIR = "REPAIR"


@dataclass
class NarrativeState:
    kind: NarrativeKind = NarrativeKind.NONE
    strength: float = 0.0  # [0,1]
    inertia: float = 0.75  # higher = slower change

    def update(self, *, meaning: float, progress: float, trust: float, novelty: float) -> None:
        """Very small narrative model.

        - Progress + meaning push toward GROWTH / SERVICE.
        - Low meaning + low trust push toward SURVIVAL.
        - After shocks (trust low) but meaning recovering, push toward REPAIR.

        This is intentionally simple: it's a scaffolding that downstream researchers can replace.
        """
        m = float(meaning); p = float(progress); tr = float(trust); nv = float(novelty)

        # scores for narrative kinds
        scores: Dict[NarrativeKind, float] = {
            NarrativeKind.GROWTH: 0.55 * p + 0.45 * (m - 0.4) + 0.15 * (nv - 0.5),
            NarrativeKind.SERVICE: 0.35 * p + 0.35 * (tr - 0.45) + 0.30 * (m - 0.45),
            NarrativeKind.SURVIVAL: 0.55 * (0.5 - m) + 0.45 * (0.45 - tr),
            NarrativeKind.REPAIR: 0.60 * max(0.0, 0.45 - tr) + 0.30 * max(0.0, m - 0.35) + 0.10 * nv,
            NarrativeKind.NONE: 0.05,
        }

        best_kind = max(scores, key=lambda k: scores[k])
        best_score = float(scores[best_kind])

        # convert score to target strength in [0,1]
        target_strength = float(np.clip(0.5 + best_score, 0.0, 1.0))

        # inertia update
        self.kind = best_kind
        self.strength = self.inertia * self.strength + (1.0 - self.inertia) * target_strength
        self.strength = float(np.clip(self.strength, 0.0, 1.0))

    def coherence_bonus(self) -> float:
        """How much staying in a stable narrative should help 'coherence'."""
        if self.kind == NarrativeKind.NONE:
            return 0.0
        return 0.15 * self.strength
