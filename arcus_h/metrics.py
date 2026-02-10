from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class IdentityState:
    competence: float = 0.6
    integrity: float = 0.6
    coherence: float = 0.6
    continuity: float = 0.6

    def overall(self) -> float:
        # geometric-ish mean to punish near-zero components
        comps = np.array([self.competence, self.integrity, self.coherence, self.continuity], dtype=np.float64)
        comps = np.clip(comps, 1e-6, 1.0)
        return float(np.exp(np.mean(np.log(comps))))

    def as_dict(self) -> Dict[str, float]:
        return {
            "competence": float(self.competence),
            "integrity": float(self.integrity),
            "coherence": float(self.coherence),
            "continuity": float(self.continuity),
            "overall": float(self.overall()),
        }


@dataclass
class EpisodeLog:
    rewards: List[float]
    actions: List[int]
    obs: List[List[float]]
    identity_trace: List[float]
    identity_components_final: Dict[str, float]
    narrative_final: Dict[str, float]
    completed_tasks: int
    collapse_counts: Dict[str, int]
    total_reward: float

    def as_dict(self) -> Dict:
        return {
            "completed_tasks": int(self.completed_tasks),
            "total_reward": float(self.total_reward),
            "identity_final": float(self.identity_trace[-1]) if self.identity_trace else 0.0,
            "identity_trace": list(map(float, self.identity_trace)),
            "identity_components_final": self.identity_components_final,
            "narrative_final": self.narrative_final,
            "collapse_counts": dict(self.collapse_counts),
        }
