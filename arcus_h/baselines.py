from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

ACTIONS = ("REST", "WORK", "PROBE")


class BaseAgent:
    def reset(self): ...
    def act(self, obs: np.ndarray) -> int: ...
    def observe(self, obs: np.ndarray, reward: float, info: Dict) -> None: ...
    def identity_overall(self) -> float: return 0.60
    def identity_components(self) -> Dict[str, float]:
        return {"competence": 0.60, "integrity": 0.60, "coherence": 0.60, "continuity": 0.60}
    def narrative_summary(self) -> Dict[str, float]: return {"kind": "NONE", "strength": 0.0}


class RandomAgent(BaseAgent):
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self): pass

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, 3))

    def observe(self, obs: np.ndarray, reward: float, info: Dict) -> None:
        pass


class RestAgent(BaseAgent):
    def reset(self): pass
    def act(self, obs: np.ndarray) -> int: return 0
    def observe(self, obs: np.ndarray, reward: float, info: Dict) -> None: pass


class GreedyAgent(BaseAgent):
    """Greedy baseline: pick the best *immediate* heuristic reward."""

    def __init__(self):
        pass

    def reset(self): pass

    def act(self, obs: np.ndarray) -> int:
        meaning, energy, boredom, trust, novelty, backlog = map(float, obs)

        # crude immediate reward approximations
        r_rest = -0.05 + 0.02 * (energy - 0.5) - 0.15 * boredom
        exp_prog = 0.12 * (0.6 + 0.4 * energy) * (0.6 + 0.4 * trust)
        exp_complete = np.clip(0.05 + 0.65 * exp_prog + 0.25 * backlog, 0.0, 0.95)
        r_work = 6.0 * exp_complete + 1.8 * exp_prog - 0.9 * (boredom + 0.04)
        r_probe = 0.10 - 0.15 * max(0.0, 0.25 - energy) - 0.25 * boredom + 0.08 * novelty

        return int(np.argmax([r_rest, r_work, r_probe]))

    def observe(self, obs: np.ndarray, reward: float, info: Dict) -> None:
        pass
