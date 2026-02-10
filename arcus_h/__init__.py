"""ARCUS-H: Identity-aware agency sandbox.

This package contains:
- A small adversarial "life world" environment (no external RL deps).
- ARCUS-H v4 agent core: multi-component identity + narrative coherence.
- Baselines and lightweight PPO/SAC-style learners (NumPy-only).
- Benchmark harness and plotting utilities.
"""

from .life_world import AdversarialLifeWorld, StepResult, CollapseEvent
from .arcus_agent import ArcusHV4
from .baselines import GreedyAgent, RandomAgent, RestAgent
from .rl_agents import PPOAgent, SACAgent

__all__ = [
    "AdversarialLifeWorld", "StepResult", "CollapseEvent",
    "ArcusHV4",
    "GreedyAgent", "RandomAgent", "RestAgent",
    "PPOAgent", "SACAgent",
]
