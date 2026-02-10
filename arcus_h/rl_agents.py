from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

ACTIONS = ("REST", "WORK", "PROBE")


def _featurize(obs: np.ndarray) -> np.ndarray:
    """Small feature map for linear policies (no neural nets)."""
    obs = np.asarray(obs, dtype=np.float64)
    m, e, b, tr, nv, bl = obs
    return np.array([
        1.0,
        m, e, b, tr, nv, bl,
        m * nv,
        e * tr,
        (1.0 - b) * nv,
        bl * e,
    ], dtype=np.float64)


class BaseRLAgent:
    def reset(self): ...
    def act(self, obs: np.ndarray) -> int: ...
    def observe(self, obs: np.ndarray, reward: float, info: Dict) -> None: ...
    def identity_overall(self) -> float: return 0.60
    def identity_components(self) -> Dict[str, float]:
        return {"competence": 0.60, "integrity": 0.60, "coherence": 0.60, "continuity": 0.60}
    def narrative_summary(self) -> Dict[str, float]: return {"kind": "NONE", "strength": 0.0}


@dataclass
class PPOConfig:
    lr: float = 0.05
    gamma: float = 0.97
    clip: float = 0.2
    ent: float = 0.02
    temperature: float = 0.9


class PPOAgent(BaseRLAgent):
    """Tiny NumPy-only PPO-style agent.

    Note: this is NOT stable-baselines PPO. It's a minimal learning baseline so you can
    compare ARCUS-H against a reward-only learner without external deps.
    """

    def __init__(self, seed: int = 0, cfg: Optional[PPOConfig] = None):
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg or PPOConfig()
        self.W = self.rng.normal(scale=0.01, size=(3, 11))
        self.V = self.rng.normal(scale=0.01, size=(11,))
        self.traj = []

    def reset(self):
        self.traj = []

    def _policy(self, x: np.ndarray) -> np.ndarray:
        logits = (self.W @ x) / max(1e-6, self.cfg.temperature)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        return p / np.sum(p)

    def act(self, obs: np.ndarray) -> int:
        x = _featurize(obs)
        p = self._policy(x)
        a = int(self.rng.choice([0, 1, 2], p=p))
        self.traj.append((x, a, p[a]))
        return a

    def observe(self, obs: np.ndarray, reward: float, info: Dict) -> None:
        # store reward and value baseline
        x = _featurize(obs)
        v = float(np.dot(self.V, x))
        self.traj[-1] = (*self.traj[-1], float(reward), v)

    def finish_episode(self):
        if not self.traj:
            return

        cfg = self.cfg
        # compute returns and advantages
        returns = []
        G = 0.0
        for (_, _, _, r, _) in reversed(self.traj):
            G = r + cfg.gamma * G
            returns.append(G)
        returns = list(reversed(returns))

        # normalize returns
        ret = np.array(returns, dtype=np.float64)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        for i, (x, a, oldp, r, v) in enumerate(self.traj):
            adv = float(ret[i] - v)
            # recompute p under current policy
            p = self._policy(x)
            newp = float(p[a])
            ratio = newp / max(1e-8, float(oldp))

            # clipped surrogate
            s1 = ratio * adv
            s2 = float(np.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)) * adv
            obj = min(s1, s2)

            # entropy bonus
            ent = -float(np.sum(p * np.log(p + 1e-9)))
            obj_total = obj + cfg.ent * ent

            # gradient of logpi for softmax linear
            # grad logits = (onehot(a) - p)
            dlog = -p
            dlog[a] += 1.0
            gradW = np.outer(dlog, x) * obj_total

            # value gradient (MSE)
            err = (ret[i] - v)
            gradV = err * x

            self.W += cfg.lr * gradW
            self.V += 0.5 * cfg.lr * gradV

        self.traj = []


@dataclass
class SACConfig:
    lr_q: float = 0.06
    lr_pi: float = 0.04
    gamma: float = 0.97
    alpha: float = 0.12
    temperature: float = 0.9


class SACAgent(BaseRLAgent):
    """Tiny NumPy-only SAC-style agent (discrete actions)."""

    def __init__(self, seed: int = 0, cfg: Optional[SACConfig] = None):
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg or SACConfig()
        self.Q = self.rng.normal(scale=0.01, size=(3, 11))  # linear Q per action
        self.W = self.rng.normal(scale=0.01, size=(3, 11))  # policy weights
        self.last = None

    def reset(self):
        self.last = None

    def _policy(self, x: np.ndarray) -> np.ndarray:
        logits = (self.W @ x) / max(1e-6, self.cfg.temperature)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        return p / np.sum(p)

    def _qvals(self, x: np.ndarray) -> np.ndarray:
        return self.Q @ x

    def act(self, obs: np.ndarray) -> int:
        x = _featurize(obs)
        p = self._policy(x)
        a = int(self.rng.choice([0, 1, 2], p=p))
        self.last = (x, a, p)
        return a

    def observe(self, obs: np.ndarray, reward: float, info: Dict) -> None:
        if self.last is None:
            return
        cfg = self.cfg
        x, a, p = self.last
        x2 = _featurize(obs)

        # target using soft value under current policy
        q2 = self._qvals(x2)
        p2 = self._policy(x2)
        v2 = float(np.sum(p2 * (q2 - cfg.alpha * np.log(p2 + 1e-9))))
        target = float(reward + cfg.gamma * v2)

        # Q update (TD)
        q = float(np.dot(self.Q[a], x))
        td = target - q
        self.Q[a] += cfg.lr_q * td * x

        # policy update: minimize KL to exp(Q/alpha) (soft actor)
        q_now = self._qvals(x)
        # desired distribution
        logits = q_now / max(1e-6, cfg.alpha)
        logits = logits - np.max(logits)
        d = np.exp(logits)
        d = d / np.sum(d)

        # gradient for softmax linear weights
        p_now = self._policy(x)
        grad = (d - p_now)  # push toward desired
        self.W += cfg.lr_pi * np.outer(grad, x)

        self.last = None
