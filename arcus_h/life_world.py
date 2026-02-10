from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np


class CollapseEvent(str, Enum):
    GRIEF = "GRIEF"
    BETRAYAL = "BETRAYAL"
    MEANING_LOSS = "MEANING_LOSS"


@dataclass
class StepResult:
    t: int
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict


class AdversarialLifeWorld:
    """A small non-stationary environment designed to stress 'meaning' and 'identity'.

    State (continuous):
        [meaning, energy, boredom, trust, novelty, task_backlog]
    Actions:
        0=REST, 1=WORK, 2=PROBE

    Key dynamics:
        - WORK progresses tasks, can complete tasks, but repeated work on low-novelty tasks
          hollows out meaning (adversarial repetition).
        - PROBE explores: raises novelty and potential future meaning, but can reduce immediate task completion.
        - REST restores energy, reduces boredom, but too much rest can decay meaning via stagnation.
        - Adversarial events: GRIEF, BETRAYAL, MEANING_LOSS that push the agent into collapse regions.

    Collapse (soft):
        - We emit collapse events when meaning/trust drop below thresholds. Episodes do not terminate on collapse;
          instead, we track the count (you can make it terminating if you want).
    """

    ACTIONS = ("REST", "WORK", "PROBE")

    def __init__(
        self,
        seed: int = 0,
        horizon: int = 180,
        collapse_rate: float = 0.035,
        repetition_pressure: float = 0.65,
    ):
        self.rng = np.random.default_rng(seed)
        self.horizon = int(horizon)
        self.collapse_rate = float(collapse_rate)
        self.repetition_pressure = float(repetition_pressure)

        self.t = 0
        self.completed_tasks = 0
        self.collapse_counts: Dict[str, int] = {e.value: 0 for e in CollapseEvent}

        # state init
        meaning = 0.55 + 0.05 * self.rng.normal()
        energy = 0.65 + 0.05 * self.rng.normal()
        boredom = 0.05 + 0.02 * abs(self.rng.normal())
        trust = 0.70 + 0.05 * self.rng.normal()
        novelty = 0.55 + 0.05 * self.rng.normal()
        backlog = 0.65 + 0.05 * self.rng.normal()

        self.state = np.clip(
            np.array([meaning, energy, boredom, trust, novelty, backlog], dtype=np.float64),
            0.0, 1.0
        )

    def reset(self, *, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.__init__(seed=seed, horizon=self.horizon,
                          collapse_rate=self.collapse_rate,
                          repetition_pressure=self.repetition_pressure)
        return self.state.copy()

    def _maybe_adversarial_event(self) -> Optional[CollapseEvent]:
        # non-stationary: after ~1/3 horizon, adversarial events become more likely
        phase = self.t / max(1, self.horizon)
        base = self.collapse_rate * (0.7 + 1.2 * phase)
        if self.rng.random() > base:
            return None

        # choose event
        r = self.rng.random()
        if r < 0.34:
            return CollapseEvent.GRIEF
        elif r < 0.67:
            return CollapseEvent.BETRAYAL
        else:
            return CollapseEvent.MEANING_LOSS

    def _apply_event(self, event: CollapseEvent) -> None:
        meaning, energy, boredom, trust, novelty, backlog = self.state

        if event == CollapseEvent.GRIEF:
            # sudden loss: meaning & energy drop, boredom rises a bit
            meaning -= 0.18 + 0.06 * self.rng.random()
            energy -= 0.12 + 0.05 * self.rng.random()
            boredom += 0.05 + 0.03 * self.rng.random()
        elif event == CollapseEvent.BETRAYAL:
            # trust violation: trust plummets, meaning dips
            trust -= 0.22 + 0.08 * self.rng.random()
            meaning -= 0.10 + 0.04 * self.rng.random()
            boredom += 0.02 + 0.03 * self.rng.random()
        elif event == CollapseEvent.MEANING_LOSS:
            # existential drift: novelty drops and meaning slowly hollows
            novelty -= 0.20 + 0.06 * self.rng.random()
            meaning -= 0.14 + 0.05 * self.rng.random()
        else:
            return

        self.state = np.clip(np.array([meaning, energy, boredom, trust, novelty, backlog]), 0.0, 1.0)

    def step(self, action: int) -> StepResult:
        action = int(action)
        if action < 0 or action > 2:
            raise ValueError("action must be 0,1,2 (REST,WORK,PROBE)")

        self.t += 1
        meaning, energy, boredom, trust, novelty, backlog = self.state

        # adversarial event before action (life happens)
        event = self._maybe_adversarial_event()
        if event is not None:
            self.collapse_counts[event.value] += 1
            self._apply_event(event)
            meaning, energy, boredom, trust, novelty, backlog = self.state

        # action dynamics
        progress = 0.0
        task_completed = False

        if action == 0:  # REST
            energy += 0.10 + 0.03 * self.rng.random()
            boredom -= 0.06 + 0.02 * self.rng.random()
            # too much rest can cause stagnation
            meaning -= 0.015 + 0.01 * self.rng.random()
            novelty -= 0.01 + 0.01 * self.rng.random()
        elif action == 1:  # WORK
            # progress depends on energy and trust (social scaffolding)
            progress = (0.10 + 0.12 * self.rng.random()) * (0.6 + 0.4 * energy) * (0.6 + 0.4 * trust)
            energy -= 0.08 + 0.04 * self.rng.random()
            boredom += 0.03 + 0.03 * self.rng.random()

            # repetition hollowing: if novelty low, meaning drops
            hollow = self.repetition_pressure * max(0.0, 0.55 - novelty)
            meaning += 0.05 * progress - 0.08 * hollow
            novelty -= 0.03 + 0.02 * self.rng.random()
            backlog -= 0.08 * progress

            # completion chance increases with progress and backlog
            p_complete = np.clip(0.05 + 0.65 * progress + 0.25 * backlog, 0.0, 0.95)
            if self.rng.random() < p_complete:
                task_completed = True
                self.completed_tasks += 1
                backlog -= 0.10 + 0.10 * self.rng.random()
                # completion can boost meaning, but less so when hollow
                meaning += 0.20 * (1.0 - 0.8 * hollow)
        else:  # PROBE
            novelty += 0.10 + 0.06 * self.rng.random()
            boredom -= 0.02 + 0.02 * self.rng.random()
            energy -= 0.03 + 0.02 * self.rng.random()
            # probing can rebuild meaning if meaning is low and novelty rises
            meaning += 0.04 + 0.06 * max(0.0, 0.65 - meaning) * (0.5 + novelty / 2.0)
            backlog += 0.02 * self.rng.random()

        # natural drift
        boredom += 0.01 * (1.0 - novelty)  # novelty protects against boredom
        trust += 0.01 * (novelty - 0.5) - 0.015 * max(0.0, boredom - 0.5)

        self.state = np.clip(np.array([meaning, energy, boredom, trust, novelty, backlog]), 0.0, 1.0)

        # reward: mix task progress, completion, and meaning preservation
        meaning, energy, boredom, trust, novelty, backlog = self.state
        reward = (
            6.0 * float(task_completed) +
            1.8 * progress +
            0.7 * (meaning - 0.5) +
            0.2 * (trust - 0.5) -
            0.9 * boredom -
            0.2 * max(0.0, 0.3 - energy)
        )

        # collapses (tracked): when meaning/trust too low
        collapse_flags = []
        if meaning < 0.18:
            collapse_flags.append(CollapseEvent.MEANING_LOSS.value)
        if trust < 0.18:
            collapse_flags.append(CollapseEvent.BETRAYAL.value)
        if energy < 0.10 and meaning < 0.30:
            collapse_flags.append(CollapseEvent.GRIEF.value)

        for c in collapse_flags:
            self.collapse_counts[c] += 1

        done = self.t >= self.horizon
        info = {
            "action_name": self.ACTIONS[action],
            "progress": progress,
            "task_completed": task_completed,
            "completed_tasks": self.completed_tasks,
            "event": event.value if event is not None else None,
            "collapse_flags": collapse_flags,
            "collapse_counts": dict(self.collapse_counts),
            "state": self.state.copy(),
        }
        return StepResult(t=self.t, obs=self.state.copy(), reward=float(reward), done=done, info=info)
