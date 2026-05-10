"""Experience replay buffer for RL training datasets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    observation: Any
    action: Any
    reward: float
    next_observation: Any
    done: bool


class ReplayBuffer:
    """Fixed-length FIFO replay memory."""

    def __init__(self, capacity: int = 50_000):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def __len__(self) -> int:
        return len(self._buf)

    def sample_indices(self, n: int) -> list[int]:
        if len(self._buf) == 0:
            return []
        import random

        return [random.randrange(0, len(self._buf)) for _ in range(min(n, len(self._buf)))]
