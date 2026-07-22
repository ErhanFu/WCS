from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LevelStorageCurve:
    """Monotone piecewise-linear level-storage mapping."""

    levels_m: np.ndarray
    storage_m3: np.ndarray

    @classmethod
    def from_nodes(cls, nodes: Iterable[tuple[float, float]]) -> "LevelStorageCurve":
        array = np.asarray(tuple(nodes), dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
            raise ValueError("Level-storage nodes must have shape (n, 2), n >= 2")
        if np.any(np.diff(array[:, 0]) <= 0) or np.any(np.diff(array[:, 1]) <= 0):
            raise ValueError("Level-storage nodes must be strictly monotone")
        return cls(array[:, 0], array[:, 1])

    def storage(self, level_m: float) -> float:
        return float(np.interp(level_m, self.levels_m, self.storage_m3))

    def level(self, storage_m3: float) -> float:
        return float(np.interp(storage_m3, self.storage_m3, self.levels_m))


@dataclass(frozen=True)
class PHQPolynomial:
    """Normalized polynomial proxy q_hat = sum beta_ij p_hat^i h_hat^j."""

    terms: tuple[tuple[int, int, float], ...]

    def evaluate(self, normalized_power: float, normalized_head: float) -> float:
        p = float(np.clip(normalized_power, 0.0, 1.0))
        h = float(np.clip(normalized_head, 0.0, 1.0))
        value = sum(beta * p**i * h**j for i, j, beta in self.terms)
        return float(max(0.0, value))


RHO_WATER = 1000.0
GRAVITY = 9.81
JOULES_PER_MWH = 3.6e9


def generation_release_m3(energy_mwh: float, head_m: float, efficiency: float) -> float:
    """Convert generated electrical energy to released water volume."""

    if energy_mwh <= 0.0:
        return 0.0
    denominator = RHO_WATER * GRAVITY * max(head_m, 1e-6) * max(efficiency, 1e-6)
    return float(energy_mwh * JOULES_PER_MWH / denominator)


def pumping_transfer_m3(energy_mwh: float, head_m: float, efficiency: float) -> float:
    """Convert pumping electrical energy to transferred water volume."""

    if energy_mwh <= 0.0:
        return 0.0
    numerator = energy_mwh * JOULES_PER_MWH * max(efficiency, 0.0)
    denominator = RHO_WATER * GRAVITY * max(head_m, 1e-6)
    return float(numerator / denominator)

