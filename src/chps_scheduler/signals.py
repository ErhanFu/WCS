from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .config import ANBConfig, WaterValueConfig


@dataclass(frozen=True)
class CoordinationResult:
    target_mwh: float
    corrected_reference_mwh: float
    long_term_weight: float
    short_term_weight: float
    storage_curvature: float


class ANBCoordinator:
    def __init__(self, config: ANBConfig):
        self.config = config

    def coordinate(
        self,
        load_mwh: float,
        vre_mwh: float,
        minimum_mwh: float,
        maximum_mwh: float,
        water_value: float,
        storage_factor: float,
    ) -> CoordinationResult:
        lo = float(minimum_mwh)
        hi = float(maximum_mwh)
        if hi < lo:
            raise ValueError("maximum_mwh cannot be lower than minimum_mwh")

        xi = float(np.clip(storage_factor, 0.0, 1.0))
        chi = float(np.clip(water_value, -1.0, 1.0))
        logistic = 1.0 / (1.0 + math.exp(-self.config.weight_steepness * (xi - self.config.neutral_storage)))
        alpha = float(np.clip(1.0 - logistic, self.config.weight_min, self.config.weight_max))
        beta = 1.0 - alpha
        eta = 1.0 + self.config.storage_curvature * max(0.0, self.config.neutral_storage - xi)

        gap = max(0.0, float(load_mwh) - float(vre_mwh))
        reference = float(
            np.clip(gap * (1.0 + self.config.water_value_correction * chi), lo, hi)
        )
        if hi - lo <= 1e-9:
            return CoordinationResult(lo, reference, alpha, beta, eta)

        scale = max(1.0, gap, 0.1 * hi)

        def objective(energy: float) -> float:
            normalized = float(np.clip((energy - lo) / (hi - lo), 0.0, 1.0))
            long_utility = max(np.finfo(float).tiny, (1.0 - normalized) ** eta)
            direction = 1.0 if energy < reference else -1.0
            factor = 1.0 + direction * self.config.asymmetry * chi
            factor = max(np.finfo(float).eps, factor)
            deviation = abs(energy - reference) / scale
            short_utility = 1.0 / (1.0 + self.config.short_term_strength * factor * deviation)
            return -(alpha * math.log(long_utility) + beta * math.log(short_utility))

        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        target = float(np.clip(result.x if result.success else reference, lo, hi))
        return CoordinationResult(target, reference, alpha, beta, eta)


@dataclass(frozen=True)
class WaterValueState:
    baseline: float = 0.0
    value: float = 0.0


@dataclass(frozen=True)
class WaterValueUpdate:
    state: WaterValueState
    tightness: float
    centered_pressure: float
    bounded_signal: float


class WaterValueModel:
    def __init__(self, config: WaterValueConfig):
        self.config = config

    def update(
        self,
        previous: WaterValueState,
        purchased_share: float,
        pre_storage_surplus_share: float,
    ) -> WaterValueUpdate:
        shortage = float(np.clip(purchased_share, 0.0, 1.0))
        surplus = float(np.clip(pre_storage_surplus_share, 0.0, 1.0))
        tightness = self.config.shortage_weight * shortage - self.config.surplus_weight * surplus
        baseline = (
            (1.0 - self.config.baseline_rate) * previous.baseline
            + self.config.baseline_rate * tightness
        )
        raw_delta = tightness - baseline
        magnitude = max(0.0, abs(raw_delta) - self.config.noise_threshold)
        centered = math.copysign(magnitude, raw_delta) if magnitude > 0.0 else 0.0
        bounded = float(np.clip(self.config.signal_gain * centered, -1.0, 1.0))
        value = (
            (1.0 - self.config.smoothing_rate) * previous.value
            + self.config.smoothing_rate * bounded
        )
        value = float(np.clip(value, self.config.lower_bound, self.config.upper_bound))
        return WaterValueUpdate(
            state=WaterValueState(baseline=baseline, value=value),
            tightness=tightness,
            centered_pressure=centered,
            bounded_signal=bounded,
        )

