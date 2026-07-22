from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DualSignal:
    name: str
    info_key: str
    step_size: float
    lower_bound: float
    upper_bound: float
    tolerance: float = 0.0

    def validate(self) -> None:
        if self.step_size <= 0.0:
            raise ValueError(f"Dual step size must be positive for {self.name}")
        if self.lower_bound < 0.0 or self.lower_bound > self.upper_bound:
            raise ValueError(f"Invalid dual bounds for {self.name}")


@dataclass(frozen=True)
class DualUpdate:
    means: dict[str, float]
    residuals: dict[str, float]
    multipliers: dict[str, float]


class EpisodeMeanPIDLagrange:
    """PID dual update using the mean signal over a complete episode."""

    def __init__(
        self,
        signals: tuple[DualSignal, ...],
        kp: float = 1.0,
        ki: float = 0.05,
        kd: float = 0.1,
        decay: float = 0.995,
    ):
        if not signals:
            raise ValueError("At least one dual signal is required")
        for signal in signals:
            signal.validate()
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.signals = {item.name: item for item in signals}
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.decay = float(decay)
        self.integral = {name: 0.0 for name in self.signals}
        self.previous = {name: 0.0 for name in self.signals}
        self._sums = {name: 0.0 for name in self.signals}
        self._counts = {name: 0 for name in self.signals}

    def observe(self, info: Mapping[str, float]) -> None:
        for name, signal in self.signals.items():
            if signal.info_key not in info:
                continue
            value = float(info[signal.info_key])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite constraint signal for {name}")
            self._sums[name] += value
            self._counts[name] += 1

    def end_episode(self, multipliers: Mapping[str, float]) -> DualUpdate:
        updated = dict(multipliers)
        means: dict[str, float] = {}
        residuals: dict[str, float] = {}
        for name, signal in self.signals.items():
            count = self._counts[name]
            if count == 0:
                raise RuntimeError(f"No observations were collected for dual signal {name}")
            mean = self._sums[name] / count
            residual = mean - signal.tolerance
            self.integral[name] += residual
            derivative = residual - self.previous[name]
            self.previous[name] = residual
            correction = signal.step_size * (
                self.kp * residual + self.ki * self.integral[name] + self.kd * derivative
            )
            candidate = float(updated.get(name, signal.lower_bound)) + correction
            if residual <= 0.0:
                candidate *= self.decay
                self.integral[name] *= self.decay
            clipped = float(np.clip(candidate, signal.lower_bound, signal.upper_bound))
            if clipped != candidate and np.sign(residual) == np.sign(self.integral[name]):
                self.integral[name] -= residual
            updated[name] = clipped
            means[name] = float(mean)
            residuals[name] = float(residual)
        self.reset_episode()
        return DualUpdate(means=means, residuals=residuals, multipliers=updated)

    def reset_episode(self) -> None:
        for name in self.signals:
            self._sums[name] = 0.0
            self._counts[name] = 0

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"integral": dict(self.integral), "previous": dict(self.previous)}


def make_sb3_callback(controller: EpisodeMeanPIDLagrange):
    """Create an SB3 callback without making Stable-Baselines3 a core dependency."""

    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - exercised only without RL extras
        raise RuntimeError("Install the project with the 'rl' extra to train SAC models") from exc

    class EpisodePIDCallback(BaseCallback):
        def _unwrap_env(self):
            env = self.training_env
            while hasattr(env, "venv"):
                env = env.venv
            if hasattr(env, "envs"):
                env = env.envs[0]
            return getattr(env, "unwrapped", env)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            if not infos:
                return True
            if len(infos) != 1:
                raise RuntimeError("EpisodePIDCallback currently requires one vectorized environment")
            controller.observe(infos[0])
            if bool(dones[0]):
                env = self._unwrap_env()
                result = controller.end_episode(env.lambdas)
                env.lambdas.update(result.multipliers)
                for name, value in result.means.items():
                    self.logger.record(f"lagrange/episode_mean_{name}", value)
                    self.logger.record(f"lagrange/lambda_{name}", result.multipliers[name])
            return True

    return EpisodePIDCallback()

