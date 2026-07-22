from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import SchedulerConfig
from .environments import ShortTermEnv


Policy = Callable[[np.ndarray], np.ndarray]


class RollingHorizonEvaluator:
    """Evaluate structured candidates and execute only the first selected action."""

    def __init__(self, config: SchedulerConfig):
        self.config = config

    def _rollout_policy(
        self,
        env: ShortTermEnv,
        policy: Policy,
        horizon: int,
    ) -> tuple[np.ndarray, list[dict], float]:
        snapshot = env.snapshot()
        actions: list[np.ndarray] = []
        infos: list[dict] = []
        total_reward = 0.0
        try:
            for _ in range(horizon):
                observation = env._observation()
                action = np.asarray(policy(observation), dtype=float)
                _, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
                infos.append(info)
                total_reward += float(reward)
                if terminated or truncated:
                    break
        finally:
            env.restore(snapshot)
        if not actions:
            return np.empty((0, self.config.short_action_size)), [], 0.0
        return np.vstack(actions), infos, total_reward

    def _candidate_variants(
        self,
        baseline: np.ndarray,
        infos: list[dict],
    ) -> list[np.ndarray]:
        ch_count = len(self.config.ch_plants)
        variants = [baseline.copy()]

        surplus = baseline.copy()
        for index, info in enumerate(infos):
            if info["segment"] == "valley" and info["curtailed_mwh"] > 0.0:
                for ps_index in range(len(self.config.ps_plants)):
                    gen = ch_count + 2 * ps_index
                    pump = gen + 1
                    surplus[index, gen] *= 0.70
                    surplus[index, pump] = np.clip(surplus[index, pump] + 0.30, 0.0, 1.0)
        variants.append(surplus)

        shortage = baseline.copy()
        for index, info in enumerate(infos):
            if info["segment"] == "peak" and info["purchased_mwh"] > 0.0:
                shortage[index, :ch_count] = np.clip(
                    shortage[index, :ch_count] + 0.20, 0.0, 1.0
                )
                for ps_index in range(len(self.config.ps_plants)):
                    gen = ch_count + 2 * ps_index
                    pump = gen + 1
                    shortage[index, pump] *= 0.50
        variants.append(shortage)

        late_pumping = baseline.copy()
        for index, info in enumerate(infos):
            if (
                info["hour"] >= 16
                and info["segment"] in ("flat", "valley")
                and info["remaining_pumping_mwh"] > 0.0
            ):
                for ps_index in range(len(self.config.ps_plants)):
                    gen = ch_count + 2 * ps_index
                    pump = gen + 1
                    late_pumping[index, gen] *= 0.80
                    late_pumping[index, pump] = np.clip(
                        late_pumping[index, pump] + 0.25, 0.0, 1.0
                    )
        variants.append(late_pumping)

        storage_safety = baseline.copy()
        for index, info in enumerate(infos):
            for ps_index, plant in enumerate(self.config.ps_plants):
                gen = ch_count + 2 * ps_index
                pump = gen + 1
                upper_margin_m = info["upper_level_margins_m"][plant.id]
                spill = info["spill_m3"].get(plant.upper_reservoir_id, 0.0)
                if spill > 0.0:
                    storage_safety[index, gen] = np.clip(
                        storage_safety[index, gen] + 0.30, 0.0, 1.0
                    )
                    storage_safety[index, pump] *= 0.30
                elif upper_margin_m < 0.50:
                    storage_safety[index, gen] = np.clip(
                        storage_safety[index, gen] + 0.20, 0.0, 1.0
                    )
                    storage_safety[index, pump] *= 0.70
        variants.append(storage_safety)

        state_response = baseline.copy()
        for index, info in enumerate(infos):
            storage_factor = info["storage_factor"]
            if info["segment"] == "peak" and storage_factor > 0.70:
                state_response[index, :ch_count] = np.clip(
                    state_response[index, :ch_count] + 0.15, 0.0, 1.0
                )
                for ps_index in range(len(self.config.ps_plants)):
                    gen = ch_count + 2 * ps_index
                    pump = gen + 1
                    state_response[index, gen] = np.clip(
                        state_response[index, gen] + 0.10, 0.0, 1.0
                    )
            if storage_factor > 0.80:
                for ps_index in range(len(self.config.ps_plants)):
                    pump = ch_count + 2 * ps_index + 1
                    state_response[index, pump] *= 0.60
            if (
                info["segment"] == "peak"
                and storage_factor < 0.30
                and info["purchased_mwh"] < 0.30 * info["load_mwh"]
            ):
                state_response[index, :ch_count] *= 0.80
        variants.append(state_response)
        return variants[: self.config.candidate_count]

    @staticmethod
    def _evaluate(env: ShortTermEnv, actions: np.ndarray) -> float:
        snapshot = env.snapshot()
        total_reward = 0.0
        try:
            for action in actions:
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
        finally:
            env.restore(snapshot)
        return total_reward

    def select_action(self, env: ShortTermEnv, policy: Policy) -> np.ndarray:
        remaining = 24 - env.hour
        horizon = min(self.config.rolling_horizon, remaining)
        baseline, infos, _ = self._rollout_policy(env, policy, horizon)
        if len(baseline) == 0:
            return np.asarray(policy(env._observation()), dtype=float)
        candidates = self._candidate_variants(baseline, infos)
        values = [self._evaluate(env, candidate) for candidate in candidates]
        return np.asarray(candidates[int(np.argmax(values))][0], dtype=float)
