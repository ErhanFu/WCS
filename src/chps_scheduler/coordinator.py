from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import SchedulerConfig
from .data import CaseData
from .environments import LongTermEnv, ShortTermEnv
from .models import InterLayerPlan
from .rolling import RollingHorizonEvaluator


Policy = Callable[[np.ndarray], np.ndarray]


class TwoLayerCoordinator:
    def __init__(self, config: SchedulerConfig, data: CaseData):
        data.validate(config)
        self.config = config
        self.data = data

    def _long_heuristic(self, env: LongTermEnv) -> Policy:
        def policy(_: np.ndarray) -> np.ndarray:
            row = env.data.daily.iloc[env.day]
            net = max(0.0, float(row["load_mwh"] - row["wind_mwh"] - row["solar_mwh"]))
            controllable = sum(item.max_power_mw * 24.0 for item in env.config.ch_plants)
            storage = env.physics.storage_factor(env.state)
            ch_share = np.clip(0.25 + 0.60 * storage + 0.15 * net / max(controllable, 1.0), 0.0, 1.0)
            action = np.zeros(env.config.long_action_size, dtype=float)
            action[: len(env.config.ch_plants)] = ch_share
            cursor = len(env.config.ch_plants)
            surplus = max(0.0, float(row["wind_mwh"] + row["solar_mwh"] - row["load_mwh"]))
            for _plant in env.config.ps_plants:
                for segment in env.config.segments:
                    if surplus > 0.0 and segment in ("flat", "valley"):
                        action[cursor : cursor + 2] = (0.0, 0.45)
                    elif net > 0.0 and segment == "peak":
                        action[cursor : cursor + 2] = (0.35, 0.0)
                    cursor += 2
            return action.astype(np.float32)

        return policy

    def build_plans(self, policy: Policy | None = None) -> list[InterLayerPlan]:
        env = LongTermEnv(self.config, self.data)
        observation, _ = env.reset()
        active_policy = policy or self._long_heuristic(env)
        terminated = False
        while not terminated:
            action = active_policy(observation)
            observation, _, terminated, _, _ = env.step(action)
        return list(env.plans)

    def _short_heuristic(self, env: ShortTermEnv) -> Policy:
        def policy(_: np.ndarray) -> np.ndarray:
            row = env._row()
            net = float(row["load_mwh"] - row["wind_mwh"] - row["solar_mwh"])
            action = np.zeros(env.config.short_action_size, dtype=float)
            ch_capacity = sum(item.max_power_mw for item in env.config.ch_plants)
            if net > 0.0:
                action[: len(env.config.ch_plants)] = np.clip(net / max(ch_capacity, 1.0), 0.0, 1.0)
            cursor = len(env.config.ch_plants)
            for plant in env.config.ps_plants:
                if net > ch_capacity:
                    action[cursor] = np.clip(
                        (net - ch_capacity) / max(plant.max_generation_mw, 1.0), 0.0, 1.0
                    )
                elif net < 0.0:
                    action[cursor + 1] = np.clip(
                        -net / max(plant.max_pumping_mw, 1.0), 0.0, 1.0
                    )
                cursor += 2
            return action.astype(np.float32)

        return policy

    def dispatch(
        self,
        plans: list[InterLayerPlan] | None = None,
        policy: Policy | None = None,
        rolling: bool = True,
    ) -> pd.DataFrame:
        plans = plans or self.build_plans()
        env = ShortTermEnv(self.config, self.data, plans)
        evaluator = RollingHorizonEvaluator(self.config)
        records: list[dict] = []
        state = env.initial_hydraulic_state.copy()
        for day_index in range(len(plans)):
            observation, _ = env.reset(
                options={"day_index": day_index, "hydraulic_state": state}
            )
            active_policy = policy or self._short_heuristic(env)
            terminated = False
            while not terminated:
                timestamp = self.data.hourly.iloc[day_index * 24 + env.hour]["timestamp"]
                action = (
                    evaluator.select_action(env, active_policy)
                    if rolling
                    else active_policy(observation)
                )
                observation, reward, terminated, _, info = env.step(action)
                ch_total = sum(info["ch_generation_mwh"].values())
                ps_generation = sum(info["ps_generation_mwh"].values())
                ps_pumping = sum(info["ps_pumping_mwh"].values())
                balance_error = (
                    info["load_mwh"]
                    + ps_pumping
                    - info["wind_mwh"]
                    - info["solar_mwh"]
                    - ch_total
                    - ps_generation
                    - info["purchased_mwh"]
                    + info["curtailed_mwh"]
                )
                record = {
                    "timestamp": timestamp,
                    "day_index": day_index,
                    "hour": info["hour"],
                    "segment": info["segment"],
                    "load_mwh": info["load_mwh"],
                    "wind_mwh": info["wind_mwh"],
                    "solar_mwh": info["solar_mwh"],
                    "ch_generation_mwh": ch_total,
                    "ps_generation_mwh": ps_generation,
                    "ps_pumping_mwh": ps_pumping,
                    "purchased_mwh": info["purchased_mwh"],
                    "curtailed_mwh": info["curtailed_mwh"],
                    "storage_factor": info["storage_factor"],
                    "water_value": info["water_value"],
                    "reward": reward,
                    "balance_error_mwh": balance_error,
                }
                for reservoir_id, storage in env.state.storage_m3.items():
                    record[f"storage_{reservoir_id}_m3"] = storage
                records.append(record)
            state = env.state.copy()
        return pd.DataFrame.from_records(records)

