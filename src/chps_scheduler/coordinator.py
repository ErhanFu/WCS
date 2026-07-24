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
            ch_share = np.clip(
                0.25 + 0.60 * storage + 0.15 * net / max(controllable, 1.0), 0.0, 1.0
            )
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
                    action[cursor + 1] = np.clip(-net / max(plant.max_pumping_mw, 1.0), 0.0, 1.0)
                cursor += 2
            return action.astype(np.float32)

        return policy

    def _execute_short_day(
        self,
        env: ShortTermEnv,
        observation: np.ndarray,
        policy: Policy,
        evaluator: RollingHorizonEvaluator,
        rolling: bool,
        records: list[dict],
    ) -> None:
        terminated = False
        while not terminated:
            row_index = env.day_index * 24 + env.hour
            timestamp = self.data.hourly.iloc[row_index]["timestamp"]
            action = evaluator.select_action(env, policy) if rolling else policy(observation)
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
                "day_index": env.day_index,
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

    def _dispatch_with_plans(
        self,
        plans: list[InterLayerPlan],
        policy: Policy | None,
        rolling: bool,
    ) -> pd.DataFrame:
        env = ShortTermEnv(self.config, self.data, plans)
        evaluator = RollingHorizonEvaluator(self.config)
        records: list[dict] = []
        state = env.initial_hydraulic_state.copy()
        for plan in sorted(plans, key=lambda item: item.day_index):
            observation, _ = env.reset(
                options={"day_index": plan.day_index, "hydraulic_state": state}
            )
            active_policy = policy or self._short_heuristic(env)
            self._execute_short_day(
                env,
                observation,
                active_policy,
                evaluator,
                rolling,
                records,
            )
            state = env.state.copy()
        return pd.DataFrame.from_records(records)

    def _dispatch_closed_loop(
        self,
        long_policy: Policy | None,
        short_policy: Policy | None,
        rolling: bool,
    ) -> pd.DataFrame:
        long_env = LongTermEnv(self.config, self.data)
        long_env.reset()
        active_long_policy = long_policy or self._long_heuristic(long_env)
        evaluator = RollingHorizonEvaluator(self.config)
        records: list[dict] = []
        hydraulic_state = long_env.state.copy()
        water_state = long_env.water_state
        short_env: ShortTermEnv | None = None

        for day_index in range(len(self.data.daily)):
            long_observation = long_env.apply_realized_feedback(
                hydraulic_state,
                water_state,
            )
            long_action = active_long_policy(long_observation)
            _, _, _, _, long_info = long_env.step(long_action)
            plan = long_info["plan"]
            planned_water_state = long_env.water_state

            if short_env is None:
                short_env = ShortTermEnv(
                    self.config,
                    self.data,
                    [plan],
                    initial_state=hydraulic_state,
                )
            else:
                short_env.set_plan(plan)
            observation, _ = short_env.reset(
                options={
                    "day_index": day_index,
                    "hydraulic_state": hydraulic_state,
                    "water_state": planned_water_state,
                }
            )
            active_short_policy = short_policy or self._short_heuristic(short_env)
            self._execute_short_day(
                short_env,
                observation,
                active_short_policy,
                evaluator,
                rolling,
                records,
            )
            hydraulic_state = short_env.state.copy()
            water_state = short_env.water_state

        return pd.DataFrame.from_records(records)

    def dispatch(
        self,
        plans: list[InterLayerPlan] | None = None,
        policy: Policy | None = None,
        rolling: bool = True,
        *,
        long_policy: Policy | None = None,
    ) -> pd.DataFrame:
        """Run closed-loop coordination unless an explicit plan sequence is supplied."""
        if plans is not None:
            return self._dispatch_with_plans(plans, policy, rolling)
        return self._dispatch_closed_loop(long_policy, policy, rolling)
