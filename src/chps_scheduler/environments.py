from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SchedulerConfig
from .curves import LevelStorageCurve
from .data import CaseData
from .gym_compat import gym, spaces
from .models import HydraulicState, InterLayerPlan
from .physics import HydraulicModel, project_ps_mode
from .signals import ANBCoordinator, WaterValueModel, WaterValueState


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-9))


def _segment_for_hour(config: SchedulerConfig, hour: int) -> str:
    cursor = 0
    for segment in config.segments:
        cursor += config.segment_hours[segment]
        if hour < cursor:
            return segment
    return config.segments[-1]


@dataclass
class ShortSnapshot:
    hydraulic_state: HydraulicState
    water_state: WaterValueState
    hour: int
    remaining_ch: dict[str, dict[str, float]]
    remaining_ps_generation: dict[str, dict[str, float]]
    remaining_ps_pumping: dict[str, dict[str, float]]


class LongTermEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: SchedulerConfig, data: CaseData):
        super().__init__()
        data.validate(config)
        self.config = config
        self.data = data
        self.physics = HydraulicModel(config)
        self.anb = ANBCoordinator(config.anb)
        self.water_model = WaterValueModel(config.water_value)
        self.action_space = spaces.Box(0.0, 1.0, shape=(config.long_action_size,), dtype=np.float32)
        self._obs_size = 2 * len(config.reservoirs) + 7
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(self._obs_size,), dtype=np.float32)
        self.max_load = max(float(data.daily["load_mwh"].max()), 1.0)
        self.max_vre = max(float((data.daily["wind_mwh"] + data.daily["solar_mwh"]).max()), 1.0)
        self.max_inflow = {
            item.id: max(float(data.daily[f"inflow_{item.id}_m3s"].max()), 1.0)
            for item in config.reservoirs
        }
        self.lambdas = {
            "target": 1.0,
            "curtailment": 1.0,
            "purchased": 1.0,
            "curtailment_share": 1.0,
            "purchased_share": 1.0,
        }
        self.plans: list[InterLayerPlan] = []
        self.reset(seed=config.seed)

    def _observation(self) -> np.ndarray:
        row = self.data.daily.iloc[min(self.day, len(self.data.daily) - 1)]
        storage = []
        inflow = []
        for reservoir in self.config.reservoirs:
            span = reservoir.max_storage_m3 - reservoir.min_storage_m3
            storage.append(
                2.0 * (self.state.storage_m3[reservoir.id] - reservoir.min_storage_m3) / span - 1.0
            )
            inflow.append(float(row[f"inflow_{reservoir.id}_m3s"]) / self.max_inflow[reservoir.id])
        load = float(row["load_mwh"])
        vre = float(row["wind_mwh"] + row["solar_mwh"])
        phase = 2.0 * math.pi * self.day / max(len(self.data.daily), 1)
        values = storage + [load / self.max_load, vre / self.max_vre, (load - vre) / self.max_load]
        values += inflow + [math.sin(phase), math.cos(phase)]
        values += [self.water_state.value, self.water_state.baseline]
        return np.asarray(values, dtype=np.float32)

    def _decode_action(self, action: np.ndarray):
        values = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
        if values.shape != (self.config.long_action_size,):
            raise ValueError(f"Expected long action shape {(self.config.long_action_size,)}")
        cursor = 0
        ch_quota: dict[str, dict[str, float]] = {}
        for plant in self.config.ch_plants:
            daily_energy = values[cursor] * plant.max_power_mw * 24.0
            cursor += 1
            ch_quota[plant.id] = {
                segment: daily_energy * self.config.segment_hours[segment] / 24.0
                for segment in self.config.segments
            }
        ps_generation: dict[str, dict[str, float]] = {}
        ps_pumping: dict[str, dict[str, float]] = {}
        for plant in self.config.ps_plants:
            ps_generation[plant.id] = {}
            ps_pumping[plant.id] = {}
            for segment in self.config.segments:
                gen_share, pump_share = project_ps_mode(values[cursor], values[cursor + 1])
                cursor += 2
                hours = self.config.segment_hours[segment]
                ps_generation[plant.id][segment] = gen_share * plant.max_generation_mw * hours
                ps_pumping[plant.id][segment] = pump_share * plant.max_pumping_mw * hours
        return ch_quota, ps_generation, ps_pumping

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.day = 0
        self.state = self.physics.initial_state()
        self.water_state = WaterValueState()
        self.plans = []
        return self._observation(), {}

    def step(self, action: np.ndarray):
        row = self.data.daily.iloc[self.day]
        ch_quota, ps_generation, ps_pumping = self._decode_action(action)
        requested_ch = {key: sum(value.values()) for key, value in ch_quota.items()}
        requested_ps_generation = {key: sum(value.values()) for key, value in ps_generation.items()}
        requested_ps_pumping = {key: sum(value.values()) for key, value in ps_pumping.items()}
        load = float(row["load_mwh"])
        wind = float(row["wind_mwh"])
        solar = float(row["solar_mwh"])
        storage_factor = self.physics.storage_factor(self.state)
        maximum = sum(item.max_power_mw * 24.0 for item in self.config.ch_plants)
        maximum += sum(item.max_generation_mw * 24.0 for item in self.config.ps_plants)
        coordination = self.anb.coordinate(
            load_mwh=load,
            vre_mwh=wind + solar,
            minimum_mwh=0.0,
            maximum_mwh=maximum,
            water_value=self.water_state.value,
            storage_factor=storage_factor,
        )
        result = self.physics.dispatch(
            state=self.state,
            duration_hours=24.0,
            load_mwh=load,
            wind_mwh=wind,
            solar_mwh=solar,
            inflow_m3s={
                item.id: float(row[f"inflow_{item.id}_m3s"]) for item in self.config.reservoirs
            },
            requested_ch_mwh=requested_ch,
            requested_ps_generation_mwh=requested_ps_generation,
            requested_ps_pumping_mwh=requested_ps_pumping,
        )
        target_error = abs(result.controllable_generation_mwh - coordination.target_mwh)
        scale = max(load, 1.0)
        purchased_share = _safe_ratio(result.purchased_mwh, load)
        vre = wind + solar
        pre_storage_surplus_share = _safe_ratio(max(0.0, vre - load), vre)
        water_update = self.water_model.update(
            self.water_state, purchased_share, pre_storage_surplus_share
        )
        self.water_state = water_update.state

        signals = {
            "g_target": target_error / 1000.0,
            "g_curtailment": result.curtailed_mwh / 1000.0,
            "g_purchased": result.purchased_mwh / 1000.0,
            "g_curtailment_share": _safe_ratio(result.curtailed_mwh, vre),
            "g_purchased_share": purchased_share,
        }
        penalty = (
            self.lambdas["target"] * signals["g_target"]
            + self.lambdas["curtailment"] * signals["g_curtailment"]
            + self.lambdas["purchased"] * signals["g_purchased"]
            + self.lambdas["curtailment_share"] * signals["g_curtailment_share"]
            + self.lambdas["purchased_share"] * signals["g_purchased_share"]
        )
        reward = -(
            result.purchased_mwh + result.curtailed_mwh + 0.25 * target_error
        ) / scale - penalty
        plan = InterLayerPlan(
            day_index=self.day,
            ch_quota_mwh=ch_quota,
            ps_generation_quota_mwh=ps_generation,
            ps_pumping_quota_mwh=ps_pumping,
            target_storage_m3=dict(self.state.storage_m3),
            water_value=self.water_state.value,
            storage_factor=storage_factor,
            long_term_weight=coordination.long_term_weight,
            short_term_weight=coordination.short_term_weight,
            coordinated_target_mwh=coordination.target_mwh,
        )
        self.plans.append(plan)
        info = {
            **signals,
            "day_index": self.day,
            "purchased_mwh": result.purchased_mwh,
            "curtailed_mwh": result.curtailed_mwh,
            "target_mwh": coordination.target_mwh,
            "storage_factor": storage_factor,
            "water_value": self.water_state.value,
            "plan": plan,
        }
        self.day += 1
        terminated = self.day >= len(self.data.daily)
        observation = np.zeros(self._obs_size, dtype=np.float32) if terminated else self._observation()
        return observation, float(reward), terminated, False, info


class ShortTermEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SchedulerConfig,
        data: CaseData,
        plans: list[InterLayerPlan],
        initial_state: HydraulicState | None = None,
    ):
        super().__init__()
        data.validate(config)
        if len(plans) != len(data.daily):
            raise ValueError("One inter-layer plan is required for every day")
        self.config = config
        self.data = data
        self.plans = plans
        self.physics = HydraulicModel(config)
        self.anb = ANBCoordinator(config.anb)
        self.water_model = WaterValueModel(config.water_value)
        self.level_storage_curves = {
            reservoir.id: LevelStorageCurve.from_nodes(reservoir.level_storage_nodes)
            for reservoir in config.reservoirs
        }
        self.initial_hydraulic_state = initial_state or self.physics.initial_state()
        self.action_space = spaces.Box(0.0, 1.0, shape=(config.short_action_size,), dtype=np.float32)
        quota_terms = len(config.ch_plants) + 2 * len(config.ps_plants)
        self._obs_size = len(config.reservoirs) + 3 + quota_terms + 4
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(self._obs_size,), dtype=np.float32)
        self.max_load = max(float(data.hourly["load_mwh"].max()), 1.0)
        self.max_vre = max(float((data.hourly["wind_mwh"] + data.hourly["solar_mwh"]).max()), 1.0)
        self.lambdas = {
            "segment_quota": 1.0,
            "terminal_storage": 1.0,
            "purchased": 1.0,
            "curtailment": 1.0,
            "purchased_share": 1.0,
            "curtailment_share": 1.0,
            "pumping_quota": 1.0,
        }
        self.next_day = 0
        self.state = self.initial_hydraulic_state.copy()
        self.reset(seed=config.seed)

    def _load_day(self, day_index: int) -> None:
        self.day_index = int(day_index)
        self.hour = 0
        self.plan = self.plans[self.day_index]
        self.water_state = WaterValueState(value=self.plan.water_value)
        self.remaining_ch = copy.deepcopy(self.plan.ch_quota_mwh)
        self.remaining_ps_generation = copy.deepcopy(self.plan.ps_generation_quota_mwh)
        self.remaining_ps_pumping = copy.deepcopy(self.plan.ps_pumping_quota_mwh)

    def _row(self):
        return self.data.hourly.iloc[self.day_index * 24 + self.hour]

    def _remaining_total(self, mapping: dict[str, dict[str, float]], plant_id: str) -> float:
        return float(sum(mapping[plant_id].values()))

    def _observation(self) -> np.ndarray:
        row = self._row()
        storage = []
        for reservoir in self.config.reservoirs:
            span = reservoir.max_storage_m3 - reservoir.min_storage_m3
            storage.append(
                2.0 * (self.state.storage_m3[reservoir.id] - reservoir.min_storage_m3) / span - 1.0
            )
        load = float(row["load_mwh"])
        vre = float(row["wind_mwh"] + row["solar_mwh"])
        quota = []
        for plant in self.config.ch_plants:
            maximum = max(plant.max_power_mw * 24.0, 1.0)
            quota.append(self._remaining_total(self.remaining_ch, plant.id) / maximum)
        for plant in self.config.ps_plants:
            quota.append(
                self._remaining_total(self.remaining_ps_generation, plant.id)
                / max(plant.max_generation_mw * 24.0, 1.0)
            )
            quota.append(
                self._remaining_total(self.remaining_ps_pumping, plant.id)
                / max(plant.max_pumping_mw * 24.0, 1.0)
            )
        phase = 2.0 * math.pi * self.hour / 24.0
        values = storage + [load / self.max_load, vre / self.max_vre, (load - vre) / self.max_load]
        values += quota + [math.sin(phase), math.cos(phase), self.water_state.value, self.plan.storage_factor]
        return np.asarray(values, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "hydraulic_state" in options:
            self.state = options["hydraulic_state"].copy()
        elif options.get("reset_hydraulic_state", False):
            self.state = self.initial_hydraulic_state.copy()
        day_index = int(options.get("day_index", self.next_day % len(self.plans)))
        self.next_day = (day_index + 1) % len(self.plans)
        self._load_day(day_index)
        return self._observation(), {"day_index": day_index}

    def snapshot(self) -> ShortSnapshot:
        return ShortSnapshot(
            hydraulic_state=self.state.copy(),
            water_state=copy.deepcopy(self.water_state),
            hour=self.hour,
            remaining_ch=copy.deepcopy(self.remaining_ch),
            remaining_ps_generation=copy.deepcopy(self.remaining_ps_generation),
            remaining_ps_pumping=copy.deepcopy(self.remaining_ps_pumping),
        )

    def restore(self, snapshot: ShortSnapshot) -> None:
        self.state = snapshot.hydraulic_state.copy()
        self.water_state = copy.deepcopy(snapshot.water_state)
        self.hour = snapshot.hour
        self.remaining_ch = copy.deepcopy(snapshot.remaining_ch)
        self.remaining_ps_generation = copy.deepcopy(snapshot.remaining_ps_generation)
        self.remaining_ps_pumping = copy.deepcopy(snapshot.remaining_ps_pumping)

    def _take_from_quota(
        self,
        mapping: dict[str, dict[str, float]],
        plant_id: str,
        segment: str,
        requested: float,
    ) -> float:
        available = max(0.0, float(mapping[plant_id][segment]))
        used = min(max(0.0, requested), available)
        mapping[plant_id][segment] = available - used
        return used

    def step(self, action: np.ndarray):
        values = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
        if values.shape != (self.config.short_action_size,):
            raise ValueError(f"Expected short action shape {(self.config.short_action_size,)}")
        row = self._row()
        segment = _segment_for_hour(self.config, self.hour)
        cursor = 0
        requested_ch: dict[str, float] = {}
        for plant in self.config.ch_plants:
            request = values[cursor] * plant.max_power_mw
            cursor += 1
            requested_ch[plant.id] = self._take_from_quota(
                self.remaining_ch, plant.id, segment, request
            )
        requested_ps_generation: dict[str, float] = {}
        requested_ps_pumping: dict[str, float] = {}
        for plant in self.config.ps_plants:
            gen_share, pump_share = project_ps_mode(values[cursor], values[cursor + 1])
            cursor += 2
            requested_ps_generation[plant.id] = self._take_from_quota(
                self.remaining_ps_generation,
                plant.id,
                segment,
                gen_share * plant.max_generation_mw,
            )
            requested_ps_pumping[plant.id] = self._take_from_quota(
                self.remaining_ps_pumping,
                plant.id,
                segment,
                pump_share * plant.max_pumping_mw,
            )
        load = float(row["load_mwh"])
        wind = float(row["wind_mwh"])
        solar = float(row["solar_mwh"])
        storage_factor = self.physics.storage_factor(self.state)
        maximum = sum(item.max_power_mw for item in self.config.ch_plants)
        maximum += sum(item.max_generation_mw for item in self.config.ps_plants)
        coordination = self.anb.coordinate(
            load_mwh=load,
            vre_mwh=wind + solar,
            minimum_mwh=0.0,
            maximum_mwh=maximum,
            water_value=self.water_state.value,
            storage_factor=storage_factor,
        )
        result = self.physics.dispatch(
            state=self.state,
            duration_hours=1.0,
            load_mwh=load,
            wind_mwh=wind,
            solar_mwh=solar,
            inflow_m3s={
                item.id: float(row[f"inflow_{item.id}_m3s"]) for item in self.config.reservoirs
            },
            requested_ch_mwh=requested_ch,
            requested_ps_generation_mwh=requested_ps_generation,
            requested_ps_pumping_mwh=requested_ps_pumping,
        )
        # Consume quotas according to executable energy, not infeasible requests.
        for plant_id, requested in requested_ch.items():
            self.remaining_ch[plant_id][segment] += max(
                0.0, requested - result.ch_generation_mwh[plant_id]
            )
        for plant_id, requested in requested_ps_generation.items():
            self.remaining_ps_generation[plant_id][segment] += max(
                0.0, requested - result.ps_generation_mwh[plant_id]
            )
        for plant_id, requested in requested_ps_pumping.items():
            self.remaining_ps_pumping[plant_id][segment] += max(
                0.0, requested - result.ps_pumping_mwh[plant_id]
            )
        target_error = abs(result.controllable_generation_mwh - coordination.target_mwh)
        vre = wind + solar
        purchased_share = _safe_ratio(result.purchased_mwh, load)
        pre_storage_surplus_share = _safe_ratio(max(0.0, vre - load), vre)
        self.water_state = self.water_model.update(
            self.water_state, purchased_share, pre_storage_surplus_share
        ).state
        is_terminal_hour = self.hour == 23
        quota_left = sum(sum(item.values()) for item in self.remaining_ch.values())
        quota_left += sum(sum(item.values()) for item in self.remaining_ps_generation.values())
        pumping_left = sum(sum(item.values()) for item in self.remaining_ps_pumping.values())
        target_storage_error = sum(
            abs(self.state.storage_m3[key] - target)
            for key, target in self.plan.target_storage_m3.items()
        ) / max(sum(item.max_storage_m3 - item.min_storage_m3 for item in self.config.reservoirs), 1.0)
        signals = {
            "g_segment_quota": quota_left / 1000.0 if is_terminal_hour else 0.0,
            "g_terminal_storage": target_storage_error if is_terminal_hour else 0.0,
            "g_purchased": result.purchased_mwh / 1000.0,
            "g_curtailment": result.curtailed_mwh / 1000.0,
            "g_purchased_share": purchased_share,
            "g_curtailment_share": _safe_ratio(result.curtailed_mwh, vre),
            "g_pumping_quota": pumping_left / 1000.0 if is_terminal_hour else 0.0,
        }
        penalty = sum(self.lambdas[name] * signals[f"g_{name}"] for name in self.lambdas)
        reward = -(
            result.purchased_mwh + result.curtailed_mwh + 0.25 * target_error
        ) / max(load, 1.0) - penalty
        info = {
            **signals,
            "day_index": self.day_index,
            "hour": self.hour,
            "segment": segment,
            "load_mwh": load,
            "wind_mwh": wind,
            "solar_mwh": solar,
            "purchased_mwh": result.purchased_mwh,
            "curtailed_mwh": result.curtailed_mwh,
            "ch_generation_mwh": dict(result.ch_generation_mwh),
            "ps_generation_mwh": dict(result.ps_generation_mwh),
            "ps_pumping_mwh": dict(result.ps_pumping_mwh),
            "storage_factor": storage_factor,
            "water_value": self.water_state.value,
            "remaining_pumping_mwh": pumping_left,
            "upper_storage_factors": {
                plant.id: (
                    self.state.storage_m3[plant.upper_reservoir_id]
                    - next(
                        item.min_storage_m3
                        for item in self.config.reservoirs
                        if item.id == plant.upper_reservoir_id
                    )
                )
                / next(
                    item.max_storage_m3 - item.min_storage_m3
                    for item in self.config.reservoirs
                    if item.id == plant.upper_reservoir_id
                )
                for plant in self.config.ps_plants
            },
            "upper_level_margins_m": {
                plant.id: (
                    self.level_storage_curves[plant.upper_reservoir_id].levels_m[-1]
                    - self.level_storage_curves[plant.upper_reservoir_id].level(
                        self.state.storage_m3[plant.upper_reservoir_id]
                    )
                )
                for plant in self.config.ps_plants
            },
            "spill_m3": dict(result.spill_m3),
        }
        self.hour += 1
        terminated = self.hour >= 24
        observation = np.zeros(self._obs_size, dtype=np.float32) if terminated else self._observation()
        return observation, float(reward), terminated, False, info
