from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import SchedulerConfig
from .curves import generation_release_m3, pumping_transfer_m3
from .models import DispatchResult, HydraulicState


SECONDS_PER_HOUR = 3600.0


def project_ps_mode(generation_share: float, pumping_share: float) -> tuple[float, float]:
    """Project a PS action onto mutually exclusive generation and pumping modes."""

    gen = float(np.clip(generation_share, 0.0, 1.0))
    pump = float(np.clip(pumping_share, 0.0, 1.0))
    if gen >= pump:
        return gen - pump, 0.0
    return 0.0, pump - gen


@dataclass
class HydraulicModel:
    config: SchedulerConfig

    def initial_state(self) -> HydraulicState:
        return HydraulicState(
            storage_m3={item.id: item.initial_storage_m3 for item in self.config.reservoirs}
        )

    def storage_factor(self, state: HydraulicState) -> float:
        factors = []
        for reservoir in self.config.reservoirs:
            span = reservoir.max_storage_m3 - reservoir.min_storage_m3
            factors.append((state.storage_m3[reservoir.id] - reservoir.min_storage_m3) / span)
        return float(np.clip(np.mean(factors), 0.0, 1.0))

    def dispatch(
        self,
        state: HydraulicState,
        duration_hours: float,
        load_mwh: float,
        wind_mwh: float,
        solar_mwh: float,
        inflow_m3s: Mapping[str, float],
        requested_ch_mwh: Mapping[str, float],
        requested_ps_generation_mwh: Mapping[str, float],
        requested_ps_pumping_mwh: Mapping[str, float],
    ) -> DispatchResult:
        if duration_hours <= 0.0:
            raise ValueError("duration_hours must be positive")
        reservoirs = {item.id: item for item in self.config.reservoirs}
        for reservoir in self.config.reservoirs:
            inflow = max(0.0, float(inflow_m3s.get(reservoir.id, 0.0)))
            state.storage_m3[reservoir.id] += inflow * duration_hours * SECONDS_PER_HOUR

        actual_ch: dict[str, float] = {}
        for plant in self.config.ch_plants:
            reservoir = reservoirs[plant.reservoir_id]
            requested = float(np.clip(requested_ch_mwh.get(plant.id, 0.0), 0.0, plant.max_power_mw * duration_hours))
            requested_release = generation_release_m3(
                requested, plant.effective_head_m, plant.efficiency
            )
            ecological_release = plant.minimum_release_m3s * duration_hours * SECONDS_PER_HOUR
            available = max(0.0, state.storage_m3[reservoir.id] - reservoir.min_storage_m3)
            total_release = min(available, max(requested_release, ecological_release))
            energy_fraction = 0.0 if requested_release <= 0.0 else min(1.0, total_release / requested_release)
            actual = requested * energy_fraction
            state.storage_m3[reservoir.id] -= total_release
            if plant.downstream_reservoir_id:
                state.storage_m3[plant.downstream_reservoir_id] += total_release
            actual_ch[plant.id] = actual

        actual_ps_generation: dict[str, float] = {}
        actual_ps_pumping: dict[str, float] = {}
        for plant in self.config.ps_plants:
            upper = reservoirs[plant.upper_reservoir_id]
            lower = reservoirs[plant.lower_reservoir_id]
            requested_gen = float(
                np.clip(
                    requested_ps_generation_mwh.get(plant.id, 0.0),
                    0.0,
                    plant.max_generation_mw * duration_hours,
                )
            )
            requested_pump = float(
                np.clip(
                    requested_ps_pumping_mwh.get(plant.id, 0.0),
                    0.0,
                    plant.max_pumping_mw * duration_hours,
                )
            )
            gen_share, pump_share = project_ps_mode(
                requested_gen / max(plant.max_generation_mw * duration_hours, 1e-9),
                requested_pump / max(plant.max_pumping_mw * duration_hours, 1e-9),
            )
            requested_gen = gen_share * plant.max_generation_mw * duration_hours
            requested_pump = pump_share * plant.max_pumping_mw * duration_hours

            if requested_gen > 0.0:
                volume = generation_release_m3(
                    requested_gen, plant.effective_head_m, plant.generation_efficiency
                )
                available = max(0.0, state.storage_m3[upper.id] - upper.min_storage_m3)
                volume = min(volume, available, lower.max_storage_m3 - state.storage_m3[lower.id])
                fraction = volume / max(
                    generation_release_m3(
                        requested_gen, plant.effective_head_m, plant.generation_efficiency
                    ),
                    1e-9,
                )
                actual_gen = requested_gen * fraction
                state.storage_m3[upper.id] -= volume
                state.storage_m3[lower.id] += volume
                actual_pump = 0.0
            elif requested_pump > 0.0:
                volume = pumping_transfer_m3(
                    requested_pump, plant.effective_head_m, plant.pumping_efficiency
                )
                available = max(0.0, state.storage_m3[lower.id] - lower.min_storage_m3)
                volume = min(volume, available, upper.max_storage_m3 - state.storage_m3[upper.id])
                fraction = volume / max(
                    pumping_transfer_m3(
                        requested_pump, plant.effective_head_m, plant.pumping_efficiency
                    ),
                    1e-9,
                )
                actual_pump = requested_pump * fraction
                state.storage_m3[lower.id] -= volume
                state.storage_m3[upper.id] += volume
                actual_gen = 0.0
            else:
                actual_gen = 0.0
                actual_pump = 0.0
            actual_ps_generation[plant.id] = actual_gen
            actual_ps_pumping[plant.id] = actual_pump

        spill: dict[str, float] = {}
        for reservoir in self.config.reservoirs:
            excess = max(0.0, state.storage_m3[reservoir.id] - reservoir.max_storage_m3)
            if excess:
                state.storage_m3[reservoir.id] -= excess
            state.storage_m3[reservoir.id] = float(
                np.clip(
                    state.storage_m3[reservoir.id],
                    reservoir.min_storage_m3,
                    reservoir.max_storage_m3,
                )
            )
            spill[reservoir.id] = excess

        supply = wind_mwh + solar_mwh + sum(actual_ch.values()) + sum(actual_ps_generation.values())
        demand = load_mwh + sum(actual_ps_pumping.values())
        purchased = max(0.0, demand - supply)
        curtailed = max(0.0, supply - demand)
        return DispatchResult(
            ch_generation_mwh=actual_ch,
            ps_generation_mwh=actual_ps_generation,
            ps_pumping_mwh=actual_ps_pumping,
            purchased_mwh=purchased,
            curtailed_mwh=curtailed,
            spill_m3=spill,
        )

