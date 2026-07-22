from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HydraulicState:
    storage_m3: dict[str, float]

    def copy(self) -> "HydraulicState":
        return HydraulicState(storage_m3=dict(self.storage_m3))


@dataclass(frozen=True)
class InterLayerPlan:
    day_index: int
    ch_quota_mwh: dict[str, dict[str, float]]
    ps_generation_quota_mwh: dict[str, dict[str, float]]
    ps_pumping_quota_mwh: dict[str, dict[str, float]]
    target_storage_m3: dict[str, float]
    water_value: float
    storage_factor: float
    long_term_weight: float
    short_term_weight: float
    coordinated_target_mwh: float


@dataclass(frozen=True)
class DispatchResult:
    ch_generation_mwh: dict[str, float]
    ps_generation_mwh: dict[str, float]
    ps_pumping_mwh: dict[str, float]
    purchased_mwh: float
    curtailed_mwh: float
    spill_m3: dict[str, float] = field(default_factory=dict)

    @property
    def controllable_generation_mwh(self) -> float:
        return sum(self.ch_generation_mwh.values()) + sum(self.ps_generation_mwh.values())

