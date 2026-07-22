from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


@dataclass(frozen=True)
class ReservoirConfig:
    id: str
    min_storage_m3: float
    max_storage_m3: float
    initial_storage_m3: float
    level_storage_nodes: tuple[tuple[float, float], ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReservoirConfig":
        nodes = tuple((float(level), float(storage)) for level, storage in raw["level_storage_nodes"])
        item = cls(
            id=str(_require(raw, "id")),
            min_storage_m3=float(_require(raw, "min_storage_m3")),
            max_storage_m3=float(_require(raw, "max_storage_m3")),
            initial_storage_m3=float(_require(raw, "initial_storage_m3")),
            level_storage_nodes=nodes,
        )
        item.validate()
        return item

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Reservoir IDs cannot be empty")
        if not self.min_storage_m3 < self.max_storage_m3:
            raise ValueError(f"Invalid storage bounds for {self.id}")
        if not self.min_storage_m3 <= self.initial_storage_m3 <= self.max_storage_m3:
            raise ValueError(f"Initial storage is outside the bounds for {self.id}")
        if len(self.level_storage_nodes) < 2:
            raise ValueError(f"At least two level-storage nodes are required for {self.id}")
        levels, storage = zip(*self.level_storage_nodes)
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError(f"Levels must be strictly increasing for {self.id}")
        if any(b <= a for a, b in zip(storage, storage[1:])):
            raise ValueError(f"Storage values must be strictly increasing for {self.id}")


@dataclass(frozen=True)
class CHPlantConfig:
    id: str
    reservoir_id: str
    downstream_reservoir_id: str | None
    max_power_mw: float
    efficiency: float
    effective_head_m: float
    minimum_release_m3s: float = 0.0
    phq_terms: tuple[tuple[int, int, float], ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CHPlantConfig":
        return cls(
            id=str(_require(raw, "id")),
            reservoir_id=str(_require(raw, "reservoir_id")),
            downstream_reservoir_id=(
                None
                if raw.get("downstream_reservoir_id") is None
                else str(raw["downstream_reservoir_id"])
            ),
            max_power_mw=float(_require(raw, "max_power_mw")),
            efficiency=float(_require(raw, "efficiency")),
            effective_head_m=float(_require(raw, "effective_head_m")),
            minimum_release_m3s=float(raw.get("minimum_release_m3s", 0.0)),
            phq_terms=tuple(
                (int(i), int(j), float(beta)) for i, j, beta in raw.get("phq_terms", [])
            ),
        )


@dataclass(frozen=True)
class PSPlantConfig:
    id: str
    upper_reservoir_id: str
    lower_reservoir_id: str
    max_generation_mw: float
    max_pumping_mw: float
    generation_efficiency: float
    pumping_efficiency: float
    effective_head_m: float
    generation_phq_terms: tuple[tuple[int, int, float], ...] = ()
    pumping_phq_terms: tuple[tuple[int, int, float], ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PSPlantConfig":
        terms = lambda key: tuple(  # noqa: E731
            (int(i), int(j), float(beta)) for i, j, beta in raw.get(key, [])
        )
        return cls(
            id=str(_require(raw, "id")),
            upper_reservoir_id=str(_require(raw, "upper_reservoir_id")),
            lower_reservoir_id=str(_require(raw, "lower_reservoir_id")),
            max_generation_mw=float(_require(raw, "max_generation_mw")),
            max_pumping_mw=float(_require(raw, "max_pumping_mw")),
            generation_efficiency=float(_require(raw, "generation_efficiency")),
            pumping_efficiency=float(_require(raw, "pumping_efficiency")),
            effective_head_m=float(_require(raw, "effective_head_m")),
            generation_phq_terms=terms("generation_phq_terms"),
            pumping_phq_terms=terms("pumping_phq_terms"),
        )


@dataclass(frozen=True)
class ANBConfig:
    weight_steepness: float = 8.0
    neutral_storage: float = 0.70
    storage_curvature: float = 6.0
    water_value_correction: float = 0.10
    short_term_strength: float = 10.0
    asymmetry: float = 0.50
    weight_min: float = 0.05
    weight_max: float = 0.95

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ANBConfig":
        return cls(**{key: float(value) for key, value in raw.items()})

    def validate(self) -> None:
        if not 0.0 <= self.weight_min < self.weight_max <= 1.0:
            raise ValueError("ANB weight bounds must satisfy 0 <= min < max <= 1")
        if not 0.0 <= self.neutral_storage <= 1.0:
            raise ValueError("neutral_storage must be in [0, 1]")
        if not 0.0 <= self.asymmetry < 1.0:
            raise ValueError("asymmetry must be in [0, 1)")


@dataclass(frozen=True)
class WaterValueConfig:
    shortage_weight: float = 1.0
    surplus_weight: float = 0.7
    baseline_rate: float = 0.02
    smoothing_rate: float = 0.20
    noise_threshold: float = 0.01
    signal_gain: float = 2.0
    lower_bound: float = -1.0
    upper_bound: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WaterValueConfig":
        return cls(**{key: float(value) for key, value in raw.items()})

    def validate(self) -> None:
        if not 0.0 < self.baseline_rate < 1.0:
            raise ValueError("baseline_rate must be in (0, 1)")
        if not 0.0 < self.smoothing_rate < 1.0:
            raise ValueError("smoothing_rate must be in (0, 1)")
        if self.noise_threshold < 0.0:
            raise ValueError("noise_threshold cannot be negative")
        if not self.lower_bound < self.upper_bound:
            raise ValueError("Invalid water-value bounds")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.995
    tau: float = 0.02
    long_steps: int = 200_000
    short_steps: int = 200_000

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainingConfig":
        return cls(
            learning_rate=float(raw.get("learning_rate", 3e-4)),
            buffer_size=int(raw.get("buffer_size", 100_000)),
            batch_size=int(raw.get("batch_size", 256)),
            gamma=float(raw.get("gamma", 0.995)),
            tau=float(raw.get("tau", 0.02)),
            long_steps=int(raw.get("long_steps", 200_000)),
            short_steps=int(raw.get("short_steps", 200_000)),
        )


@dataclass(frozen=True)
class SchedulerConfig:
    seed: int
    segments: tuple[str, ...]
    segment_hours: dict[str, int]
    reservoirs: tuple[ReservoirConfig, ...]
    ch_plants: tuple[CHPlantConfig, ...]
    ps_plants: tuple[PSPlantConfig, ...]
    anb: ANBConfig = field(default_factory=ANBConfig)
    water_value: WaterValueConfig = field(default_factory=WaterValueConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rolling_horizon: int = 6
    candidate_count: int = 6

    @classmethod
    def load(cls, path: str | Path) -> "SchedulerConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(
            seed=int(raw.get("seed", 42)),
            segments=tuple(str(item) for item in raw.get("segments", ["peak", "flat", "valley"])),
            segment_hours={str(key): int(value) for key, value in raw["segment_hours"].items()},
            reservoirs=tuple(ReservoirConfig.from_dict(item) for item in raw["reservoirs"]),
            ch_plants=tuple(CHPlantConfig.from_dict(item) for item in raw["ch_plants"]),
            ps_plants=tuple(PSPlantConfig.from_dict(item) for item in raw["ps_plants"]),
            anb=ANBConfig.from_dict(raw.get("anb", {})),
            water_value=WaterValueConfig.from_dict(raw.get("water_value", {})),
            training=TrainingConfig.from_dict(raw.get("training", {})),
            rolling_horizon=int(raw.get("rolling_horizon", 6)),
            candidate_count=int(raw.get("candidate_count", 6)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        reservoir_ids = [item.id for item in self.reservoirs]
        unit_ids = [item.id for item in self.ch_plants] + [item.id for item in self.ps_plants]
        if len(reservoir_ids) != len(set(reservoir_ids)):
            raise ValueError("Reservoir IDs must be unique")
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Plant IDs must be unique")
        known = set(reservoir_ids)
        for plant in self.ch_plants:
            if plant.reservoir_id not in known:
                raise ValueError(f"Unknown reservoir for {plant.id}: {plant.reservoir_id}")
            if plant.downstream_reservoir_id and plant.downstream_reservoir_id not in known:
                raise ValueError(
                    f"Unknown downstream reservoir for {plant.id}: {plant.downstream_reservoir_id}"
                )
            if plant.max_power_mw <= 0 or plant.effective_head_m <= 0:
                raise ValueError(f"Power and head must be positive for {plant.id}")
            if not 0.0 < plant.efficiency <= 1.0:
                raise ValueError(f"Invalid efficiency for {plant.id}")
        for plant in self.ps_plants:
            if plant.upper_reservoir_id not in known or plant.lower_reservoir_id not in known:
                raise ValueError(f"Unknown PS reservoir for {plant.id}")
            if not 0.0 < plant.generation_efficiency <= 1.0:
                raise ValueError(f"Invalid generation efficiency for {plant.id}")
            if not 0.0 < plant.pumping_efficiency <= 1.0:
                raise ValueError(f"Invalid pumping efficiency for {plant.id}")
        if set(self.segment_hours) != set(self.segments):
            raise ValueError("segment_hours must define every segment exactly once")
        if sum(self.segment_hours.values()) != 24:
            raise ValueError("The segment hours must sum to 24")
        if self.rolling_horizon < 1:
            raise ValueError("Rolling horizon must be positive")
        if not 1 <= self.candidate_count <= 6:
            raise ValueError("candidate_count must be between 1 and 6")
        self.anb.validate()
        self.water_value.validate()

    @property
    def long_action_size(self) -> int:
        return len(self.ch_plants) + 2 * len(self.ps_plants) * len(self.segments)

    @property
    def short_action_size(self) -> int:
        return len(self.ch_plants) + 2 * len(self.ps_plants)
