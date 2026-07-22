from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SchedulerConfig


BASE_COLUMNS = ("timestamp", "load_mwh", "wind_mwh", "solar_mwh")


@dataclass(frozen=True)
class CaseData:
    daily: pd.DataFrame
    hourly: pd.DataFrame

    @classmethod
    def from_csv(
        cls,
        daily_path: str | Path,
        hourly_path: str | Path,
        config: SchedulerConfig,
    ) -> "CaseData":
        daily = pd.read_csv(daily_path)
        hourly = pd.read_csv(hourly_path)
        daily["timestamp"] = pd.to_datetime(daily["timestamp"])
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
        item = cls(daily=daily, hourly=hourly)
        item.validate(config)
        return item

    def validate(self, config: SchedulerConfig) -> None:
        expected = set(BASE_COLUMNS)
        expected.update(f"inflow_{item.id}_m3s" for item in config.reservoirs)
        for name, frame in (("daily", self.daily), ("hourly", self.hourly)):
            missing = expected.difference(frame.columns)
            if missing:
                raise ValueError(f"Missing {name} input columns: {sorted(missing)}")
            numeric = frame[list(expected.difference({"timestamp"}))].to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise ValueError(f"{name} input contains non-finite values")
            if (frame[["load_mwh", "wind_mwh", "solar_mwh"]] < 0.0).any().any():
                raise ValueError(f"{name} energy columns cannot be negative")
        if len(self.hourly) != 24 * len(self.daily):
            raise ValueError("Hourly inputs must contain exactly 24 rows per daily input row")
        if not self.daily["timestamp"].is_monotonic_increasing:
            raise ValueError("Daily timestamps must be ordered")
        if not self.hourly["timestamp"].is_monotonic_increasing:
            raise ValueError("Hourly timestamps must be ordered")


def synthetic_case(config: SchedulerConfig, days: int = 14, seed: int | None = None) -> CaseData:
    """Generate an anonymous deterministic case for smoke tests and examples."""

    if days < 2:
        raise ValueError("Synthetic cases require at least two days")
    rng = np.random.default_rng(config.seed if seed is None else seed)
    hours = days * 24
    timestamps = pd.date_range("2030-01-01", periods=hours, freq="h")
    hour = np.arange(hours)
    daily_cycle = 1.0 + 0.18 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0)
    weekly_cycle = 1.0 + 0.05 * np.sin(2.0 * np.pi * hour / (24.0 * 7.0))
    load = 520.0 * daily_cycle * weekly_cycle + rng.normal(0.0, 8.0, hours)
    solar = 260.0 * np.maximum(0.0, np.sin(np.pi * ((hour % 24) - 6.0) / 12.0))
    wind = 125.0 * (1.0 + 0.25 * np.sin(2.0 * np.pi * hour / 37.0))
    hourly = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mwh": np.maximum(load, 0.0),
            "wind_mwh": np.maximum(wind, 0.0),
            "solar_mwh": np.maximum(solar, 0.0),
        }
    )
    for index, reservoir in enumerate(config.reservoirs):
        profile = 18.0 + 4.0 * np.sin(2.0 * np.pi * hour / (24.0 * (5.0 + index)))
        hourly[f"inflow_{reservoir.id}_m3s"] = np.maximum(profile, 0.0)

    aggregation = {
        "load_mwh": "sum",
        "wind_mwh": "sum",
        "solar_mwh": "sum",
        **{f"inflow_{item.id}_m3s": "mean" for item in config.reservoirs},
    }
    daily = (
        hourly.set_index("timestamp")
        .resample("D")
        .agg(aggregation)
        .reset_index()
    )
    case = CaseData(daily=daily, hourly=hourly)
    case.validate(config)
    return case

