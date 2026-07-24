from pathlib import Path

import numpy as np

from chps_scheduler.config import SchedulerConfig
from chps_scheduler.coordinator import TwoLayerCoordinator
from chps_scheduler.data import synthetic_case
from chps_scheduler.environments import ShortTermEnv
from chps_scheduler.physics import project_ps_mode


ROOT = Path(__file__).resolve().parents[1]


def test_ps_projection_is_mutually_exclusive():
    generation, pumping = project_ps_mode(0.8, 0.3)
    assert generation > 0.0
    assert pumping == 0.0
    generation, pumping = project_ps_mode(0.2, 0.7)
    assert generation == 0.0
    assert pumping > 0.0


def test_two_layer_smoke_dispatch_balances_energy():
    config = SchedulerConfig.load(ROOT / "configs" / "synthetic_case.json")
    data = synthetic_case(config, days=2)
    coordinator = TwoLayerCoordinator(config, data)
    result = coordinator.dispatch(rolling=True)
    assert len(result) == 48
    assert result["balance_error_mwh"].abs().max() < 1e-8
    assert (result["purchased_mwh"] >= 0.0).all()
    assert (result["curtailed_mwh"] >= 0.0).all()


def test_short_environment_training_reset_starts_from_the_first_plan():
    config = SchedulerConfig.load(ROOT / "configs" / "synthetic_case.json")
    data = synthetic_case(config, days=2)
    plans = TwoLayerCoordinator(config, data).build_plans()
    environment = ShortTermEnv(config, data, plans)

    _, info = environment.reset()

    assert info["day_index"] == 0


def test_realized_short_term_state_is_returned_to_the_next_daily_plan():
    config = SchedulerConfig.load(ROOT / "configs" / "synthetic_case.json")
    data = synthetic_case(config, days=2)
    coordinator = TwoLayerCoordinator(config, data)
    long_observations = []

    def long_policy(observation):
        long_observations.append(np.asarray(observation, dtype=float).copy())
        return np.zeros(config.long_action_size, dtype=np.float32)

    result = coordinator.dispatch(long_policy=long_policy, rolling=False)

    assert len(long_observations) == 2
    day_one_end = result.loc[(result["day_index"] == 0) & (result["hour"] == 23)].iloc[0]
    expected_storage = []
    for reservoir in config.reservoirs:
        storage = day_one_end[f"storage_{reservoir.id}_m3"]
        span = reservoir.max_storage_m3 - reservoir.min_storage_m3
        expected_storage.append(2.0 * (storage - reservoir.min_storage_m3) / span - 1.0)

    np.testing.assert_allclose(
        long_observations[1][: len(config.reservoirs)],
        expected_storage,
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isclose(
        long_observations[1][-2],
        day_one_end["water_value"],
        rtol=1e-6,
        atol=1e-6,
    )
