from pathlib import Path

from chps_scheduler.config import SchedulerConfig
from chps_scheduler.coordinator import TwoLayerCoordinator
from chps_scheduler.data import synthetic_case
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
    plans = coordinator.build_plans()
    result = coordinator.dispatch(plans=plans, rolling=True)
    assert len(result) == 48
    assert result["balance_error_mwh"].abs().max() < 1e-8
    assert (result["purchased_mwh"] >= 0.0).all()
    assert (result["curtailed_mwh"] >= 0.0).all()

