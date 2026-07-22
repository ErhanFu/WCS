from pathlib import Path

from chps_scheduler.config import SchedulerConfig
from chps_scheduler.signals import ANBCoordinator, WaterValueModel, WaterValueState


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_config_is_valid():
    config = SchedulerConfig.load(ROOT / "configs" / "synthetic_case.json")
    assert config.long_action_size == 8
    assert config.short_action_size == 4


def test_water_value_and_reference_move_in_expected_direction():
    config = SchedulerConfig.load(ROOT / "configs" / "synthetic_case.json")
    water_model = WaterValueModel(config.water_value)
    shortage = water_model.update(WaterValueState(), purchased_share=0.5, pre_storage_surplus_share=0.0)
    surplus = water_model.update(WaterValueState(), purchased_share=0.0, pre_storage_surplus_share=0.5)
    assert shortage.state.value > 0.0
    assert surplus.state.value < 0.0

    coordinator = ANBCoordinator(config.anb)
    low = coordinator.coordinate(100.0, 20.0, 0.0, 120.0, -0.5, 0.5)
    high = coordinator.coordinate(100.0, 20.0, 0.0, 120.0, 0.5, 0.5)
    assert high.corrected_reference_mwh > low.corrected_reference_mwh

