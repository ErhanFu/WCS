from pathlib import Path

import numpy as np

from chps_scheduler.config import SchedulerConfig
from chps_scheduler.rolling import RollingHorizonEvaluator


ROOT = Path(__file__).resolve().parents[1]


def _info(**overrides):
    values = {
        "segment": "flat",
        "hour": 12,
        "load_mwh": 100.0,
        "purchased_mwh": 0.0,
        "curtailed_mwh": 0.0,
        "remaining_pumping_mwh": 0.0,
        "storage_factor": 0.50,
        "upper_level_margins_m": {"ps_01": 1.0},
        "spill_m3": {"ps_upper_01": 0.0},
    }
    values.update(overrides)
    return values


def _evaluator():
    config = SchedulerConfig.load(ROOT / "configs" / "synthetic_case.json")
    return RollingHorizonEvaluator(config)


def test_peak_shortage_rule_matches_manuscript_table():
    evaluator = _evaluator()
    baseline = np.array([[0.20, 0.30, 0.40, 0.50]])
    variants = evaluator._candidate_variants(
        baseline,
        [_info(segment="peak", purchased_mwh=10.0)],
    )

    np.testing.assert_allclose(variants[2][0], [0.40, 0.50, 0.40, 0.25])


def test_upper_reservoir_rule_uses_half_meter_margin():
    evaluator = _evaluator()
    baseline = np.array([[0.20, 0.30, 0.40, 0.50]])
    variants = evaluator._candidate_variants(
        baseline,
        [_info(upper_level_margins_m={"ps_01": 0.49})],
    )

    np.testing.assert_allclose(variants[4][0], [0.20, 0.30, 0.60, 0.35])


def test_high_storage_pumping_rule_is_not_limited_to_peak_hours():
    evaluator = _evaluator()
    baseline = np.array([[0.20, 0.30, 0.40, 0.50]])
    variants = evaluator._candidate_variants(
        baseline,
        [_info(segment="flat", storage_factor=0.81)],
    )

    np.testing.assert_allclose(variants[5][0], [0.20, 0.30, 0.40, 0.30])
