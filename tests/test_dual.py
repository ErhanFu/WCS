from chps_scheduler.dual import DualSignal, EpisodeMeanPIDLagrange


def test_dual_update_uses_episode_mean_not_last_step():
    controller = EpisodeMeanPIDLagrange(
        signals=(DualSignal("constraint", "g", 0.1, 0.0, 100.0, tolerance=1.0),),
        kp=1.0,
        ki=0.0,
        kd=0.0,
        decay=1.0,
    )
    controller.observe({"g": 0.0})
    controller.observe({"g": 4.0})
    result = controller.end_episode({"constraint": 2.0})
    assert result.means["constraint"] == 2.0
    assert result.residuals["constraint"] == 1.0
    assert result.multipliers["constraint"] == 2.1

