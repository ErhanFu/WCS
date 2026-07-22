from __future__ import annotations

from pathlib import Path

from .config import SchedulerConfig
from .coordinator import TwoLayerCoordinator
from .data import CaseData
from .dual import DualSignal, EpisodeMeanPIDLagrange, make_sb3_callback
from .environments import LongTermEnv, ShortTermEnv


def _long_dual_controller() -> EpisodeMeanPIDLagrange:
    return EpisodeMeanPIDLagrange(
        signals=(
            DualSignal("target", "g_target", 1e-2, 1e-2, 2e4),
            DualSignal("curtailment", "g_curtailment", 2e-3, 1e-5, 20.0),
            DualSignal("purchased", "g_purchased", 5e-3, 1e-5, 50.0, tolerance=0.2),
            DualSignal("curtailment_share", "g_curtailment_share", 1e-3, 1e-5, 150.0),
            DualSignal("purchased_share", "g_purchased_share", 1e-3, 1e-5, 150.0),
        )
    )


def _short_dual_controller() -> EpisodeMeanPIDLagrange:
    return EpisodeMeanPIDLagrange(
        signals=(
            DualSignal("segment_quota", "g_segment_quota", 1e-3, 1e-8, 1e2),
            DualSignal("terminal_storage", "g_terminal_storage", 1e-4, 1e-6, 1.0),
            DualSignal("purchased", "g_purchased", 1e-9, 1e-12, 1e-2),
            DualSignal("curtailment", "g_curtailment", 1e-9, 1e-12, 1e-2),
            DualSignal("purchased_share", "g_purchased_share", 1e-3, 1e-6, 1e2),
            DualSignal("curtailment_share", "g_curtailment_share", 1e-3, 1e-6, 1e2),
            DualSignal("pumping_quota", "g_pumping_quota", 1e-7, 1e-12, 1e-2),
        )
    )


def _sac_kwargs(config: SchedulerConfig) -> dict:
    training = config.training
    return {
        "policy": "MlpPolicy",
        "learning_rate": training.learning_rate,
        "buffer_size": training.buffer_size,
        "batch_size": training.batch_size,
        "gamma": training.gamma,
        "tau": training.tau,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "ent_coef": "auto",
        "seed": config.seed,
        "verbose": 1,
    }


def train_two_layer(
    config: SchedulerConfig,
    data: CaseData,
    output_dir: str | Path,
) -> dict[str, Path]:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the project with the 'rl' extra to train SAC models") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    long_env = LongTermEnv(config, data)
    long_vec = DummyVecEnv([lambda: Monitor(long_env)])
    long_vec = VecNormalize(long_vec, norm_obs=True, norm_reward=True, gamma=config.training.gamma)
    long_model = SAC(env=long_vec, **_sac_kwargs(config))
    long_callback = make_sb3_callback(_long_dual_controller())
    long_model.learn(total_timesteps=config.training.long_steps, callback=long_callback)
    long_model_path = output / "long_term_sac"
    long_norm_path = output / "long_term_vecnormalize.pkl"
    long_model.save(long_model_path)
    long_vec.save(long_norm_path)

    # Build inter-layer plans from the trained daily policy.
    long_vec.training = False
    long_vec.norm_reward = False
    observation = long_vec.reset()
    plans = []
    finished = False
    while not finished:
        action, _ = long_model.predict(observation, deterministic=True)
        observation, _, dones, infos = long_vec.step(action)
        plans.append(infos[0]["plan"])
        finished = bool(dones[0])

    short_env = ShortTermEnv(config, data, plans)
    short_vec = DummyVecEnv([lambda: Monitor(short_env)])
    short_vec = VecNormalize(short_vec, norm_obs=True, norm_reward=True, gamma=config.training.gamma)
    short_model = SAC(env=short_vec, **_sac_kwargs(config))
    short_callback = make_sb3_callback(_short_dual_controller())
    short_model.learn(total_timesteps=config.training.short_steps, callback=short_callback)
    short_model_path = output / "short_term_sac"
    short_norm_path = output / "short_term_vecnormalize.pkl"
    short_model.save(short_model_path)
    short_vec.save(short_norm_path)

    return {
        "long_model": long_model_path.with_suffix(".zip"),
        "long_normalization": long_norm_path,
        "short_model": short_model_path.with_suffix(".zip"),
        "short_normalization": short_norm_path,
    }


def heuristic_dispatch(config: SchedulerConfig, data: CaseData):
    coordinator = TwoLayerCoordinator(config, data)
    plans = coordinator.build_plans()
    return coordinator.dispatch(plans=plans, rolling=True)

