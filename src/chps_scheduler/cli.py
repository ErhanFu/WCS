from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SchedulerConfig
from .coordinator import TwoLayerCoordinator
from .data import CaseData, synthetic_case
from .training import train_two_layer


def _case_data(args, config: SchedulerConfig) -> CaseData:
    if args.daily or args.hourly:
        if not args.daily or not args.hourly:
            raise ValueError("Both --daily and --hourly are required for private inputs")
        return CaseData.from_csv(args.daily, args.hourly, config)
    return synthetic_case(config, days=args.days)


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--daily", type=Path, help="Private daily input CSV")
    parser.add_argument("--hourly", type=Path, help="Private hourly input CSV")
    parser.add_argument("--days", type=int, default=14, help="Synthetic demo length")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chps-scheduler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate configuration and inputs")
    validate.add_argument("--config", type=Path, required=True)
    _add_data_arguments(validate)

    demo = subparsers.add_parser("demo", help="Run the anonymous deterministic demo")
    demo.add_argument("--config", type=Path, required=True)
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--no-rolling", action="store_true")
    _add_data_arguments(demo)

    train = subparsers.add_parser("train", help="Train daily and hourly SAC policies")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    _add_data_arguments(train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SchedulerConfig.load(args.config)
    data = _case_data(args, config)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "status": "ok",
                    "daily_rows": len(data.daily),
                    "hourly_rows": len(data.hourly),
                    "long_action_size": config.long_action_size,
                    "short_action_size": config.short_action_size,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "demo":
        coordinator = TwoLayerCoordinator(config, data)
        plans = coordinator.build_plans()
        dispatch = coordinator.dispatch(plans=plans, rolling=not args.no_rolling)
        args.output.mkdir(parents=True, exist_ok=True)
        dispatch.to_csv(args.output / "dispatch.csv", index=False)
        summary = {
            "hours": len(dispatch),
            "purchased_mwh": float(dispatch["purchased_mwh"].sum()),
            "curtailed_mwh": float(dispatch["curtailed_mwh"].sum()),
            "maximum_balance_error_mwh": float(dispatch["balance_error_mwh"].abs().max()),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "train":
        paths = train_two_layer(config, data, args.output)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

