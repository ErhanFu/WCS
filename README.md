# CH-PS Two-Layer Scheduler

This repository contains a privacy-safe implementation of a two-layer scheduling
framework for cascade hydropower (CH), pumped storage (PS), and variable renewable
energy (VRE). It combines the original daily and hourly implementations behind one
configuration model and one command-line entry point.

The public tree contains no site names, engineering records, historical profiles,
absolute paths, trained weights, or machine-specific metadata. The bundled case is
synthetic and exists only for smoke testing.

## Included mechanisms

- Daily resource allocation and hourly executable dispatch.
- Water-state-aware asymmetric Nash bargaining (ANB).
- Dynamic water-value signal driven by shortage and pre-storage VRE surplus pressure.
- Episode-mean PID-Lagrangian dual updates.
- Rolling-horizon candidate evaluation with first-action execution.
- Generic reservoir, CH, PS, power-balance, and inter-layer plan interfaces.

## Install and run

```bash
python -m venv .venv
python -m pip install -e ".[rl,dev]"
chps-scheduler validate --config configs/synthetic_case.json
chps-scheduler demo --config configs/synthetic_case.json --output runs/demo
pytest
```

The demo uses a deterministic policy and does not require training. SAC training is
available through `chps-scheduler train` when the `rl` extra is installed.

## Private data interface

Keep confidential files outside the repository or under `data/private/`, which is
ignored by Git. A private daily CSV uses the following generic columns:

```text
timestamp,load_mwh,wind_mwh,solar_mwh,inflow_<reservoir_id>_m3s,...
```

The hourly CSV uses the same schema at hourly resolution. Reservoir and unit IDs must
match a private configuration derived from `configs/synthetic_case.json`. The bundled
synthetic example uses generic level-storage curves and hydraulic conversion
parameters; station-specific operating records and engineering curves are not
included.

## Important implementation correction

Dual variables are updated from the mean constraint signal over the complete episode.
The legacy scripts updated daily-layer multipliers from only the final step of a yearly
episode. The unified implementation accumulates every step and applies the configured
tolerance after aggregation, matching the stated episode-level formulation.

## Publication safety

Run the repository scanner before every public push:

```bash
python scripts/privacy_scan.py .
```

Also complete [PUBLICATION_CHECKLIST.md](PUBLICATION_CHECKLIST.md). The scanner is a
guardrail, not a substitute for a human review of the Git history and release assets.
