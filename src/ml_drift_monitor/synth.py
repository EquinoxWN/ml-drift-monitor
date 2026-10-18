"""Synthetic credit-application data with optional, known drift, for demos and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import write_csv

SCHEMA = [
    {"name": "age", "kind": "numeric"},
    {"name": "income", "kind": "numeric"},
    {"name": "credit_score", "kind": "numeric"},
    {"name": "open_accounts", "kind": "numeric"},
    {"name": "region", "kind": "categorical"},
    {"name": "channel", "kind": "categorical"},
]
REGIONS = ["north", "south", "east", "west", "central"]
REGION_P = [0.25, 0.20, 0.20, 0.20, 0.15]
CHANNELS = ["web", "branch", "partner"]
CHANNEL_P = [0.6, 0.3, 0.1]
DRIFTS = ("none", "income_up_15pct", "region_mix", "score_spread", "age_missing", "new_channel")

# Demo scenario: the drift applied to each live batch.
DEMO_BATCHES = ["none", "none", "income_up_15pct", "region_mix", "score_spread", "age_missing", "new_channel"]


def sample(rows: int, seed: int, drift: str = "none") -> dict[str, list[object]]:
    """Generate rows applications; drift names one known change from DRIFTS."""
    if drift not in DRIFTS:
        raise ValueError(f"unknown drift {drift!r}; choose from {DRIFTS}")
    if not 1 <= rows <= 1_000_000:
        raise ValueError("rows must be 1 to 1,000,000")
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(41, 12, rows), 18, 90).round()
    income = rng.lognormal(np.log(52_000), 0.45, rows)
    if drift == "income_up_15pct":
        income *= 1.15
    spread = 1.6 if drift == "score_spread" else 1.0
    score = np.clip(300 + 550 * rng.beta(5, 2, rows), 300, 850)
    score = np.clip(score.mean() + (score - score.mean()) * spread, 300, 850).round()
    accounts = rng.poisson(3, rows)
    region_p = [0.10, 0.20, 0.20, 0.20, 0.30] if drift == "region_mix" else REGION_P
    region = rng.choice(REGIONS, size=rows, p=region_p)
    channels, channel_p = (
        ([*CHANNELS, "app"], [0.45, 0.25, 0.1, 0.2]) if drift == "new_channel" else (CHANNELS, CHANNEL_P)
    )
    channel = rng.choice(channels, size=rows, p=channel_p)
    age_col: list[object] = [int(a) for a in age]
    if drift == "age_missing":
        for i in np.flatnonzero(rng.random(rows) < 0.12):
            age_col[i] = None
    return {
        "age": age_col,
        "income": [round(float(x), 2) for x in income],
        "credit_score": [int(s) for s in score],
        "open_accounts": [int(a) for a in accounts],
        "region": [str(r) for r in region],
        "channel": [str(c) for c in channel],
    }


def write_demo(out: Path, training_rows: int = 20_000, batch_rows: int = 2_000, seed: int = 7) -> list[tuple[str, str]]:
    """Write training.csv and live/batch-NN.csv for the demo scenario; returns (file, drift) pairs."""
    write_csv(out / "training.csv", sample(training_rows, seed))
    plan = []
    for i, drift in enumerate(DEMO_BATCHES, start=1):
        name = f"batch-{i:02d}.csv"
        write_csv(out / "live" / name, sample(batch_rows, seed + i, drift))
        plan.append((name, drift))
    return plan
