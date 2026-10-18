"""Measure the decision rule on synthetic data: false alarms on stable batches, detection per drift type.

usage: python scripts/measure.py [--batches 200] [--rows 2000]
Prints one Markdown table row per scenario; every batch is a fresh sample (seeded, reproducible).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ml_drift_monitor import monitor, reference, synth
from ml_drift_monitor.data import Batch, parse_schema

TARGET = {
    "none": "-",
    "income_up_15pct": "income",
    "region_mix": "region",
    "score_spread": "credit_score",
    "age_missing": "age",
    "new_channel": "channel",
}


def to_batch(cols: dict[str, list[object]]) -> Batch:
    """Turn generated columns into a Batch without a CSV round trip."""
    numeric = {
        k: np.asarray([np.nan if v is None else float(v) for v in cols[k]])  # type: ignore[arg-type]
        for k in ("age", "income", "credit_score", "open_accounts")
    }
    categorical = {k: [None if v is None else str(v) for v in cols[k]] for k in ("region", "channel")}
    return Batch(rows=len(cols["age"]), numeric=numeric, categorical=categorical)


def main() -> None:
    """Run every scenario and print the table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--rows", type=int, default=2_000)
    args = parser.parse_args()
    features = parse_schema(synth.SCHEMA)
    ref = reference.build(to_batch(synth.sample(20_000, 7)), features, Path(__file__))
    print(f"reference: 20,000 rows (seed 7); {args.batches} batches of {args.rows} rows per scenario\n")
    print(
        "| Scenario | Target feature | Batch status drift / warn / ok | Target feature flagged drift | "
        "Other features flagged drift | ms per batch (generate + check) |"
    )
    print("|---|---|---|---|---|---|")
    for i, drift in enumerate(TARGET):
        statuses: Counter[str] = Counter()
        target_hits = other_hits = 0
        started = time.perf_counter()
        for b in range(args.batches):
            report = monitor.check(ref, to_batch(synth.sample(args.rows, 10_000 * (i + 1) + b, drift)), drift)
            statuses[report.status] += 1
            for f in report.features:
                if f.status == "drift":
                    if f.name == TARGET[drift]:
                        target_hits += 1
                    else:
                        other_hits += 1
        ms = (time.perf_counter() - started) * 1000 / args.batches
        target = "-" if drift == "none" else f"{target_hits}/{args.batches}"
        others = f"{other_hits}/{args.batches * (len(features) - (drift != 'none'))}"
        print(
            f"| {drift} | {TARGET[drift]} | {statuses['drift']} / {statuses['warn']} / {statuses['ok']} | "
            f"{target} | {others} | {ms:.1f} |"
        )


if __name__ == "__main__":
    main()
