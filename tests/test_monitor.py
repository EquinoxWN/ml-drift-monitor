"""Drift decisions on batches with known changes."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import MakeBatch

from ml_drift_monitor import monitor, reference
from ml_drift_monitor.data import Batch


def _status(report: monitor.Report) -> dict[str, str]:
    """Feature name to status."""
    return {f.name: f.status for f in report.features}


@pytest.mark.parametrize("seed", [101, 102, 103, 104, 105])
def test_fresh_data_from_the_same_distribution_is_not_drift(
    ref: reference.Reference, make_batch: MakeBatch, seed: int
) -> None:
    report = monitor.check(ref, make_batch(2_000, seed), "same")
    assert report.status != "drift", [(f.name, f.reasons) for f in report.features]


@pytest.mark.parametrize(
    ("drift", "feature", "test"),
    [
        ("region_mix", "region", "chi2"),
        ("score_spread", "credit_score", "ks"),
        ("age_missing", "age", "ks"),
        ("new_channel", "channel", "chi2"),
    ],
)
def test_each_injected_drift_is_flagged_on_its_feature_only(
    ref: reference.Reference, make_batch: MakeBatch, drift: str, feature: str, test: str
) -> None:
    report = monitor.check(ref, make_batch(2_000, 200, drift), drift)
    status = _status(report)
    assert status[feature] == "drift"
    assert report.status == "drift"
    assert next(f for f in report.features if f.name == feature).test == test
    others = {k: v for k, v in status.items() if k != feature}
    assert "drift" not in others.values(), others


def test_income_up_15_percent_is_caught_by_ks_while_psi_stays_below_its_drift_band(
    ref: reference.Reference, make_batch: MakeBatch
) -> None:
    results = [
        next(
            f
            for f in monitor.check(ref, make_batch(2_000, 300 + i, "income_up_15pct"), "x").features
            if f.name == "income"
        )
        for i in range(40)
    ]
    assert all(f.p_value is not None and f.p_value < 1e-10 for f in results)  # KS: always significant
    assert all(f.psi is not None and f.psi < 0.2 for f in results)  # PSI alone at 0.2 would never alarm
    assert {f.status for f in results} <= {"warn", "drift"}  # the combined rule never calls it ok


def test_new_category_goes_to_other(ref: reference.Reference, make_batch: MakeBatch) -> None:
    batch = make_batch(2_000, 400, "new_channel")
    counts = reference.categorical_counts(batch.categorical["channel"], ref.features["channel"].categories)
    assert counts[-1] == batch.categorical["channel"].count("app")
    assert counts[-1] > 0


def test_small_batches_are_insufficient_not_ok(ref: reference.Reference, make_batch: MakeBatch) -> None:
    report = monitor.check(ref, make_batch(150, 500), "small")
    assert {f.status for f in report.features} == {"insufficient"}
    assert report.status == "insufficient"


def test_missing_column_is_an_error(ref: reference.Reference) -> None:
    batch = Batch(rows=500, numeric={"age": np.ones(500)}, categorical={})
    with pytest.raises(KeyError):
        monitor.check(ref, batch, "broken")


@pytest.mark.parametrize(
    ("psi", "p", "want"),
    [
        (0.05, 0.5, "ok"),
        (0.05, 0.001, "warn"),  # significant but tiny effect: large batches make KS significant easily
        (0.15, 0.5, "warn"),
        (0.15, 0.001, "drift"),
        (0.25, 0.5, "drift"),
        (0.25, None, "drift"),
    ],
)
def test_decision_rule(psi: float, p: float | None, want: str) -> None:
    assert monitor.decide(psi, p, monitor.Thresholds())[0] == want
