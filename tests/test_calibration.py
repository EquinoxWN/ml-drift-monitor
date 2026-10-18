"""False alarms on stable data and detection rates on shifted data, over many simulated batches."""

from __future__ import annotations

import numpy as np

from ml_drift_monitor import monitor, stats
from ml_drift_monitor.reference import NUMERIC_BINS


def _profile(train: np.ndarray) -> tuple[list[float], list[int], np.ndarray]:
    """Decile edges, counts and a 5,000-value sample, as the reference builder makes them."""
    edges = sorted({float(q) for q in np.quantile(train, np.linspace(0, 1, NUMERIC_BINS + 1)[1:-1])})
    counts = np.bincount(np.searchsorted(edges, train, side="left"), minlength=len(edges) + 1)
    return edges, counts.tolist(), np.sort(train[:5_000])


def _drift_rate(shift_sd: float, trials: int, rows: int, seed: int) -> float:
    """Share of batches the rule calls drift when the live mean moves by shift_sd standard deviations."""
    rng = np.random.default_rng(seed)
    edges, ref_counts, sample = _profile(rng.normal(0, 1, 20_000))
    th = monitor.Thresholds()
    hits = 0
    for _ in range(trials):
        live = rng.normal(shift_sd, 1, rows)
        live_counts = np.bincount(np.searchsorted(edges, live, side="left"), minlength=len(edges) + 1)
        status, _ = monitor.decide(stats.psi(ref_counts, live_counts), stats.ks_2samp(sample, live).p_value, th)
        hits += status == "drift"
    return hits / trials


def test_no_false_drift_alarms_on_stable_data() -> None:
    assert _drift_rate(0.0, trials=300, rows=2_000, seed=1) == 0.0


def test_ks_alone_false_alarm_rate_is_near_alpha() -> None:
    rng = np.random.default_rng(2)
    sample = np.sort(rng.normal(0, 1, 5_000))
    p = np.array([stats.ks_2samp(sample, rng.normal(0, 1, 2_000)).p_value for _ in range(400)])
    assert (p < 0.01).mean() <= 0.03  # alpha 0.01; 400 trials allow some noise


def test_a_half_standard_deviation_shift_is_always_drift() -> None:
    assert _drift_rate(0.5, trials=100, rows=2_000, seed=3) == 1.0


def test_small_shifts_warn_before_they_alarm() -> None:
    assert _drift_rate(0.1, trials=100, rows=2_000, seed=4) <= 0.05
