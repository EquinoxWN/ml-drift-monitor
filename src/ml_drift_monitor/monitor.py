"""Compare one live batch with the reference window, feature by feature."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from . import stats
from .data import Batch
from .reference import Reference, categorical_counts, numeric_counts

Status = Literal["ok", "warn", "drift", "insufficient"]
ORDER: dict[Status, int] = {"ok": 0, "insufficient": 1, "warn": 2, "drift": 3}


@dataclass(frozen=True)
class Thresholds:
    """Decision rules; PSI bands follow the common 0.1 / 0.2 convention."""

    psi_warn: float = 0.1
    psi_drift: float = 0.2
    alpha: float = 0.01
    min_rows: int = 200
    null_rate_delta: float = 0.05


@dataclass
class FeatureResult:
    """Outcome for one feature."""

    name: str
    kind: str
    rows: int
    null_rate: float
    reference_null_rate: float
    psi: float | None
    test: str
    statistic: float | None
    p_value: float | None
    status: Status
    reasons: list[str] = field(default_factory=list)


@dataclass
class Report:
    """Outcome for one batch."""

    batch: str
    rows: int
    status: Status
    features: list[FeatureResult]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


def decide(psi: float, p_value: float | None, th: Thresholds) -> tuple[Status, list[str]]:
    """Drift when PSI is large, or the test is significant and PSI is at least moderate."""
    significant = p_value is not None and p_value < th.alpha
    reasons = []
    if psi >= th.psi_drift:
        reasons.append(f"PSI {psi:.3f} >= {th.psi_drift}")
    elif psi >= th.psi_warn:
        reasons.append(f"PSI {psi:.3f} >= {th.psi_warn}")
    if significant:
        reasons.append(f"p {p_value:.2g} < {th.alpha}")
    if psi >= th.psi_drift or (significant and psi >= th.psi_warn):
        return "drift", reasons
    if psi >= th.psi_warn or significant:
        return "warn", reasons
    return "ok", reasons


def check(ref: Reference, batch: Batch, name: str, th: Thresholds | None = None) -> Report:
    """Run PSI plus KS (numeric) or chi-square (categorical) for every reference feature."""
    th = th or Thresholds()
    results: list[FeatureResult] = []
    for p in ref.features.values():
        if p.kind == "numeric":
            values = batch.numeric[p.name]
            present = values[~np.isnan(values)]
            live_counts = numeric_counts(values, p.edges)
            test_name = "ks"
        else:
            cats = batch.categorical[p.name]
            present = np.asarray([c for c in cats if c is not None], dtype=object)
            live_counts = categorical_counts(cats, p.categories)
            test_name = "chi2"
        null_rate = 1 - present.size / batch.rows if batch.rows else 1.0
        res = FeatureResult(
            name=p.name,
            kind=p.kind,
            rows=int(present.size),
            null_rate=round(null_rate, 6),
            reference_null_rate=round(p.null_rate, 6),
            psi=None,
            test=test_name,
            statistic=None,
            p_value=None,
            status="insufficient",
        )
        if present.size < th.min_rows:
            res.reasons.append(f"{present.size} values, need {th.min_rows}")
            results.append(res)
            continue
        res.psi = stats.psi(p.counts, live_counts)
        if p.kind == "numeric":
            t = stats.ks_2samp(p.sample, present.astype(np.float64))
        else:
            t = stats.chi2_counts(p.counts, live_counts)
        res.statistic, res.p_value = t.statistic, t.p_value
        res.status, res.reasons = decide(res.psi, res.p_value, th)
        if null_rate - p.null_rate > th.null_rate_delta:
            res.status = "drift"
            res.reasons.append(f"missing values {null_rate:.1%} vs {p.null_rate:.1%} in reference")
        results.append(res)
    worst: Status = max((r.status for r in results), key=lambda s: ORDER[s], default="ok")
    return Report(batch=name, rows=batch.rows, status=worst, features=results)
