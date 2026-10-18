"""Drift statistics: Population Stability Index, two-sample Kolmogorov-Smirnov and chi-square.

Implemented with numpy and the standard library only; the tests check every value against SciPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

PSI_EPSILON = 1e-4


@dataclass(frozen=True)
class TestResult:
    """One statistic and, where defined, its p-value."""

    statistic: float
    p_value: float | None = None
    dof: int | None = None


def psi(reference_counts: npt.ArrayLike, live_counts: npt.ArrayLike, epsilon: float = PSI_EPSILON) -> float:
    """Population Stability Index over matching bins; empty bins are floored at epsilon."""
    ref = np.asarray(reference_counts, dtype=np.float64)
    live = np.asarray(live_counts, dtype=np.float64)
    if ref.shape != live.shape or ref.ndim != 1 or ref.size < 2:
        raise ValueError("PSI needs two 1-D count arrays of the same length (at least 2 bins)")
    if (ref < 0).any() or (live < 0).any() or ref.sum() == 0 or live.sum() == 0:
        raise ValueError("counts must be non-negative with a positive total")
    p = np.maximum(ref / ref.sum(), epsilon)
    q = np.maximum(live / live.sum(), epsilon)
    return float(np.sum((q - p) * np.log(q / p)))


def kolmogorov_sf(lam: float) -> float:
    """Survival function of the Kolmogorov distribution, Q(lam) = 2 sum (-1)^(k-1) exp(-2 k^2 lam^2)."""
    if lam < 0.2:
        return 1.0  # the series converges slowly here and the value is 1 to 1e-16
    total, sign = 0.0, 1.0
    for k in range(1, 101):
        term = sign * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-16:
            break
        sign = -sign
    return min(1.0, max(0.0, 2.0 * total))


def ks_2samp(reference: npt.ArrayLike, live: npt.ArrayLike) -> TestResult:
    """Two-sample KS statistic (largest ECDF gap) with the asymptotic p-value (Stephens' correction)."""
    a = np.sort(np.asarray(reference, dtype=np.float64))
    b = np.sort(np.asarray(live, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        raise ValueError("KS needs two non-empty samples")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("KS samples must be finite")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(a.size * b.size / (a.size + b.size))
    p = kolmogorov_sf((en + 0.12 + 0.11 / en) * d)
    return TestResult(statistic=d, p_value=p)


def _gamma_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by its series (x < a + 1)."""
    term = total = 1.0 / a
    ap = a
    for _ in range(1000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_fraction(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by Lentz's continued fraction (x >= a + 1)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(statistic: float, dof: int) -> float:
    """Upper tail probability of the chi-square distribution."""
    if dof < 1:
        raise ValueError("degrees of freedom must be at least 1")
    if statistic <= 0:
        return 1.0
    a, x = dof / 2.0, statistic / 2.0
    if x < a + 1.0:
        return max(0.0, 1.0 - _gamma_series(a, x))
    return max(0.0, _gamma_fraction(a, x))


def chi2_counts(reference_counts: npt.ArrayLike, live_counts: npt.ArrayLike) -> TestResult:
    """Chi-square test of homogeneity on a 2 x K table; categories empty in both are dropped."""
    ref = np.asarray(reference_counts, dtype=np.float64)
    live = np.asarray(live_counts, dtype=np.float64)
    if ref.shape != live.shape or ref.ndim != 1:
        raise ValueError("chi-square needs two 1-D count arrays of the same length")
    if (ref < 0).any() or (live < 0).any() or ref.sum() == 0 or live.sum() == 0:
        raise ValueError("counts must be non-negative with a positive total")
    keep = (ref + live) > 0
    table = np.vstack([ref[keep], live[keep]])
    if table.shape[1] < 2:
        return TestResult(statistic=0.0, p_value=1.0, dof=0)
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    stat = float(np.sum((table - expected) ** 2 / expected))
    dof = table.shape[1] - 1
    return TestResult(statistic=stat, p_value=chi2_sf(stat, dof), dof=dof)
