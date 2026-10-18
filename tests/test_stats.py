"""Every statistic against SciPy, plus properties that must hold for any input."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats as sp

from ml_drift_monitor import stats

counts = st.lists(st.integers(min_value=0, max_value=10_000), min_size=2, max_size=30).filter(lambda c: sum(c) > 0)


@given(counts)
def test_psi_is_zero_for_identical_proportions(c: list[int]) -> None:
    assert stats.psi(c, [x * 3 for x in c]) == pytest.approx(0.0, abs=1e-12)


@given(st.data())
@settings(max_examples=200)
def test_psi_is_non_negative_and_symmetric(data: st.DataObject) -> None:
    a = data.draw(counts)
    b = data.draw(st.lists(st.integers(0, 10_000), min_size=len(a), max_size=len(a)).filter(lambda c: sum(c) > 0))
    assert stats.psi(a, b) >= 0
    assert stats.psi(a, b) == pytest.approx(stats.psi(b, a), rel=1e-9)


def test_psi_known_value() -> None:
    # p = (0.5, 0.5), q = (0.9, 0.1): 0.4 ln 1.8 + (-0.4) ln 0.2
    assert stats.psi([50, 50], [90, 10]) == pytest.approx(0.4 * np.log(1.8) - 0.4 * np.log(0.2), rel=1e-12)


@pytest.mark.parametrize(("ref", "live"), [([1], [1]), ([1, 2], [1, 2, 3]), ([0, 0], [1, 1]), ([1, -1], [1, 1])])
def test_psi_refuses_bad_counts(ref: list[int], live: list[int]) -> None:
    with pytest.raises(ValueError, match=r"PSI|counts"):
        stats.psi(ref, live)


def test_kolmogorov_survival_matches_scipy() -> None:
    for lam in np.linspace(0.2, 4.0, 77):
        assert stats.kolmogorov_sf(float(lam)) == pytest.approx(sp.kstwobign.sf(lam), rel=1e-9, abs=1e-15)


@given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=80), st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=80))
@settings(max_examples=300)
def test_ks_statistic_equals_scipy_including_ties(a: list[float], b: list[float]) -> None:
    ours = stats.ks_2samp(a, b)
    with np.errstate(all="ignore"):  # SciPy's own p-value divides by zero for one-element samples
        oracle = sp.ks_2samp(a, b, method="asymp").statistic
    assert ours.statistic == pytest.approx(oracle, abs=1e-12)
    assert ours.p_value is not None
    assert 0.0 <= ours.p_value <= 1.0


def test_ks_p_value_close_to_scipy_exact() -> None:
    rng = np.random.default_rng(0)
    worst = 0.0
    for n, m in [(50, 60), (200, 300), (1_000, 1_000), (5_000, 2_000)]:
        for shift in (0.0, 0.05, 0.1, 0.2):
            a, b = rng.normal(0, 1, n), rng.normal(shift, 1, m)
            p = stats.ks_2samp(a, b).p_value
            assert p is not None
            worst = max(worst, abs(p - sp.ks_2samp(a, b, method="exact").pvalue))
    assert worst < 0.025  # asymptotic with Stephens' correction vs exact, n >= 50


def test_ks_refuses_empty_or_non_finite() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        stats.ks_2samp([], [1.0])
    with pytest.raises(ValueError, match="finite"):
        stats.ks_2samp([1.0, float("nan")], [1.0])


def test_chi2_survival_matches_scipy() -> None:
    for dof in (1, 2, 3, 5, 10, 30, 60):
        for x in (0.01, 0.5, 1.0, dof * 0.5, dof, dof + 1.0, dof * 2.0, dof * 5.0, 200.0):
            assert stats.chi2_sf(x, dof) == pytest.approx(sp.chi2.sf(x, dof), rel=1e-9, abs=1e-300)
    assert stats.chi2_sf(0.0, 3) == 1.0
    with pytest.raises(ValueError, match="degrees"):
        stats.chi2_sf(1.0, 0)


@given(st.data())
@settings(max_examples=200)
def test_chi2_counts_matches_scipy_contingency(data: st.DataObject) -> None:
    k = data.draw(st.integers(2, 12))
    ref = data.draw(st.lists(st.integers(1, 5_000), min_size=k, max_size=k))
    live = data.draw(st.lists(st.integers(1, 5_000), min_size=k, max_size=k))
    ours = stats.chi2_counts(ref, live)
    stat, p, dof, _ = sp.chi2_contingency(np.array([ref, live]), correction=False)
    assert ours.statistic == pytest.approx(stat, rel=1e-9, abs=1e-9)
    assert ours.p_value == pytest.approx(p, rel=1e-7, abs=1e-12)
    assert ours.dof == dof


def test_chi2_counts_drops_categories_empty_on_both_sides() -> None:
    assert stats.chi2_counts([10, 0, 5], [12, 0, 7]).dof == 1
    assert stats.chi2_counts([10, 0], [12, 0]) == stats.TestResult(0.0, 1.0, 0)
