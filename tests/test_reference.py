"""The reference window: profiles, storage, and refusing damaged or inconsistent files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml_drift_monitor import reference
from ml_drift_monitor.data import Batch, Feature


def test_numeric_profile_has_decile_bins_and_a_sorted_sample(ref: reference.Reference) -> None:
    income = ref.features["income"]
    assert len(income.edges) == reference.NUMERIC_BINS - 1
    share = np.asarray(income.counts) / sum(income.counts)
    assert np.all(np.abs(share - 0.1) < 0.01)  # quantile bins hold about 10% each
    assert len(income.sample) == reference.SAMPLE_SIZE
    assert income.sample == sorted(income.sample)
    assert income.null_rate == 0.0


def test_discrete_feature_collapses_duplicate_edges(ref: reference.Reference) -> None:
    accounts = ref.features["open_accounts"]
    assert accounts.edges == sorted(set(accounts.edges))
    assert len(accounts.edges) < reference.NUMERIC_BINS - 1  # Poisson values repeat across deciles


def test_categorical_profile_counts_every_value(ref: reference.Reference) -> None:
    region = ref.features["region"]
    assert set(region.categories) == {"north", "south", "east", "west", "central"}
    assert sum(region.counts) == 20_000
    assert region.counts[-1] == 0  # nothing in OTHER for the training data


def test_categories_are_capped_with_the_rest_in_other() -> None:
    cats: list[str | None] = [f"c{i:03d}" for i in range(80) for _ in range(i + 1)]
    batch = Batch(rows=len(cats), numeric={}, categorical={"x": cats})
    ref = reference.build(batch, [Feature("x", "categorical")], Path(__file__))
    prof = ref.features["x"]
    assert len(prof.categories) == reference.MAX_CATEGORIES
    assert prof.categories[0] == "c079"  # most frequent first
    assert prof.counts[-1] == sum(range(1, 31))  # the 30 rarest categories fall into OTHER


def test_save_load_round_trip(ref: reference.Reference, tmp_path: Path) -> None:
    path = tmp_path / "ref.json"
    reference.save(ref, path)
    back = reference.load(path)
    assert back.features == ref.features
    assert back.source_sha256 == ref.source_sha256
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["features"][0]["counts"].__setitem__(0, 999_999), "checksum"),
        (lambda d: d.__setitem__("version", 2), "version"),
    ],
)
def test_load_refuses_tampered_files(ref: reference.Reference, tmp_path: Path, edit, message: str) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "ref.json"
    reference.save(ref, path)
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        reference.load(path)


def test_load_refuses_inconsistent_profiles_even_with_a_valid_checksum(
    ref: reference.Reference, tmp_path: Path
) -> None:
    body = reference._body(ref)
    body["features"][0]["counts"] = body["features"][0]["counts"][:-1]  # one bin short
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({**body, "sha256": reference._digest(body)}))
    with pytest.raises(ValueError, match="inconsistent"):
        reference.load(path)
    body = reference._body(ref)
    body["features"][0]["extra"] = 1
    path.write_text(json.dumps({**body, "sha256": reference._digest(body)}))
    with pytest.raises(ValueError, match="unknown fields"):
        reference.load(path)


def test_build_refuses_tiny_windows(features: list[Feature], make_batch) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="at least 100 rows"):
        reference.build(make_batch(50, 1), features, Path(__file__))
