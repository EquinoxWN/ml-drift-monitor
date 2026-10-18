"""Shared fixtures: the demo schema, synthetic batches and a reference built from training data."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ml_drift_monitor import reference, synth
from ml_drift_monitor.data import Batch, Feature, parse_schema, read_csv, write_csv

MakeBatch = Callable[..., Batch]


@pytest.fixture(scope="session")
def features() -> list[Feature]:
    """The demo schema."""
    return parse_schema(synth.SCHEMA)


@pytest.fixture(scope="session")
def make_batch(tmp_path_factory: pytest.TempPathFactory, features: list[Feature]) -> MakeBatch:
    """Write a synthetic batch to CSV and read it back through the real reader."""
    folder = tmp_path_factory.mktemp("batches")

    def make(rows: int, seed: int, drift: str = "none") -> Batch:
        path = folder / f"{drift}-{rows}-{seed}.csv"
        write_csv(path, synth.sample(rows, seed, drift))
        return read_csv(path, features)

    return make


@pytest.fixture(scope="session")
def training_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """20,000 training rows (seed 7), as in the demo."""
    path = tmp_path_factory.mktemp("training") / "training.csv"
    write_csv(path, synth.sample(20_000, 7))
    return path


@pytest.fixture(scope="session")
def ref(training_csv: Path, features: list[Feature]) -> reference.Reference:
    """The reference window built from the training data."""
    return reference.build(read_csv(training_csv, features), features, training_csv)
