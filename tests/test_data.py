"""CSV reading and schema validation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ml_drift_monitor.data import Feature, parse_schema, read_csv, write_csv

SCHEMA = [Feature("x", "numeric"), Feature("c", "categorical")]


def _csv(tmp_path: Path, text: str) -> Path:
    """Write a CSV file."""
    path = tmp_path / "b.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_numbers_categories_and_missing_values(tmp_path: Path) -> None:
    b = read_csv(_csv(tmp_path, "x,c,ignored\n1.5,a,z\n,b,z\n3,,z\n"), SCHEMA)
    assert b.rows == 3
    assert b.numeric["x"][0] == 1.5
    assert math.isnan(b.numeric["x"][1])
    assert b.categorical["c"] == ["a", "b", None]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("x\n1\n", "missing columns"),
        ("x,c\nabc,a\n", "not a number"),
        ("x,c\ninf,a\n", "not finite"),
        ("x,c\nnan,a\n", "not finite"),
    ],
)
def test_refuses_bad_files(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_csv(_csv(tmp_path, text), SCHEMA)


def test_long_categories_are_truncated(tmp_path: Path) -> None:
    b = read_csv(_csv(tmp_path, "x,c\n1," + "y" * 500 + "\n"), SCHEMA)
    assert b.categorical["c"] == ["y" * 64]


@pytest.mark.parametrize(
    "spec",
    [
        [],
        [{"name": "a", "kind": "text"}],
        [{"name": "a", "kind": "numeric"}, {"name": "a", "kind": "numeric"}],
        [{"kind": "numeric"}],
    ],
)
def test_schema_validation(spec: list[dict[str, str]]) -> None:
    with pytest.raises(ValueError, match=r"feature|schema"):
        parse_schema(spec)


def test_write_csv_refuses_ragged_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="length"):
        write_csv(tmp_path / "x.csv", {"a": [1, 2], "b": [1]})
