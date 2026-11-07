"""Reading feature batches from CSV with an explicit schema and hard size limits."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

Kind = Literal["numeric", "categorical"]
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_ROWS = 5_000_000
MAX_CATEGORY_CHARS = 64


@dataclass(frozen=True)
class Feature:
    """One monitored column and how it is compared."""

    name: str
    kind: Kind


@dataclass
class Batch:
    """Columns of one batch: float arrays (NaN = missing) or category lists (None = missing)."""

    rows: int
    numeric: dict[str, npt.NDArray[np.float64]]
    categorical: dict[str, list[str | None]]


def parse_schema(spec: list[dict[str, str]]) -> list[Feature]:
    """Validate a schema: unique names, known kinds, at least one feature."""
    features: list[Feature] = []
    seen: set[str] = set()
    if not isinstance(spec, list):
        raise ValueError("schema features must be a list")
    for item in spec:
        if not isinstance(item, dict):
            raise ValueError(f"schema entry {item!r} is not an object")
        name, kind = item.get("name", ""), item.get("kind", "")
        if not name or len(name) > 64 or name in seen:
            raise ValueError(f"feature name {name!r} empty, too long or repeated")
        if kind not in ("numeric", "categorical"):
            raise ValueError(f"feature {name!r}: kind must be numeric or categorical, not {kind!r}")
        seen.add(name)
        features.append(Feature(name, kind))  # type: ignore[arg-type]
    if not features:
        raise ValueError("schema has no features")
    return features


def read_csv(path: Path, features: list[Feature]) -> Batch:
    """Read the schema's columns; a missing column, an oversized file or a bad number is an error."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{path.name}: larger than {MAX_FILE_BYTES} bytes")
    numeric: dict[str, list[float]] = {f.name: [] for f in features if f.kind == "numeric"}
    categorical: dict[str, list[str | None]] = {f.name: [] for f in features if f.kind == "categorical"}
    rows = 0
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [f.name for f in features if f.name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        for line, row in enumerate(reader, start=2):
            rows += 1
            if rows > MAX_ROWS:
                raise ValueError(f"{path.name}: more than {MAX_ROWS} rows")
            for name, values in numeric.items():
                values.append(_number(row[name], path.name, line, name))
            for name, cats in categorical.items():
                cell = (row[name] or "").strip()
                cats.append(cell[:MAX_CATEGORY_CHARS] if cell else None)
    return Batch(
        rows=rows, numeric={k: np.asarray(v, dtype=np.float64) for k, v in numeric.items()}, categorical=categorical
    )


def _number(cell: str | None, file: str, line: int, name: str) -> float:
    """Parse one numeric cell; empty means missing, anything non-finite is refused."""
    text = (cell or "").strip()
    if not text:
        return math.nan
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{file}:{line}: {name}={text[:20]!r} is not a number") from None
    if not math.isfinite(value):
        raise ValueError(f"{file}:{line}: {name} is not finite")
    return value


def write_csv(path: Path, columns: dict[str, list[object]]) -> None:
    """Write equal-length columns as CSV (used by the synthetic data generator)."""
    names = list(columns)
    n = len(columns[names[0]])
    if any(len(columns[c]) != n for c in names):
        raise ValueError("columns differ in length")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(names)
        for i in range(n):
            writer.writerow(["" if columns[c][i] is None else columns[c][i] for c in names])
