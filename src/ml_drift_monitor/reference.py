"""Reference window: a stored profile of each feature in the training data."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import Batch, Feature, Kind

SCHEMA_VERSION = 1
NUMERIC_BINS = 10
SAMPLE_SIZE = 5_000
MAX_CATEGORIES = 50
OTHER = "__other__"
MAX_REFERENCE_BYTES = 32 * 1024 * 1024


@dataclass
class FeatureProfile:
    """What the training data looked like for one feature."""

    name: str
    kind: Kind
    rows: int
    null_rate: float
    # numeric: inner bin edges (quantiles), counts per bin (len(edges) + 1), sorted sample for KS
    edges: list[float] = field(default_factory=list)
    # categorical: kept categories in order; counts has one more slot for OTHER
    categories: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    sample: list[float] = field(default_factory=list)


@dataclass
class Reference:
    """The reference window for every monitored feature, with where it came from."""

    source: str
    source_sha256: str
    rows: int
    features: dict[str, FeatureProfile]
    version: int = SCHEMA_VERSION

    def schema(self) -> list[Feature]:
        """The monitored features in stored order."""
        return [Feature(p.name, p.kind) for p in self.features.values()]


def numeric_counts(values: np.ndarray, edges: list[float]) -> list[int]:
    """Counts per bin for the inner edges; bins are (-inf, e0], (e0, e1], ..., (e_last, inf)."""
    present = values[~np.isnan(values)]
    idx = np.searchsorted(np.asarray(edges, dtype=np.float64), present, side="left")
    return np.bincount(idx, minlength=len(edges) + 1).astype(int).tolist()


def categorical_counts(values: list[str | None], categories: list[str]) -> list[int]:
    """Counts per kept category, with unseen categories in the last (OTHER) slot."""
    index = {c: i for i, c in enumerate(categories)}
    counts = [0] * (len(categories) + 1)
    for v in values:
        if v is not None:
            counts[index.get(v, len(categories))] += 1
    return counts


def build(batch: Batch, features: list[Feature], source: Path, seed: int = 0) -> Reference:
    """Profile the training batch: quantile bins and a sample per numeric feature, counts per category."""
    if batch.rows < 100:
        raise ValueError(f"a reference window needs at least 100 rows, got {batch.rows}")
    rng = np.random.default_rng(seed)
    profiles: dict[str, FeatureProfile] = {}
    for f in features:
        if f.kind == "numeric":
            values = batch.numeric[f.name]
            present = values[~np.isnan(values)]
            if present.size < 100:
                raise ValueError(f"{f.name}: fewer than 100 non-missing values")
            quantiles = np.quantile(present, np.linspace(0, 1, NUMERIC_BINS + 1)[1:-1])
            edges = sorted({float(q) for q in quantiles})
            take = rng.choice(present.size, size=min(SAMPLE_SIZE, present.size), replace=False)
            profiles[f.name] = FeatureProfile(
                name=f.name,
                kind="numeric",
                rows=batch.rows,
                null_rate=1 - present.size / batch.rows,
                edges=edges,
                counts=numeric_counts(values, edges),
                sample=sorted(float(x) for x in present[take]),
            )
        else:
            cats = batch.categorical[f.name]
            present_cats = [c for c in cats if c is not None]
            freq: dict[str, int] = {}
            for c in present_cats:
                freq[c] = freq.get(c, 0) + 1
            kept = sorted(freq, key=lambda c: (-freq[c], c))[:MAX_CATEGORIES]
            profiles[f.name] = FeatureProfile(
                name=f.name,
                kind="categorical",
                rows=batch.rows,
                null_rate=1 - len(present_cats) / batch.rows,
                categories=kept,
                counts=categorical_counts(cats, kept),
            )
    return Reference(source=source.name, source_sha256=sha256_file(source), rows=batch.rows, features=profiles)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _body(ref: Reference) -> dict[str, Any]:
    """The stored fields, without the checksum."""
    return {
        "version": ref.version,
        "source": ref.source,
        "source_sha256": ref.source_sha256,
        "rows": ref.rows,
        "features": [asdict(p) for p in ref.features.values()],
    }


def _digest(body: dict[str, Any]) -> str:
    """Checksum of the canonical JSON body."""
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(ref: Reference, path: Path) -> None:
    """Write the reference atomically with a checksum over its content."""
    body = _body(ref)
    data = json.dumps({**body, "sha256": _digest(body)}, indent=1, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load(path: Path) -> Reference:
    """Read a reference, refusing unknown versions, a bad checksum or inconsistent profiles."""
    if path.stat().st_size > MAX_REFERENCE_BYTES:
        raise ValueError(f"{path.name}: larger than {MAX_REFERENCE_BYTES} bytes")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        raise ValueError(f"{path.name}: not a version {SCHEMA_VERSION} reference")
    stored = raw.pop("sha256", None)
    if stored != _digest(raw):
        raise ValueError(f"{path.name}: checksum mismatch (file changed or corrupted)")
    allowed = set(FeatureProfile.__dataclass_fields__)
    profiles: dict[str, FeatureProfile] = {}
    for item in raw["features"]:
        if set(item) - allowed:
            raise ValueError(f"{path.name}: unknown fields {sorted(set(item) - allowed)}")
        p = FeatureProfile(**item)
        _check_profile(p)
        profiles[p.name] = p
    return Reference(
        source=raw["source"],
        source_sha256=raw["source_sha256"],
        rows=int(raw["rows"]),
        features=profiles,
        version=raw["version"],
    )


def _check_profile(p: FeatureProfile) -> None:
    """Consistency rules a profile must meet before it is used."""
    if p.kind == "numeric":
        ok = len(p.counts) == len(p.edges) + 1 and p.edges == sorted(set(p.edges)) and len(p.sample) > 0
    elif p.kind == "categorical":
        ok = len(p.counts) == len(p.categories) + 1 and len(set(p.categories)) == len(p.categories)
    else:
        ok = False
    if not ok or any(c < 0 for c in p.counts) or sum(p.counts) == 0 or not 0 <= p.null_rate <= 1:
        raise ValueError(f"feature {p.name!r}: inconsistent profile")
