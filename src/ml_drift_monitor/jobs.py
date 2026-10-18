"""The scheduled job: check every new or changed batch in a folder exactly once."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import monitor
from .data import read_csv
from .reference import Reference, sha256_file

MAX_BATCHES_PER_RUN = 1_000


@dataclass
class RunSummary:
    """What one scheduled run did."""

    checked: list[monitor.Report]
    skipped: int
    worst: monitor.Status


class LockedError(RuntimeError):
    """Another run holds the lock."""


def _write_atomic(path: Path, data: Any) -> None:
    """Write JSON through a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_pending(ref: Reference, live_dir: Path, out_dir: Path, th: monitor.Thresholds | None = None) -> RunSummary:
    """Check batches not yet seen (by name and SHA-256), write one report each, update the state."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise LockedError(f"{lock} exists: another run is active (delete it if a run crashed)") from None
    os.close(fd)
    try:
        state_path = out_dir / "state.json"
        state: dict[str, str] = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        batches = sorted(p for p in live_dir.glob("*.csv") if p.is_file())[:MAX_BATCHES_PER_RUN]
        checked: list[monitor.Report] = []
        skipped = 0
        for path in batches:
            digest = sha256_file(path)
            if state.get(path.name) == digest:
                skipped += 1
                continue
            report = monitor.check(ref, read_csv(path, ref.schema()), path.name, th)
            _write_atomic(out_dir / f"{path.stem}.json", report.to_dict())
            state[path.name] = digest
            _write_atomic(state_path, state)  # after each batch, so a crash never repeats finished work
            checked.append(report)
        worst: monitor.Status = max((r.status for r in checked), key=lambda s: monitor.ORDER[s], default="ok")
        return RunSummary(checked=checked, skipped=skipped, worst=worst)
    finally:
        lock.unlink(missing_ok=True)
