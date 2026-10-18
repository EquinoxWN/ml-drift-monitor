"""The scheduled job checks each batch once, survives crashes and refuses to run twice at once."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_drift_monitor import jobs, monitor, reference, synth
from ml_drift_monitor.data import write_csv


def _live(tmp_path: Path, drifts: list[str]) -> Path:
    """Write one batch per drift name into a live folder."""
    live = tmp_path / "live"
    for i, d in enumerate(drifts, start=1):
        write_csv(live / f"batch-{i:02d}.csv", synth.sample(1_000, 900 + i, d))
    return live


def test_second_run_skips_finished_batches_and_rechecks_changed_ones(ref: reference.Reference, tmp_path: Path) -> None:
    live = _live(tmp_path, ["none", "region_mix", "none"])
    out = tmp_path / "reports"
    first = jobs.run_pending(ref, live, out)
    assert [r.batch for r in first.checked] == ["batch-01.csv", "batch-02.csv", "batch-03.csv"]
    assert first.worst == "drift"
    assert json.loads((out / "batch-02.json").read_text())["status"] == "drift"
    second = jobs.run_pending(ref, live, out)
    assert second.checked == []
    assert second.skipped == 3
    write_csv(live / "batch-03.csv", synth.sample(1_000, 999, "new_channel"))  # a corrected upload
    write_csv(live / "batch-04.csv", synth.sample(1_000, 1_000))
    third = jobs.run_pending(ref, live, out)
    assert [r.batch for r in third.checked] == ["batch-03.csv", "batch-04.csv"]
    assert third.skipped == 2


def test_a_crash_keeps_the_work_already_done(
    ref: reference.Reference, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _live(tmp_path, ["none", "none", "none"])
    out = tmp_path / "reports"
    real = monitor.check

    def crash_on_third(r: reference.Reference, b, name: str, th=None):  # type: ignore[no-untyped-def]
        if name == "batch-03.csv":
            raise RuntimeError("worker killed")
        return real(r, b, name, th)

    monkeypatch.setattr(monitor, "check", crash_on_third)
    with pytest.raises(RuntimeError, match="killed"):
        jobs.run_pending(ref, live, out)
    assert sorted(json.loads((out / "state.json").read_text())) == ["batch-01.csv", "batch-02.csv"]
    assert not (out / ".lock").exists()  # the lock is released even when the run fails
    monkeypatch.setattr(monitor, "check", real)
    rerun = jobs.run_pending(ref, live, out)
    assert [r.batch for r in rerun.checked] == ["batch-03.csv"]


def test_concurrent_runs_are_refused(ref: reference.Reference, tmp_path: Path) -> None:
    live = _live(tmp_path, ["none"])
    out = tmp_path / "reports"
    out.mkdir()
    (out / ".lock").write_text("")
    with pytest.raises(jobs.LockedError):
        jobs.run_pending(ref, live, out)
    assert not (out / "batch-01.json").exists()
