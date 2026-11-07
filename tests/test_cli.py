"""End to end through the command line, as the scheduled job runs it."""

from __future__ import annotations

import io
import json
from pathlib import Path

from ml_drift_monitor.cli import main


def _run(*args: str) -> tuple[int, str]:
    """Run the CLI and capture its output."""
    out = io.StringIO()
    return main(list(args), out=out), out.getvalue()


def test_synth_reference_run_and_check(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    code, text = _run("synth", "--out", str(demo), "--training-rows", "5000", "--batch-rows", "1000")
    assert code == 0
    assert text.count("wrote live/") == 7
    code, text = _run(
        "reference",
        "--training",
        str(demo / "training.csv"),
        "--schema",
        str(demo / "schema.json"),
        "--out",
        str(demo / "reference.json"),
    )
    assert code == 0
    assert "5000 rows, 6 features" in text
    args = (
        "run",
        "--reference",
        str(demo / "reference.json"),
        "--live",
        str(demo / "live"),
        "--out",
        str(demo / "reports"),
    )
    code, text = _run(*args)
    assert code == 1  # drift found
    assert "RESULT run: 7 checked, 0 already done" in text
    assert "batch-04.csv: DRIFT" in text
    code, text = _run(*args)
    assert code == 0
    assert "RESULT run: 0 checked, 7 already done, 0 with drift" in text
    report_path = tmp_path / "b1.json"
    code, text = _run(
        "check",
        "--reference",
        str(demo / "reference.json"),
        "--batch",
        str(demo / "live" / "batch-01.csv"),
        "--json",
        str(report_path),
    )
    assert code == 0
    assert json.loads(report_path.read_text())["batch"] == "batch-01.csv"


def test_bad_input_exits_2(tmp_path: Path) -> None:
    code, _ = _run("check", "--reference", str(tmp_path / "missing.json"), "--batch", str(tmp_path / "b.csv"))
    assert code == 2
    bad = tmp_path / "ref.json"
    bad.write_text('{"version": 1}')
    code, _ = _run("check", "--reference", str(bad), "--batch", str(tmp_path / "b.csv"))
    assert code == 2


def test_a_schema_that_is_not_an_object_exits_2(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("[1]")
    training = tmp_path / "t.csv"
    training.write_text("a,b" + chr(10) + "1,x" + chr(10))
    code, _ = _run("reference", "--training", str(training), "--schema", str(schema), "--out", str(tmp_path / "r.json"))
    assert code == 2
