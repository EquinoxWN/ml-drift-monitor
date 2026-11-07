"""Command line: synth, reference, check and run (the scheduled job).

Exit codes: 0 no drift, 1 drift found, 2 bad input or another run holds the lock.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from . import jobs, monitor, reference, synth
from .data import parse_schema, read_csv


def _table(report: monitor.Report, out: TextIO) -> None:
    """Print one report as aligned rows."""
    for f in report.features:
        psi = "-" if f.psi is None else f"{f.psi:.3f}"
        stat = "-" if f.statistic is None else f"{f.statistic:.3f}"
        p = "-" if f.p_value is None else f"{f.p_value:.1e}"
        why = "; ".join(f.reasons)
        print(f"  {f.name:<14} {f.test:<4} psi {psi:>6}  stat {stat:>8}  p {p:>8}  {f.status:<12} {why}", file=out)


def _write_schema(path: Path) -> None:
    """Write the demo schema next to the demo data."""
    path.write_text(json.dumps({"features": synth.SCHEMA}, indent=1) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None, out: TextIO = sys.stdout) -> int:
    """Parse arguments and run one command."""
    parser = argparse.ArgumentParser(
        prog="drift", description="Feature drift checks against a stored reference window."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="write synthetic training data, live batches and a schema")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--training-rows", type=int, default=20_000)
    s.add_argument("--batch-rows", type=int, default=2_000)
    s.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("reference", help="profile training data into a reference window")
    r.add_argument("--training", type=Path, required=True)
    r.add_argument("--schema", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check", help="compare one live batch with the reference")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--batch", type=Path, required=True)
    c.add_argument("--json", type=Path)
    j = sub.add_parser("run", help="scheduled job: check every new or changed batch in a folder")
    j.add_argument("--reference", type=Path, required=True)
    j.add_argument("--live", type=Path, required=True)
    j.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return _dispatch(args, out)
    except (ValueError, OSError, KeyError, jobs.LockedError) as exc:
        print(f"drift: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, out: TextIO) -> int:
    """Run the parsed command."""
    if args.cmd == "synth":
        plan = synth.write_demo(args.out, args.training_rows, args.batch_rows, args.seed)
        _write_schema(args.out / "schema.json")
        for name, drift in plan:
            print(f"wrote live/{name}: drift={drift}", file=out)
        return 0
    if args.cmd == "reference":
        spec = json.loads(args.schema.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or "features" not in spec:
            raise ValueError(f"{args.schema}: expected an object with a features list")
        features = parse_schema(spec["features"])
        ref = reference.build(read_csv(args.training, features), features, args.training)
        reference.save(ref, args.out)
        print(
            f"reference: {ref.rows} rows, {len(ref.features)} features, source sha256 {ref.source_sha256[:12]}",
            file=out,
        )
        return 0
    ref = reference.load(args.reference)
    if args.cmd == "check":
        report = monitor.check(ref, read_csv(args.batch, ref.schema()), args.batch.name)
        print(f"{report.batch}: {report.status.upper()} ({report.rows} rows)", file=out)
        _table(report, out)
        if args.json:
            args.json.write_text(json.dumps(report.to_dict(), indent=1) + "\n", encoding="utf-8")
        return 1 if report.status == "drift" else 0
    summary = jobs.run_pending(ref, args.live, args.out)
    for rep in summary.checked:
        print(f"{rep.batch}: {rep.status.upper()} ({rep.rows} rows)", file=out)
        _table(rep, out)
    drifted = sum(r.status == "drift" for r in summary.checked)
    print(f"RESULT run: {len(summary.checked)} checked, {summary.skipped} already done, {drifted} with drift", file=out)
    return 1 if summary.worst == "drift" else 0


if __name__ == "__main__":
    raise SystemExit(main())
