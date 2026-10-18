# Contributing

Thanks for helping. This is a Python project (numpy at runtime); issues and pull requests are
welcome.

## Set up and check your change

Needs Python 3.11+ (CI uses 3.12) and make. Use a virtualenv inside the repo (`.venv/`, git-ignored).

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
make setup   # pip install -e ".[dev]"
make lint    # ruff, ruff format --check, mypy --strict
make test    # pytest (warnings are errors)
make audit   # pip-audit, as in CI
make demo    # synthetic data, reference window, two scheduled runs
make bench   # false alarms and detection over 200 batches per scenario
```

A pull request is ready when `make lint` and `make test` pass and CI is green.

## Guidelines

- Keep each change focused; explain the problem it solves in the pull request.
- New statistics must be checked against SciPy (or another reference implementation) in
  `tests/test_stats.py`, with property tests where they apply.
- Changing a threshold needs the `make bench` table before and after in the pull request, and an
  update to `docs/results/`.
- Changes to the reference file format bump `SCHEMA_VERSION`.
- Significant design changes need an ADR in `docs/adr/` (copy `0001` for the format).
- Keep function comments to one short phrase; let names and tests explain the rest.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your work is released under the [MIT License](LICENSE).
