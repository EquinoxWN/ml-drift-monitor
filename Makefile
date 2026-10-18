.PHONY: setup lint test audit demo bench ci

PY ?= python
DEMO := .tmp/demo

setup:
	$(PY) -m pip install --upgrade pip && $(PY) -m pip install -e ".[dev]"

# ruff, ruff format and mypy --strict (src, tests and scripts).
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && $(PY) -m mypy

# Statistics against SciPy, reference storage, decisions, the scheduled job, CLI end to end.
test:
	$(PY) -m pytest -q

# Known vulnerabilities in the installed Python dependencies.
audit:
	$(PY) -m pip_audit --skip-editable --cache-dir .tmp/pip-audit

# Synthetic training data and 7 live batches, the reference window, then two scheduled runs:
# the first checks every batch (exit 1: drift found), the second finds nothing new.
demo:
	drift synth --out $(DEMO)
	drift reference --training $(DEMO)/training.csv --schema $(DEMO)/schema.json --out $(DEMO)/reference.json
	drift run --reference $(DEMO)/reference.json --live $(DEMO)/live --out $(DEMO)/reports || [ $$? -eq 1 ]
	drift run --reference $(DEMO)/reference.json --live $(DEMO)/live --out $(DEMO)/reports

# False alarms on stable data and detection per drift type, 200 batches each.
bench:
	$(PY) scripts/measure.py --batches 200

ci: setup lint test
