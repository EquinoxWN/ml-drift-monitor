# RFC 0001: ml-drift-monitor design

- **Status:** Accepted (M1 built)
- **Author:** EquinoxWN
- **Created:** 2026-10-18

## Problem (Define)

A deployed model rarely fails loudly. The inputs it sees drift away from the training data (a new
sales channel, a price change, a broken upstream join that fills a column with blanks) and accuracy
erodes for weeks before anyone looks, because labels arrive late or never. Teams need a cheap,
explainable check that runs on a schedule, compares each new batch of inputs with what the model
was trained on, and says which feature changed and how much.

Users: ML engineers and data scientists who own a model in production; on-call engineers who need
an alert they can act on.

### Critical-to-quality measures

| CTQ | Target (M1) | How it is measured |
|---|---|---|
| Correct statistics | PSI, KS and chi-square equal SciPy's values (statistic exactly, p-values within 1e-9 relative, KS asymptotic p within 0.025 of SciPy's exact) | `tests/test_stats.py`, including property tests over random inputs |
| False alarms on stable data | 0 drift verdicts in 200 stable batches of 2,000 rows | `scripts/measure.py` |
| Detection | Each injected drift flagged as drift on its own feature in at least 95% of batches, with no other feature flagged | `scripts/measure.py`, `tests/test_monitor.py` |
| Each batch checked once | A rerun of the scheduled job re-checks only new or changed files; a crash loses no finished work; two runs cannot overlap | `tests/test_jobs.py` |
| Reference integrity | A changed or inconsistent reference file is refused | `tests/test_reference.py` |

## Goals

- M1: store a reference window per feature from the training data; a scheduled job compares each
  new live batch with it using PSI plus KS (numeric) or chi-square (categorical), and writes one
  report per batch.
- Explainable output: per feature, the statistic, the p-value, the PSI and the reason for the
  verdict.

## Non-goals

- Prediction and confidence tracking, delayed-label accuracy (M2).
- Prometheus metrics, alerting and the detection-delay simulation (M3).
- Multivariate drift (correlations between features), concept drift without input drift.
- Streaming per-event checks: M1 works on batches (files).

## Design

```
training.csv + schema.json --drift reference--> reference.json (bins, counts, 5,000-value sample, checksum)
live/batch-NN.csv ----drift run (scheduled)----> reports/batch-NN.json + state.json   exit 1 if drift
```

- **Reference window** (`reference.py`): for numeric features, decile bin edges, the training counts
  per bin, the missing-value rate and a seeded sample of 5,000 values (sorted) for KS; for
  categorical features, the 50 most frequent categories and their counts, with everything else in
  `__other__`. Stored as JSON with a schema version and a SHA-256 over the content; loading refuses
  a bad checksum, unknown fields or inconsistent counts ([ADR 0003](../adr/0003-reference-as-bins-and-sample.md)).
- **Statistics** (`stats.py`, numpy only): PSI over the reference bins with a 1e-4 floor for empty
  bins; two-sample KS with the asymptotic p-value and Stephens' small-sample correction; chi-square
  test of homogeneity on the 2 x K table of counts with the regularized incomplete gamma for the
  p-value.
- **Decision** (`monitor.py`): drift if PSI ≥ 0.2, or the test is significant (p < 0.01) and PSI ≥
  0.1; warn if PSI ≥ 0.1 or the test is significant; a missing-value rate more than 5 points above
  the reference is drift; fewer than 200 values is "insufficient", never "ok"
  ([ADR 0002](../adr/0002-psi-and-significance-together.md)).
- **Scheduled job** (`jobs.py`): lists `live/*.csv`, skips files whose name and SHA-256 are already
  in `state.json`, writes each report and then the state atomically, and holds an exclusive lock
  file for the run. Cron or a CI schedule calls `drift run`; exit code 1 means drift.

## Alternatives considered

| Option | Why not (yet) |
|---|---|
| Evidently or another drift library | Complete and well tested, but a large dependency tree; M1 keeps the statistics small enough to read and checks them against SciPy. M3 compares results with Evidently. |
| PSI only | The common industry check, but it misses shifts that stay inside the deciles: a 15% income increase gave PSI 0.071 to 0.137 over 100 batches, below the 0.2 alarm level every time. |
| KS / chi-square p-values only | Sensitive, but with large batches tiny, harmless changes become significant; alone they cause alert fatigue. |
| Keep the full training data as the reference | Exact, but large, slow to load and a copy of possibly sensitive data; bins plus a 5,000-value sample are enough for these tests. |
| Wasserstein or Jensen-Shannon distance | Good effect sizes, but no standard thresholds; candidates for M3's comparison. |

## Measurement plan (Measure)

- `pytest`: statistics against SciPy, properties (PSI ≥ 0, symmetric, 0 for equal shares), the
  decision table, false alarms (300 stable batches) and power (0.5 standard deviation shift).
- `scripts/measure.py`: 200 batches of 2,000 rows per scenario (stable plus five injected drifts),
  counting batch verdicts and per-feature drift flags.
- M3: detection delay per drift type when drift starts mid-stream.

## Milestones

- **M1 (done):** RFC; reference window per feature; scheduled job with PSI, KS and chi-square.
- **M2:** prediction distributions and confidence; rolling accuracy when labels arrive.
- **M3:** Prometheus metrics and alerts; drift-injection simulation and detection-delay chart.

## Risks and open questions

- Thresholds (0.1, 0.2, alpha 0.01, 200 values) are conventions tuned on synthetic data; real
  features need their own false-alarm review.
- Testing six features at alpha 0.01 gives about a 6% chance per stable batch that some feature is
  significant (7 warn verdicts in 200 stable batches were measured); the PSI requirement keeps
  these at warn, not drift.
- Discrete features (counts) produce tied values; KS is conservative there.
- The demo data is synthetic; behaviour on a real dataset is not measured in M1.
