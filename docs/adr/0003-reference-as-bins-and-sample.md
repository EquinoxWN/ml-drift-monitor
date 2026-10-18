# ADR 0003: Store the reference window as bins, counts and a fixed sample

- **Status:** Accepted

## Context

Every check compares a live batch with the training data. Keeping the whole training set as the
reference is exact but heavy (it can be gigabytes), slow to load in every scheduled run, and a
second copy of data that may be sensitive. PSI needs only bin counts; chi-square needs only
category counts; KS needs a sample of raw values.

## Decision

`drift reference` profiles the training data once:

- numeric: decile bin edges (duplicate edges merged, so discrete features get fewer bins), the
  count per bin, the missing-value rate, and a seeded random sample of 5,000 values, sorted;
- categorical: the 50 most frequent categories with their counts, and one `__other__` count for
  everything else (live categories never seen in training land there too).

The profile is stored as JSON with a schema version, the training file's name and SHA-256, and a
SHA-256 over the canonical content. Loading checks all of them and refuses unknown fields and
inconsistent counts.

## Consequences

- The demo reference for 20,000 rows and six features is a 242 KB JSON file, mostly the
  5,000-value samples; the training data is not needed after profiling.
- KS compares the live batch with a 5,000-value sample instead of all training values; the KS
  p-value uses both sample sizes, so it is correct for what is compared, with slightly less power
  than using all rows.
- PSI is limited to the decile resolution: shifts that stay within bins are invisible to PSI, which
  is one reason ADR 0002 pairs it with KS.
- A changed or corrupted reference is detected before any check runs, and the training file's
  SHA-256 records exactly which data the reference came from.
- Reference windows are immutable files; refreshing the reference (for example after retraining)
  means building a new one, which is an explicit step.
