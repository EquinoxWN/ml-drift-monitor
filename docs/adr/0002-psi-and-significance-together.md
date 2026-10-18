# ADR 0002: Decide drift from PSI and a significance test together

- **Status:** Accepted

## Context

Two families of drift checks are common:

- **PSI** (Population Stability Index) measures how much the share of data in each bin changed.
  The industry convention is "below 0.1 stable, 0.1 to 0.2 moderate, above 0.2 significant". It
  does not depend on batch size, but it only sees changes that move data between bins.
- **Hypothesis tests** (two-sample KS for numeric features, chi-square for categories) give a
  p-value. They see any change in shape, but with large batches they flag harmless differences,
  and testing many features at once multiplies false alarms.

Measured on the synthetic data (200 batches of 2,000 rows each):

- A 15% income increase: KS p-value below 1e-10 in every batch, but PSI between 0.071 and 0.137
  (100 batches), never above 0.2. PSI alone would never alarm.
- Stable data: 7 of 200 batches had at least one feature with p < 0.01 across six features. A test
  alone would raise an alarm on about 3.5% of stable batches.

## Decision

`monitor.decide` uses both:

- **drift** when PSI ≥ 0.2, or when the test is significant (p < 0.01) and PSI ≥ 0.1;
- **warn** when PSI ≥ 0.1 or the test is significant;
- **ok** otherwise.

Missing values get their own rule (drift when the missing rate rises by more than 5 points), and
batches with fewer than 200 values are "insufficient" rather than "ok".

## Consequences

- 0 drift verdicts in 200 stable batches; the 7 significant-only batches show up as warn.
- Large shifts (category mix, spread, new category, missing values) are drift in 200 of 200 batches.
- The 15% income shift is split: drift in 102 of 200 batches, warn in the other 98, never ok.
  Teams that must catch it every time should alert on warn for that feature or lower its PSI band.
- Every verdict lists its reasons (`PSI 0.237 >= 0.2; p 1.9e-86 < 0.01`), so an alert can be
  explained without rerunning anything.
- The thresholds are configuration (`Thresholds`), not code; M3's simulation will measure the delay
  each choice costs.
