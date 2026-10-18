# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub:
**Security > Report a vulnerability** on this repository
(`https://github.com/EquinoxWN/ml-drift-monitor/security/advisories/new`). Do not open a public issue.

Include what you found, how to reproduce it, and the impact you expect. You will get an answer
within 7 days. Fixes are released as soon as they are ready, and you are credited unless you ask
not to be.

## Supported versions

This project is pre-1.0. Only the latest commit on `main` receives fixes.

## Scope

- **In scope:** crafted CSV or reference files that crash the checker, use unbounded memory or
  time, or make it read or write outside the folders it was given; a modified reference file being
  accepted; two scheduled runs overlapping.
- **Out of scope:** drift verdicts on your data that you disagree with (thresholds are
  configuration), and the synthetic demo data.

## How this repository protects itself

- CSV input: size cap (256 MiB), row cap (5 million), explicit schema, missing columns refused,
  non-finite numbers refused, category values truncated to 64 characters.
- Reference files: size cap (32 MiB), schema version, SHA-256 over the content, unknown fields and
  inconsistent counts refused. JSON only; nothing is unpickled or evaluated.
- The scheduled job writes reports and state atomically and holds an exclusive lock per run.
- Runtime dependency: numpy only. SciPy is a test oracle and is not installed at runtime.
- Every GitHub Action is pinned to a full commit SHA, and workflows run with read-only
  permissions and without persisted credentials.
- Dependabot update pull requests are paused (`open-pull-requests-limit: 0`) so the repository keeps a single `main` branch; dependencies are updated by hand and checked by the CI audit.
- CI runs lint, strict type checks, tests and a known-vulnerability check (pip-audit) on every push and pull request.
