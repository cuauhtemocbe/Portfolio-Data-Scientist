---
title: Fail-Closed Trivy Pre-Push CVE Gate
status: in-progress
created: 2026-09-18
updated: 2026-09-18
issue: meta-projects#41
---

# Fail-Closed Trivy Pre-Push CVE Gate

## Objective

Add a `pre-push` git hook that runs Trivy's filesystem vulnerability scanner
and blocks the push if it finds any CRITICAL-severity, fixable CVE — and
also blocks the push if Trivy itself isn't installed. This is
Portfolio-Data-Scientist's slice of a fleet-wide rollout (meta-projects
issue #41) standardizing a local, fail-closed CVE gate across repos before
code leaves a developer's machine.

## Context

Portfolio-Data-Scientist is a notebooks repo (course material and data
science projects). Its only dependency manifest is
`Precipitación-Australia/requirements/requirements.txt`. The repo has no
hook infrastructure at all: no `.githooks/`, no `.husky/`, no
`.pre-commit-config.yaml`, no `Makefile`, and `core.hooksPath` is unset.
Nothing stops a developer from pushing a branch that pins a CRITICAL,
fixable vulnerable dependency.

Because nothing exists yet, this change creates both pieces of the standard
fleet layout: a versioned hooks directory (`.githooks/pre-push`) and a
one-time, explicit activation script (`scripts/install-hooks.sh`). The hook
is **not** active after a clone; a contributor opts in by running the
install script.

The hook calls the `trivy` CLI directly on the host. Trivy is not a project
dependency, so the hook must fail closed — block the push — if `trivy` isn't
found on `PATH`, rather than silently skipping the scan.

The one-time pre-activation scan
(`trivy fs . --scanners vuln --severity CRITICAL --ignore-unfixed`) was run
on this branch's base before the hook was added: 0 vulnerabilities, so the
first push after rollout is not blocked by pre-existing debt.

## Requirements

### Functional Requirements

- [ ] A new `.githooks/pre-push` hook runs on `git push` once
      `scripts/install-hooks.sh` has been run.
- [ ] The hook checks for `trivy` on `PATH`. If missing, it prints a message
      pointing to `.claude/skills/trivy-scan/setup.md` and exits non-zero,
      blocking the push (fail closed, not skip-if-missing).
- [ ] If `trivy` is present, the hook runs
      `trivy fs . --scanners vuln --severity CRITICAL --exit-code 1 --ignore-unfixed --quiet`
      against the repo root.
- [ ] If Trivy finds any CRITICAL-severity vulnerability with an available
      fix, the hook exits non-zero and the push is blocked.
- [ ] HIGH/MEDIUM findings and unfixed CRITICALs do not block.
- [ ] If Trivy finds no such vulnerability, the hook exits 0 and the push
      proceeds.
- [ ] A new `scripts/install-hooks.sh` sets `core.hooksPath` to `.githooks`
      and `chmod +x`-es its contents.
- [ ] `README.md` documents the opt-in install step
      (`bash scripts/install-hooks.sh`), the `trivy` prerequisite, the
      fail-closed behavior, and the CRITICAL+fixable-only threshold.

### Non-Functional Requirements

- [ ] Side-effect-free: the hook makes no commits or file changes; it only
      scans and may abort.
- [ ] Local-only: no CI workflow and no server-side gate.
- [ ] Explicit activation: nothing is wired to run on clone; the hook is
      inert until `scripts/install-hooks.sh` is run.
- [ ] The hook contains only the Trivy step; no other hook logic.
- [ ] Fail-closed by design: any condition that prevents the scan from
      running (missing binary) blocks the push rather than letting it
      through silently.

## Architecture

### Components

- `.githooks/pre-push` (new): bash script, `set -uo pipefail` (deliberately
  not `-e`, since the script's own `if` branches must reach their explicit
  `exit` statements). Two gated steps:
  1. `command -v trivy` check → exit 1 with a setup-doc pointer if absent.
  2. `trivy fs . --scanners vuln --severity CRITICAL --exit-code 1
     --ignore-unfixed --quiet` → a non-zero result exits 1; otherwise exit 0.
- `scripts/install-hooks.sh` (new): `git config core.hooksPath .githooks`
  plus `chmod +x .githooks/*`, then lists the active hooks.
- `README.md`: short "Git hooks" section.

### Data Model

N/A — no persisted state; this is git-hook tooling only.

### External Dependencies

- [Trivy](https://github.com/aquasecurity/trivy) CLI: must be present on
  the developer's `PATH`. Setup instructions live in
  `.claude/skills/trivy-scan/setup.md`.
  - Note: `.claude/` is not tracked on `main` yet; it arrives with the open
    skills-sync PR (#1). Until then the path in the hook's message resolves
    only in clones that have the skills locally. The hook block is kept
    identical to the fleet reference regardless.

## User Stories

Tracked at the fleet level in meta-projects issue #41. No repo-local GitHub
issue is created for this slice; this spec is the tracking artifact.

```gherkin
Feature: Local CVE gate at push time

  Scenario: A CRITICAL, fixable vulnerability blocks the push
    Given the pre-push hook is installed and active
    And a dependency has a CRITICAL CVE with a published fix
    When I run git push
    Then the hook exits non-zero and the push is aborted

  Scenario: A HIGH-severity or unfixed-CRITICAL finding does not block
    Given the pre-push hook is installed and active
    And a dependency has only a HIGH CVE, or a CRITICAL CVE with no fix
    When I run git push
    Then the Trivy step does not block

  Scenario: Trivy missing on the machine fails closed
    Given trivy is not on PATH
    When the pre-push hook runs
    Then the hook blocks the push with a message pointing at trivy setup

  Scenario: Installing the hook is explicit, not automatic
    Given .githooks/pre-push and scripts/install-hooks.sh are committed
    But core.hooksPath has not been set
    When a contributor clones the repo and pushes
    Then the hook does not run
    And running scripts/install-hooks.sh activates it for future pushes
```

## Testing Strategy

### Unit Tests

N/A — shell tooling, no application code.

### Integration Tests

Manual verification by direct hook invocation (a git hook, not app code with
a test harness):

- Hook on the real repo exits 0.
- `trivy` absent from `PATH`
  (`/usr/bin/env -i PATH=/nonexistent /bin/bash .githooks/pre-push`) exits 1
  with the "trivy not found" message.
- Throwaway manifest, outside the repo, pinning an old package with a
  CRITICAL+fixable CVE: hook exits 1 and names the package and CVE.
- Throwaway manifest with only a HIGH CVE, and one with only an unfixed
  finding: hook exits 0.
- Hook inactive until install: a `git push --dry-run` in a fresh clone does
  not run the hook; after `scripts/install-hooks.sh` it does.

### E2E Tests

N/A — direct hook invocation is the accepted verification standard for the
fleet rollout.

### Performance Tests

N/A. First run downloads the Trivy vulnerability DB; later runs use the
cache.

## Boundaries & Constraints

### In Scope

- `.githooks/pre-push` and `scripts/install-hooks.sh`.
- A short "Git hooks" section in `README.md`.
- This spec.

### Out of Scope

- Any hook logic other than the Trivy step.
- Changing the blocking threshold or scanners (CRITICAL+fixable, `vuln`
  only) — this replicates the fleet default.
- CI-side scanning.
- Auto-installing Trivy or auto-activating the hook on clone.
- Any tracking of `.claude/` or `CLAUDE.md` (untracked local copies here).

### Technical Constraints

- Hook is bash (`#!/usr/bin/env bash`), executable (mode 755), and committed
  under `.githooks/`.
- The hook contains only the Trivy step; no other hook logic.

## Success Criteria

- [ ] `.githooks/pre-push` and `scripts/install-hooks.sh` exist, mode 755.
- [ ] The hook's Trivy block is identical to the fleet reference.
- [ ] `trivy fs . --scanners vuln --severity CRITICAL --ignore-unfixed`
      against the repo is clean, and the hook exits 0 on it.
- [ ] Hook fails closed when `trivy` is missing, blocks on CRITICAL+fixable,
      passes on HIGH-only and unfixed-only.
- [ ] Hook does nothing before `scripts/install-hooks.sh` is run.
- [ ] README documents the opt-in install.

## Implementation Plan

No separate `-plan.md` — two small, fully specified files (see
Architecture). Fleet-level plan:
`meta-projects/specs/global/trivy-pre-push-fleet-rollout-plan.md`.
