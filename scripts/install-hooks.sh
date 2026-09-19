#!/usr/bin/env bash
# One-time activation for this repo's versioned git hooks (.githooks/).
# Explicit, not automatic on clone — this repo has no hook infrastructure
# otherwise, and nothing should be wired to run without an explicit trigger.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git -C "$REPO" config core.hooksPath .githooks
chmod +x "$REPO"/.githooks/*

echo "Installed: core.hooksPath -> .githooks"
echo "Active hooks:"
for hook in "$REPO"/.githooks/*; do
  echo "  - $(basename "$hook")"
done
