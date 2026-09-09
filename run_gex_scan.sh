#!/usr/bin/env bash
# Wrapper for cron: sets up the environment and runs the GEX scanner.
# Edit NTFY_TOPIC below to the topic you subscribed to in the ntfy app.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NTFY_TOPIC="smarttrade-gex-bb4b87815d16"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 gex_scanner.py
