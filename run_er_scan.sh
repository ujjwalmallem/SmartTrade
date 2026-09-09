#!/usr/bin/env bash
# Wrapper for cron: sets up the environment and runs the ER dashboard.
# Edit NTFY_TOPIC below to the topic you subscribed to in the ntfy app.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export NTFY_TOPIC="smarttrade-er-7bfbe66211d7"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 er_dashboard.py
