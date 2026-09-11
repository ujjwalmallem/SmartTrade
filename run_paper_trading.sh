#!/usr/bin/env bash
# Runs one paper-trading action (open/check/close) for one source (gex/er),
# then commits the updated ledger back to git if it changed. Ledger state
# only persists across GitHub Actions runs if it's committed -- each run
# starts from a fresh checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOURCE="${1:-}"
ACTION="${2:-}"

case "$SOURCE" in
    gex) export NTFY_TOPIC="smarttrade-gex-bb4b87815d16"; MODULE="paper_trading.trade_gex" ;;
    er)  export NTFY_TOPIC="smarttrade-er-7bfbe66211d7";  MODULE="paper_trading.trade_er" ;;
    *) echo "usage: run_paper_trading.sh <gex|er> <open|check|close>"; exit 1 ;;
esac

case "$ACTION" in
    open|check|close) ;;
    *) echo "usage: run_paper_trading.sh <gex|er> <open|check|close>"; exit 1 ;;
esac

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 -m "$MODULE" "$ACTION"

LEDGER="paper_trading/ledger_${SOURCE}.json"
if [ -n "$(git status --porcelain -- "$LEDGER" 2>/dev/null)" ]; then
    git config user.name "paper-trading-bot"
    git config user.email "paper-trading-bot@users.noreply.github.com"
    git add "$LEDGER"
    git commit -m "Paper trading (${SOURCE}): ${ACTION} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git push
else
    echo "[run_paper_trading] ${LEDGER} unchanged, nothing to commit."
fi
