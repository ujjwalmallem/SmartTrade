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
    *) echo "usage: run_paper_trading.sh <gex|er> <open|check|close|report|score>"; exit 1 ;;
esac

case "$ACTION" in
    open|check|close) ;;
    report|score)
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        python3 -m "$MODULE" "$ACTION"
        exit 0
        ;;
    *) echo "usage: run_paper_trading.sh <gex|er> <open|check|close|report|score>"; exit 1 ;;
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

    # The gex and er workflows run on the same cron schedule and both push to
    # develop, so a rejected push (someone else's commit landed first) is the
    # expected case, not an error -- rebase onto the latest and retry. Each
    # source only ever touches its own ledger file, so this never conflicts.
    attempt=1
    max_attempts=5
    until git push; do
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "[run_paper_trading] push failed after ${max_attempts} attempts" >&2
            exit 1
        fi
        echo "[run_paper_trading] push rejected (attempt ${attempt}/${max_attempts}); fetching and rebasing..."
        sleep "$(( (RANDOM % 5) + 1 ))"
        git fetch origin develop
        git rebase origin/develop
        attempt=$((attempt + 1))
    done
else
    echo "[run_paper_trading] ${LEDGER} unchanged, nothing to commit."
fi
