#!/usr/bin/env bash
# Keep an unattended sweep alive. If the python process dies without writing its completion marker
# (session teardown, OOM, transient fault), restart it with --resume, which picks up at the first
# scenario missing from results/performance_curves.csv. Bounded retries so a genuine, repeatable
# crash surfaces instead of looping forever.
cd "$(dirname "$0")/.." || exit 1
LOG=sweep.log
DONE="wrote results/REPORT.md"
for attempt in $(seq 1 12); do
    if grep -q "$DONE" "$LOG" 2>/dev/null; then
        echo "[supervisor] sweep complete after $attempt attempt(s)" >> "$LOG"; exit 0
    fi
    if pgrep -f "analysis/run[.]py --max" > /dev/null; then
        sleep 60; continue
    fi
    echo "[supervisor] $(date -Is): run not active, (re)starting with --resume (attempt $attempt)" >> "$LOG"
    python3 analysis/run.py --max --keep-traces --resume >> "$LOG" 2>&1
done
echo "[supervisor] giving up after 12 attempts" >> "$LOG"
