#!/bin/sh
set -eu
: "${STATE_DIR:=/state}"
export STATE_DIR
mkdir -p "$STATE_DIR"
cd "$STATE_DIR"
if [ "$#" -eq 0 ]; then set -- --recursive /videos; fi
if [ -n "${CRON_SCHEDULE:-}" ]; then
    /usr/local/bin/python /app/configure_cron.py "$@"
    touch "$STATE_DIR/cron.log"
    /usr/local/bin/run-scheduled-renamer || true
    cron
    exec tail -F "$STATE_DIR/cron.log"
fi
exec /usr/local/bin/python /app/batinapapka.py "$@"
