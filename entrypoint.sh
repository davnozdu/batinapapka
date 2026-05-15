#!/bin/sh
# Container entrypoint. Behaviour:
#   - CMD args (default: "--recursive /videos") are passed straight to the
#     renamer, so the image runs out of the box with no extra wiring.
#   - If CRON_SCHEDULE is non-empty, the same command is also installed as a
#     cron job, an immediate run is performed once, then we hand off to cron
#     and tail the log. Drop CRON_SCHEDULE="" to make the container a
#     one-shot run instead.
#   - State (cache, renamed-files index, log) lives under STATE_DIR (/state
#     by default), so it survives container recreates when /state is a
#     persistent volume.

set -eu

: "${STATE_DIR:=/state}"
mkdir -p "$STATE_DIR"
cd "$STATE_DIR"

if [ "$#" -eq 0 ]; then
    set -- --recursive /videos
fi

# Quote each arg for the cron command line so paths with spaces work.
QUOTED=""
for arg in "$@"; do
    esc=$(printf "%s" "$arg" | sed "s/'/'\\\\''/g")
    QUOTED="$QUOTED '$esc'"
done

RENAMER="/usr/local/bin/python /app/batinapapka.py$QUOTED"

if [ -n "${CRON_SCHEDULE:-}" ]; then
    if [ -z "${BRAVE_API_KEY:-}" ]; then
        echo "[entrypoint] BRAVE_API_KEY must be set" >&2
        exit 2
    fi

    {
        # Cron strips the parent environment; export only what the renamer needs.
        echo "BRAVE_API_KEY=${BRAVE_API_KEY}"
        echo "${CRON_SCHEDULE} root cd ${STATE_DIR} && ${RENAMER} >> ${STATE_DIR}/cron.log 2>&1"
    } > /etc/cron.d/batinapapka
    chmod 0644 /etc/cron.d/batinapapka
    touch "${STATE_DIR}/cron.log"

    echo "[entrypoint] one-shot run (cron schedule: ${CRON_SCHEDULE})"
    sh -c "$RENAMER" || true

    cron
    exec tail -F "${STATE_DIR}/cron.log"
else
    echo "[entrypoint] one-shot mode (no CRON_SCHEDULE)"
    exec sh -c "$RENAMER"
fi
