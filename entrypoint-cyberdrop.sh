#!/bin/sh
# Combined entrypoint: starts SSH (if SHELL_USER/SHELL_PASSWORD are set), then
# behaves the same as the slim image's entrypoint — installs a crontab when
# CRON_SCHEDULE is non-empty, does an immediate run, and tails the cron log.
# cyberdrop-dl is preinstalled on PATH for interactive use over SSH.

set -eu

: "${STATE_DIR:=/state}"
mkdir -p "$STATE_DIR" /run/sshd
cd "$STATE_DIR"

if [ -n "${SHELL_USER:-}" ] && [ -n "${SHELL_PASSWORD:-}" ]; then
    if ! id -u "$SHELL_USER" >/dev/null 2>&1; then
        useradd -m -s /bin/bash "$SHELL_USER"
        usermod -aG sudo "$SHELL_USER"
    fi
    echo "$SHELL_USER:$SHELL_PASSWORD" | chpasswd
    /etc/init.d/ssh start
    echo "[entrypoint] SSH enabled for user '$SHELL_USER' on port 22"
elif [ -n "${SHELL_USER:-}" ] || [ -n "${SHELL_PASSWORD:-}" ]; then
    echo "[entrypoint] both SHELL_USER and SHELL_PASSWORD must be set together" >&2
    exit 2
fi

if [ "$#" -eq 0 ]; then
    set -- --recursive /videos
fi

QUOTED=""
for arg in "$@"; do
    esc=$(printf "%s" "$arg" | sed "s/'/'\\\\''/g")
    QUOTED="$QUOTED '$esc'"
done

RENAMER="/usr/local/bin/python /app/batinapapka.py$QUOTED"

if [ -n "${CRON_SCHEDULE:-}" ]; then
    if [ -z "${BRAVE_API_KEY:-}" ]; then
        echo "[entrypoint] BRAVE_API_KEY must be set when CRON_SCHEDULE is set" >&2
        exit 2
    fi

    {
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
