#!/bin/sh
# Combined entrypoint: starts SSH (if SHELL_USER/SHELL_PASSWORD are set), then
# behaves the same as the slim image's entrypoint — installs a crontab when
# CRON_SCHEDULE is non-empty, does an immediate run, and tails the cron log.
# cyberdrop-dl is preinstalled on PATH for interactive use over SSH.

set -eu

: "${STATE_DIR:=/state}"
mkdir -p "$STATE_DIR" /run/sshd /videos
cd "$STATE_DIR"

# Point default download folders at the mounted /videos. cyberdrop-dl writes
# to ~/Downloads by default; aliasing it via symlink means downloads land in
# /videos with no extra flags, and the renamer cron job then picks them up.
#
# IMPORTANT: never `rm -rf` the target — if it already is a symlink to
# /videos, recursive rm would delete the contents of the mounted volume.
link_downloads() {
    target_home="$1"
    [ -d "$target_home" ] || return 0
    dl="$target_home/Downloads"
    # If there's already a real (non-symlink) directory there, leave it alone:
    # the user may have set it up deliberately.
    if [ -d "$dl" ] && [ ! -L "$dl" ]; then
        return 0
    fi
    rm -f "$dl"
    ln -sfn /videos "$dl"
    [ -n "${2:-}" ] && chown -h "$2:$2" "$dl" 2>/dev/null || true
}

# Make sure SSH host keys exist on slim images where the postinst hook
# may not have created them yet.
[ -f /etc/ssh/ssh_host_ed25519_key ] || ssh-keygen -A >/dev/null

link_downloads "/root"

if [ -n "${SHELL_USER:-}" ] && [ -n "${SHELL_PASSWORD:-}" ]; then
    if ! id -u "$SHELL_USER" >/dev/null 2>&1; then
        useradd -m -s /bin/bash "$SHELL_USER"
        usermod -aG sudo "$SHELL_USER"
    fi
    echo "$SHELL_USER:$SHELL_PASSWORD" | chpasswd
    link_downloads "/home/$SHELL_USER" "$SHELL_USER"
    /etc/init.d/ssh start
    echo "[entrypoint] SSH on :22 for user '$SHELL_USER'. cyberdrop-dl,"
    echo "[entrypoint] tmux and screen are on PATH. ~/Downloads -> /videos."
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
