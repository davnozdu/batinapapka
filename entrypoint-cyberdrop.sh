#!/bin/sh
# Combined entrypoint: starts SSH (if SHELL_USER/SHELL_PASSWORD are set), then
# behaves the same as the slim image's entrypoint — installs a crontab when
# CRON_SCHEDULE is non-empty, does an immediate run, and tails the cron log.
# cyberdrop-dl is preinstalled on PATH for interactive use over SSH.

set -eu

: "${STATE_DIR:=/state}"
mkdir -p "$STATE_DIR" /run/sshd /videos
cd "$STATE_DIR"

# Persistent location for everything cyberdrop-dl and the renamer must keep
# across container recreates: AppData/Configs, AppData/Cache, URLs.txt.
CYBERDROP_STATE="${STATE_DIR}/cyberdrop"
mkdir -p "$CYBERDROP_STATE/AppData"
[ -f "$CYBERDROP_STATE/URLs.txt" ] || touch "$CYBERDROP_STATE/URLs.txt"

# Point default download folders at the mounted /videos. cyberdrop-dl writes
# to ~/Downloads by default; aliasing it via symlink means downloads land in
# /videos with no extra flags, and the renamer cron job then picks them up.
#
# IMPORTANT: never `rm -rf` a target — if it already is a symlink to /videos,
# recursive rm would delete the contents of the mounted volume.
link_downloads() {
    target_home="$1"
    [ -d "$target_home" ] || return 0
    dl="$target_home/Downloads"
    if [ -d "$dl" ] && [ ! -L "$dl" ]; then
        return 0
    fi
    rm -f "$dl"
    ln -sfn /videos "$dl"
    [ -n "${2:-}" ] && chown -h "$2:$2" "$dl" 2>/dev/null || true
}

# Symlink cyberdrop's own state into /state/cyberdrop so configs, the
# settings DB and URLs.txt survive `docker compose down && up`. If a fresh
# install has written real files into ~/AppData or ~/URLs.txt before the
# symlinks existed, we migrate them once into /state/cyberdrop and then
# replace them with the symlink.
link_cyberdrop_state() {
    target_home="$1"
    owner="${2:-}"
    [ -d "$target_home" ] || return 0

    src_appdata="$target_home/AppData"
    if [ -d "$src_appdata" ] && [ ! -L "$src_appdata" ]; then
        # Real dir present and persistent slot empty → adopt it.
        if [ -z "$(ls -A "$CYBERDROP_STATE/AppData" 2>/dev/null)" ]; then
            rmdir "$CYBERDROP_STATE/AppData" 2>/dev/null || true
            mv "$src_appdata" "$CYBERDROP_STATE/AppData"
        fi
    fi
    rm -f "$src_appdata"
    ln -sfn "$CYBERDROP_STATE/AppData" "$src_appdata"

    src_urls="$target_home/URLs.txt"
    if [ -f "$src_urls" ] && [ ! -L "$src_urls" ]; then
        if [ ! -s "$CYBERDROP_STATE/URLs.txt" ]; then
            mv "$src_urls" "$CYBERDROP_STATE/URLs.txt"
        fi
    fi
    rm -f "$src_urls"
    ln -sfn "$CYBERDROP_STATE/URLs.txt" "$src_urls"

    if [ -n "$owner" ]; then
        chown -h "$owner:$owner" "$src_appdata" "$src_urls" 2>/dev/null || true
        # The data itself needs to be writable by cyberdrop-dl, which runs as
        # SHELL_USER over SSH; chown the persistent tree to the same uid.
        chown -R "$owner:$owner" "$CYBERDROP_STATE" 2>/dev/null || true
    fi
}

# Make sure SSH host keys exist on slim images where the postinst hook
# may not have created them yet.
[ -f /etc/ssh/ssh_host_ed25519_key ] || ssh-keygen -A >/dev/null

link_downloads "/root"
link_cyberdrop_state "/root"

if [ -n "${SHELL_USER:-}" ] && [ -n "${SHELL_PASSWORD:-}" ]; then
    if ! id -u "$SHELL_USER" >/dev/null 2>&1; then
        useradd -m -s /bin/bash "$SHELL_USER"
        usermod -aG sudo "$SHELL_USER"
    fi
    echo "$SHELL_USER:$SHELL_PASSWORD" | chpasswd
    link_downloads "/home/$SHELL_USER" "$SHELL_USER"
    link_cyberdrop_state "/home/$SHELL_USER" "$SHELL_USER"
    /etc/init.d/ssh start
    echo "[entrypoint] SSH on :22 for user '$SHELL_USER'."
    echo "[entrypoint]   ~/Downloads  -> /videos"
    echo "[entrypoint]   ~/AppData    -> ${CYBERDROP_STATE}/AppData (persistent)"
    echo "[entrypoint]   ~/URLs.txt   -> ${CYBERDROP_STATE}/URLs.txt (persistent)"
    echo "[entrypoint] cyberdrop-dl, tmux, screen are on PATH."
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
