#!/bin/sh
# Persistent browser queue, optional SSH, and scheduled renaming.
set -eu
: "${STATE_DIR:=/state}"
export STATE_DIR
mkdir -p "$STATE_DIR/cyberdrop/AppData" /run/sshd /videos
cd "$STATE_DIR"

if [ -n "${SHELL_USER:-}" ] || [ -n "${SHELL_PASSWORD:-}" ]; then
    : "${SHELL_USER:?SHELL_USER and SHELL_PASSWORD must both be set}"
    : "${SHELL_PASSWORD:?SHELL_USER and SHELL_PASSWORD must both be set}"
    case "$SHELL_USER" in root|*[!a-z0-9_-]*|-*|[0-9]*|'') echo 'Use a non-root Unix SHELL_USER' >&2; exit 2;; esac
    if ! id -u "$SHELL_USER" >/dev/null 2>&1; then
        useradd -m -s /bin/bash "$SHELL_USER"
    fi
    /usr/local/bin/python -c 'import os; value=os.environ["SHELL_PASSWORD"]; assert "\n" not in value and "\r" not in value, "SHELL_PASSWORD must be a single line"'
    printf '%s:%s\n' "$SHELL_USER" "$SHELL_PASSWORD" | chpasswd
    chown -R "$SHELL_USER:$SHELL_USER" "$STATE_DIR"
    chown "$SHELL_USER:$SHELL_USER" /videos
fi

# Preserve conflicting legacy home files; never remove a real directory.
/usr/local/bin/python - <<'PY'
import os, pathlib, pwd, shlex
state = pathlib.Path(os.environ['STATE_DIR'])
root = state / 'cyberdrop'
(root / 'URLs.txt').touch(exist_ok=True)
users = ['root'] + ([os.environ['SHELL_USER']] if os.environ.get('SHELL_USER') else [])
for user in users:
    account = pwd.getpwnam(user)
    home = pathlib.Path(account.pw_dir)
    for name, dest in [('Downloads', pathlib.Path('/videos')), ('AppData', root/'AppData'), ('URLs.txt', root/'URLs.txt')]:
        src = home / name
        if src.is_symlink():
            src.unlink()
        elif src.exists():
            # Adopt old files only if the persistent slot is empty, otherwise back up.
            empty = dest.is_dir() and not any(dest.iterdir()) or dest.is_file() and dest.stat().st_size == 0
            if name != 'Downloads' and empty:
                dest.rmdir() if dest.is_dir() else dest.unlink()
                src.rename(dest)
            else:
                backup = home / (name + '.before-persistent')
                if backup.exists():
                    continue
                src.rename(backup)
        src.symlink_to(dest, target_is_directory=dest.is_dir())
        os.lchown(src, account.pw_uid, account.pw_gid)
# SSH/PAM environment; no sed interpolation of credentials.
path = pathlib.Path('/etc/environment')
values = {k: os.environ.get(k, '') for k in ('BRAVE_API_KEY','TPDB_API_KEY','STATE_DIR','CYBERDROP_AUTO_UPDATE','CYBERDROP_UPDATE_INTERVAL','CYBERDROP_BUNDLED_VERSION')}
lines = [line for line in path.read_text().splitlines() if line.split('=',1)[0] not in values]
for key, value in values.items():
    if '\n' in value or '\r' in value or '"' in value or '\\' in value:
        raise ValueError(f'Unsupported characters in {key}')
    lines.append(f'{key}="{value}"')
path.write_text('\n'.join(lines)+'\n')
PY

if [ -n "${SHELL_USER:-}" ]; then
    # Host identities belong to the installation, not to the shared image.
    mkdir -p "$STATE_DIR/ssh"
    chmod 0700 "$STATE_DIR/ssh"
    for kind in ed25519 rsa; do
        key="$STATE_DIR/ssh/ssh_host_${kind}_key"
        [ -f "$key" ] || ssh-keygen -q -t "$kind" -N '' -f "$key"
    done
    cat > /etc/ssh/sshd_config.d/batinapapka.conf <<EOF
PermitRootLogin no
AllowUsers $SHELL_USER
HostKey $STATE_DIR/ssh/ssh_host_ed25519_key
HostKey $STATE_DIR/ssh/ssh_host_rsa_key
EOF
    chown -R root:root "$STATE_DIR/ssh"
    chown -R "$SHELL_USER:$SHELL_USER" "$STATE_DIR/cyberdrop"
    /usr/sbin/sshd
fi

/usr/local/bin/python /app/configure_cron.py "$@"
if [ -f /etc/cron.d/batinapapka ]; then
    cron
    # Don't delay the web UI while a potentially long first pass runs.
    if [ -n "${SHELL_USER:-}" ]; then
        setpriv --reuid="$SHELL_USER" --regid="$SHELL_USER" --init-groups /usr/local/bin/run-scheduled-renamer &
    else
        /usr/local/bin/run-scheduled-renamer &
    fi
fi

# The combined image always stays alive for the browser/SSH, even with cron disabled.
if [ -n "${SHELL_USER:-}" ]; then
    exec setpriv --reuid="$SHELL_USER" --regid="$SHELL_USER" --init-groups /usr/local/bin/python /app/cyberdrop_admin.py
fi
exec /usr/local/bin/python /app/cyberdrop_admin.py
