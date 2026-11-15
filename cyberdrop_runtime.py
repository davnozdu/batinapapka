"""Update Cyberdrop in an isolated persistent venv; retain working releases."""
import fcntl
from contextlib import closing
import json
import logging
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time
import urllib.request
import uuid

LOG = logging.getLogger(__name__)


def run_checked(args, timeout):
    # Inherit the process group so queue cancellation also stops pip/build children.
    subprocess.run(args, stdin=subprocess.DEVNULL, check=True, timeout=timeout)


def executable(root):
    current = root / 'current' / 'bin' / 'cyberdrop-dl'
    return str(current if current.exists() else Path(os.environ.get('CYBERDROP_BUNDLED', '/opt/cyberdrop/venv/bin/cyberdrop-dl')))


def update(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.update.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status_path = root / 'status.json'
        try:
            status = json.loads(status_path.read_text())
        except (OSError, ValueError):
            status = {}
        interval = max(300, int(os.environ.get('CYBERDROP_UPDATE_INTERVAL', '86400')))
        if time.time() - status.get('checked', 0) < interval:
            return
        version = status.get('version')
        try:
            if not version:
                version = subprocess.check_output([executable(root), '--version'], timeout=30, text=True).strip()
            if os.environ.get('CYBERDROP_AUTO_UPDATE', '1').lower() not in ('0', 'false', 'no'):
                with urllib.request.urlopen('https://pypi.org/pypi/cyberdrop-dl-patched/json', timeout=20) as response:
                    latest = json.load(response)['info']['version']
                if latest not in version.split():
                    target = root / f'venv-{latest}-{uuid.uuid4().hex[:8]}'
                    run_checked([sys.executable, '-m', 'venv', str(target)], 120)
                    run_checked([str(target / 'bin/pip'), 'install', '--disable-pip-version-check',
                                 '--no-cache-dir', '--upgrade', 'pip'], 180)
                    run_checked([str(target / 'bin/pip'), 'install', '--disable-pip-version-check',
                                 '--no-cache-dir', f'cyberdrop-dl-patched=={latest}'], 600)
                    run_checked([str(target / 'bin/pip'), 'check'], 30)
                    candidate = str(target / 'bin/cyberdrop-dl')
                    run_checked([candidate, '--version'], 30)
                    help_text = subprocess.check_output([candidate, 'download', '--help'], timeout=30, text=True)
                    for flag in ('--input-file', '--download-folder', '--database-file', '--ui', '--logs.console-level', '--log-file', '--logs.no-rotate', '--no-dump-json', '--no-dump-responses', '--logs.no-http-traffic'):
                        if flag not in help_text:
                            raise RuntimeError(f'New Cyberdrop version does not support {flag}')
                    link = root / f'.current-{uuid.uuid4().hex}'
                    link.symlink_to(target.name, target_is_directory=True)
                    os.replace(link, root / 'current')
                    version = latest
            message = f'Cyberdrop: {version} · автоматическое обновление перед загрузкой (интервал {interval} с)'
        except Exception as exc:
            LOG.exception('Update failed; retaining working Cyberdrop')
            message = f'Обновление Cyberdrop не удалось; сохранена рабочая версия {version or "из образа"}: {exc}'
        temp = root / '.status.tmp'
        temp.write_text(json.dumps({'checked': time.time(), 'version': version, 'message': message}, ensure_ascii=False))
        os.replace(temp, status_path)


def preserve_legacy_database(state):
    legacy = state / 'cyberdrop/AppData/Cache/cyberdrop.db'
    current = state / 'cyberdrop/cyberdrop.db'
    if not legacy.is_file() or legacy.stat().st_size == 0 or (current.exists() and current.stat().st_size):
        return
    # SQLite backup preserves committed WAL data too; never move/delete legacy state.
    from urllib.parse import quote
    temp = current.with_name('.legacy-db-' + uuid.uuid4().hex)
    try:
        with closing(sqlite3.connect('file:' + quote(str(legacy)) + '?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(temp)) as dest:
                source.backup(dest)
        os.replace(temp, current)
        LOG.info('Copied legacy Cyberdrop history into V10; original database retained')
    except (OSError, sqlite3.Error):
        LOG.exception('Legacy history copy failed; original database retained')
    finally:
        temp.unlink(missing_ok=True)


def main():
    logging.basicConfig(level=logging.INFO)
    state = Path(os.environ.get('STATE_DIR', '/state')).resolve()
    root = state / 'cyberdrop' / 'runtime'
    # Also serialize manual SSH downloads with the web worker and renamer.
    lock = None
    if os.environ.get('CYBERDROP_QUEUE_LOCKED') != '1':
        lock = (state / '.batinapapka.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.set_inheritable(lock.fileno(), True)
    # V10 scans cwd/AppData even with an explicit --config-file. Never start
    # inside the persistent legacy AppData tree or an SSH home that links to it.
    work = state / 'cyberdrop' / 'work-v10'
    work.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    # Resolve user paths before changing cwd; queue paths are already absolute.
    path_flags = {'--input-file','--input','-i','--input-folder','--config-file','--config','-c',
                  '--database-file','--db','--cache-file','--download-folder','--output','-o','-d',
                  '--cookies','--logs.folder','--log-file','--logs.files.main'}
    for index, arg in enumerate(args):
        name, separator, value = arg.partition('=')
        if name in path_flags:
            if separator:
                args[index] = name + '=' + os.path.abspath(os.path.expanduser(value))
            elif index + 1 < len(args):
                args[index+1] = os.path.abspath(os.path.expanduser(args[index+1]))
        elif arg.lower().endswith(('.txt','.html')) and not arg.startswith(('http://','https://')):
            args[index] = os.path.abspath(os.path.expanduser(arg))
    preserve_legacy_database(state)
    os.chdir(work)
    # V10 also uses XDG paths; persist them for interactive SSH sessions.
    for key, folder in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'), ('XDG_CACHE_HOME', 'cache'), ('XDG_STATE_HOME', 'state')]:
        os.environ[key] = str(state / 'cyberdrop' / ('xdg-' + folder))
    update(root)
    if args == ['--update-only']:
        return 0
    if args and args[0] == 'download':
        defaults = [('--config-file', '--config', '-c', str(state / 'cyberdrop/config.yaml')),
                    ('--database-file', '--db', str(state / 'cyberdrop/cyberdrop.db')),
                    ('--cache-file', str(state / 'cyberdrop/cache.json')),
                    ('--download-folder', '--output', '-o', '-d', os.environ.get('DOWNLOAD_DIR', '/videos'))]
        for *flags, value in defaults:
            if not any(arg.split('=', 1)[0] in flags for arg in args):
                if flags[0] in ('--config-file', '--database-file', '--cache-file'):
                    Path(value).touch(exist_ok=True)
                args.extend([flags[0], value])
    os.execv(executable(root), [executable(root), *args])


if __name__ == '__main__':
    sys.exit(main())
