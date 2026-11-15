"""Bound managed job logs without touching queue data, config or downloads."""
import os
from pathlib import Path
import re
import time

JOB_ID = re.compile(r'[0-9a-f]{32}')


def trim_append_file(path, limit):
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= limit:
        return
    # Native CSV writers reopen in append mode for every record.
    with path.open('r+b') as stream:
        stream.seek(-limit, os.SEEK_END)
        data = stream.read(limit)
        newline = data.find(b'\n')
        if newline != -1:
            data = data[newline + 1:]
        stream.seek(0)
        stream.write(data)
        stream.truncate()


def clean_job_logs(jobs, active_ids=(), *, max_age_days=30, max_total_bytes=200*1024*1024):
    """Prune only recognized log paths. Active job output stays protected."""
    jobs = Path(jobs)
    active_ids = set(active_ids)
    now = time.time()
    files = []
    for path in jobs.iterdir():
        if path.is_symlink():
            continue
        ident = path.name.split('.', 1)[0]
        if not JOB_ID.fullmatch(ident):
            continue
        if path.is_dir():
            # Cyberdrop-owned diagnostic logs, never AppData or video paths.
            paths = [p for p in path.rglob('*') if p.is_file() and not p.is_symlink() and
                     not any(parent.is_symlink() for parent in p.parents if parent.is_relative_to(path)) and
                     p.suffix in ('.log', '.csv', '.jsonl')]
        elif path.name in (ident+'.log', ident+'.log.1'):
            paths = [path]
        else:
            continue
        for file in paths:
            try:
                if ident in active_ids and file.suffix == '.csv':
                    trim_append_file(file, 1024*1024)
                info = file.stat()
                if ident not in active_ids and now-info.st_mtime > max_age_days*86400:
                    file.unlink()
                else:
                    files.append((info.st_mtime, file, info.st_size, ident in active_ids))
            except FileNotFoundError:
                continue
    total = sum(size for _, _, size, _ in files)
    for _, path, size, active in sorted(files, key=lambda entry:entry[0]):
        if total <= max_total_bytes:
            break
        if not active:
            try:
                path.unlink()
                total -= size
            except FileNotFoundError:
                pass
