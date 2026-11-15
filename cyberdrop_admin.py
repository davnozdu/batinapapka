"""Persistent, single-worker Cyberdrop queue and authenticated browser UI."""
from __future__ import annotations

from contextlib import contextmanager
import codecs
import fcntl
import hmac
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

from flask import Flask, abort, jsonify, request, send_from_directory

LOG = logging.getLogger(__name__)
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


class Queue:
    def __init__(self, state: Path):
        self.state = state
        self.root = state / 'cyberdrop'
        self.jobs = self.root / 'jobs'
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'queue.db'
        self.stop = threading.Event()
        self.proc = None
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, urls TEXT NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, started REAL, finished REAL, exit_code INTEGER)''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='queued', started=NULL WHERE status='running'")
            db.execute("UPDATE jobs SET status='cancelled', finished=? WHERE status='cancelling'", (time.time(),))

    def add(self, text):
        urls = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                parsed = urlsplit(line)
                valid = parsed.scheme in ('http', 'https') and parsed.hostname and not parsed.username
            except ValueError:
                valid = False
            if not valid or any(c.isspace() or ord(c) < 32 for c in line) or len(line) > 8192:
                raise ValueError('Каждая строка должна содержать одну HTTP/HTTPS ссылку.')
            if line not in urls:
                urls.append(line)
        if not urls or len(urls) > 1000:
            raise ValueError('Добавьте от 1 до 1000 ссылок.')
        ident = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('INSERT INTO jobs(id,urls,status,created) VALUES(?,?,?,?)',
                       (ident, json.dumps(urls), 'queued', time.time()))
        return ident

    def list(self):
        with self.connect() as db:
            return [dict(r) | {'urls': json.loads(r['urls'])} for r in db.execute(
                'SELECT * FROM jobs ORDER BY created DESC LIMIT 200')]

    def action(self, ident, action):
        with self.connect() as db:
            if action == 'cancel':
                result = db.execute("""UPDATE jobs SET status=CASE WHEN status='running'
                    THEN 'cancelling' ELSE 'cancelled' END WHERE id=? AND status IN ('running','queued')""", (ident,))
            else:
                result = db.execute("""UPDATE jobs SET status='queued',started=NULL,finished=NULL,exit_code=NULL
                    WHERE id=? AND status IN ('cancelled','failed','done')""", (ident,))
            if not result.rowcount:
                raise ValueError('Задание не найдено или его статус уже изменился.')

    def status(self, ident):
        with self.connect() as db:
            row = db.execute('SELECT status FROM jobs WHERE id=?', (ident,)).fetchone()
            return row['status'] if row else None

    def log(self, ident):
        if not re.fullmatch('[0-9a-f]{32}', ident) or self.status(ident) is None:
            return None
        path = self.jobs / f'{ident}.log'
        if not path.exists():
            return ''
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 48000))
            return ANSI.sub('', stream.read().decode('utf-8', errors='replace')).replace('\r', '\n')

    def command(self, job):
        return [sys.executable, str(Path(__file__).with_name('cyberdrop_runtime.py')), 'download',
                '--input-file', str(self.jobs / f"{job['id']}.txt"),
                '--download-folder', os.environ.get('DOWNLOAD_DIR', '/videos'),
                '--ui', 'simple', '--refresh-rate', '1', '--logs.console-level', 'INFO',
                '--config-file', str(self.root / 'config.yaml'),
                '--database-file', str(self.root / 'cyberdrop.db'),
                '--cache-file', str(self.root / 'cache.json'),
                '--logs.folder', str(self.jobs / job['id'])]

    def terminate(self):
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass

    def run_job(self, job):
        ident = job['id']
        (self.jobs / f'{ident}.txt').write_text('\n'.join(json.loads(job['urls'])) + '\n', encoding='utf-8')
        for filename in ('config.yaml', 'cyberdrop.db', 'cache.json'):
            (self.root / filename).touch(exist_ok=True)
        # The same lock as the renamer: never rename an active download.
        with (self.state / '.batinapapka.lock').open('a') as lock:
            while not self.stop.is_set():
                if self.status(ident) == 'cancelling':
                    break
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.stop.wait(.5)
            if self.stop.is_set() or self.status(ident) == 'cancelling':
                return
            handler = RotatingFileHandler(self.jobs / f'{ident}.log', maxBytes=5_000_000, backupCount=1)
            output = logging.getLogger(f'job.{ident}')
            output.setLevel(logging.INFO)
            output.propagate = False
            output.addHandler(handler)
            output.info('Starting / resuming Cyberdrop download')
            try:
                self.proc = subprocess.Popen(self.command(job), cwd=self.root, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                    env=os.environ | {'PYTHONUNBUFFERED': '1', 'FORCE_COLOR': '1', 'COLUMNS': '100', 'CYBERDROP_QUEUE_LOCKED': '1'})
                # Drain output independently so cancellation works even if no output arrives.
                def drain():
                    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                    for chunk in iter(lambda: self.proc.stdout.read1(8192), b''):
                        output.info('%s', decoder.decode(chunk))
                    final = decoder.decode(b'', final=True)
                    if final:
                        output.info('%s', final)
                reader = threading.Thread(target=drain, daemon=True)
                reader.start()
                while self.proc.poll() is None:
                    if self.stop.wait(.25) or self.status(ident) == 'cancelling':
                        self.terminate()
                        break
                reader.join(timeout=5)
                code = self.proc.wait()
                with self.connect() as db:
                    status = self.status(ident)
                    final = 'cancelled' if status == 'cancelling' else ('queued' if self.stop.is_set() else ('done' if code == 0 else 'failed'))
                    db.execute('UPDATE jobs SET status=?,finished=?,exit_code=? WHERE id=?',
                               (final, time.time(), code, ident))
                output.info('Process exited: %s; status: %s', code, final)
            finally:
                self.terminate()
                if self.proc and self.proc.stdout:
                    self.proc.stdout.close()
                self.proc = None
                output.removeHandler(handler)
                handler.close()

    def worker(self):
        # Prevent a second service from replaying active jobs on the same volume.
        with (self.root / '.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.recover()
            while not self.stop.is_set():
                with self.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    job = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                    if job:
                        db.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), job['id']))
                if not job:
                    self.stop.wait(1)
                    continue
                try:
                    self.run_job(job)
                except Exception:
                    LOG.exception('Job %s failed', job['id'])
                    with self.connect() as db:
                        db.execute("UPDATE jobs SET status='failed',finished=? WHERE id=? AND status='running'", (time.time(), job['id']))
                finally:
                    with self.connect() as db:
                        db.execute("UPDATE jobs SET status='cancelled',finished=? WHERE id=? AND status='cancelling'", (time.time(), job['id']))
                        db.execute("UPDATE jobs SET status='queued' WHERE id=? AND status='running'", (job['id'],))


def create_app(queue, username, password):
    if not username or not password:
        raise ValueError('Set ADMIN_USER and ADMIN_PASSWORD (or SHELL_USER and SHELL_PASSWORD)')
    app = Flask(__name__, static_folder='web', static_url_path='/static')
    app.config['MAX_CONTENT_LENGTH'] = 1_048_576

    @app.before_request
    def protect():
        auth = request.authorization
        if not auth or auth.type.lower() != 'basic' or not (
            hmac.compare_digest((auth.username or '').encode(), username.encode()) and
            hmac.compare_digest((auth.password or '').encode(), password.encode())):
            return ('Authentication required', 401, {'WWW-Authenticate': 'Basic realm="BatinaPapka", charset="UTF-8"'})
        # Browser cross-origin forms cannot provide this custom header; CORS is disabled.
        if request.method == 'POST' and request.headers.get('X-Admin-Request') != '1':
            abort(403)
        if request.headers.get('Sec-Fetch-Site') == 'cross-site':
            abort(403)

    @app.after_request
    def headers(response):
        response.headers.update({'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Content-Security-Policy': "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            'Referrer-Policy': 'no-referrer'})
        return response

    @app.get('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    @app.get('/api/jobs')
    def jobs():
        return jsonify(queue.list())

    @app.post('/api/jobs')
    def add():
        text = request.form.get('urls', '')
        upload = request.files.get('file')
        if upload and upload.filename:
            if not upload.filename.lower().endswith('.txt'):
                return jsonify(error='Загрузите TXT-файл со ссылками, по одной в строке.'), 400
            try:
                text += '\n' + upload.read().decode('utf-8-sig')
            except UnicodeDecodeError:
                return jsonify(error='Файл должен быть в кодировке UTF-8.'), 400
        try:
            return jsonify(id=queue.add(text)), 201
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.post('/api/jobs/<ident>/<action>')
    def action(ident, action):
        if action not in ('cancel', 'retry'):
            abort(404)
        try:
            queue.action(ident, action)
            return jsonify(ok=True)
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    @app.get('/api/jobs/<ident>/log')
    def log(ident):
        value = queue.log(ident)
        if value is None:
            abort(404)
        return jsonify(log=value)

    @app.get('/api/runtime')
    def runtime():
        path = queue.root / 'runtime' / 'status.json'
        try:
            return jsonify(json.loads(path.read_text()))
        except (OSError, ValueError):
            return jsonify(message='Используется версия Cyberdrop из образа')

    return app


def main():
    from waitress import create_server
    logging.basicConfig(level=logging.INFO)
    queue = Queue(Path(os.environ.get('STATE_DIR', '/state')))
    app = create_app(queue, os.environ.get('ADMIN_USER') or os.environ.get('SHELL_USER'),
                     os.environ.get('ADMIN_PASSWORD') or os.environ.get('SHELL_PASSWORD'))
    server = create_server(app, host='0.0.0.0', port=int(os.environ.get('ADMIN_PORT', '8080')), threads=4)
    failed = threading.Event()
    def run_worker():
        try:
            queue.worker()
        except Exception:
            LOG.exception('Queue worker stopped unexpectedly')
            failed.set()
            queue.stop.set()
            server.close()
    worker = threading.Thread(target=run_worker, daemon=True)
    worker.start()
    def shutdown(*_):
        queue.stop.set()
        server.close()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.run()
    finally:
        queue.stop.set()
        worker.join(timeout=25)
    if failed.is_set():
        raise RuntimeError("Queue worker failed")


if __name__ == '__main__':
    main()
