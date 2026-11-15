"""Persistent, single-worker Cyberdrop queue and authenticated browser UI."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
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
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import tempfile
from urllib.parse import urlsplit
import uuid

from log_retention import clean_job_logs
from metadata_lookup import load_settings
import telegram_bot

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
        self.last_log_cleanup = 0
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, urls TEXT NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, started REAL, finished REAL, exit_code INTEGER,
                kind TEXT NOT NULL DEFAULT 'download')''')
            columns = [r['name'] for r in db.execute('PRAGMA table_info(jobs)')]
            if 'kind' not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'download'")
            for name, definition in [('telegram_chat_id','TEXT'), ('telegram_update_key','TEXT'),
                                     ('telegram_notified','INTEGER NOT NULL DEFAULT 0')]:
                if name not in columns:
                    db.execute(f'ALTER TABLE jobs ADD COLUMN {name} {definition}')
            db.execute('CREATE UNIQUE INDEX IF NOT EXISTS telegram_update_key ON jobs(telegram_update_key)')
            db.execute('CREATE TABLE IF NOT EXISTS telegram_state (key TEXT PRIMARY KEY, value TEXT)')

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
            db.execute("UPDATE jobs SET status='paused', finished=? WHERE status='pausing'", (time.time(),))

    def add(self, text, telegram_chat_id=None, telegram_update_key=None):
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
            db.execute('BEGIN IMMEDIATE')
            if telegram_update_key:
                receipt = db.execute('SELECT value FROM telegram_state WHERE key=?', ('receipt:'+telegram_update_key,)).fetchone()
                if receipt:
                    return receipt['value']
                existing = db.execute('SELECT id FROM jobs WHERE telegram_update_key=?', (telegram_update_key,)).fetchone()
                if existing:
                    return existing['id']
            db.execute('INSERT INTO jobs(id,urls,status,created,telegram_chat_id,telegram_update_key) VALUES(?,?,?,?,?,?)',
                       (ident, json.dumps(urls), 'queued', time.time(), telegram_chat_id, telegram_update_key))
            if telegram_update_key:
                db.execute('INSERT INTO telegram_state(key,value) VALUES(?,?)',('receipt:'+telegram_update_key,ident))
        return ident

    def add_renamer(self):
        ident = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id,urls,status,created,kind) VALUES(?,?,'queued',?,'rename')",
                       (ident, '[]', time.time()))
        return ident

    def list(self):
        with self.connect() as db:
            return [dict(r) | {'urls': json.loads(r['urls'])} for r in db.execute(
                """SELECT * FROM jobs ORDER BY CASE WHEN status IN ('running','cancelling','pausing') THEN 0
                WHEN status='queued' THEN 1 WHEN status='paused' THEN 2 ELSE 3 END,
                CASE WHEN status='queued' THEN created END ASC, created DESC LIMIT 200""")]

    def delete(self, ident):
        if not re.fullmatch('[0-9a-f]{32}', ident):
            raise ValueError('Задание не найдено.')
        # Serialize against claiming queued jobs and retry/cancel requests.
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status FROM jobs WHERE id=?', (ident,)).fetchone()
            if not row:
                raise ValueError('Задание не найдено.')
            if row['status'] not in ('queued','done','failed','cancelled','paused'):
                raise ValueError('Сначала отмените задание и дождитесь остановки процесса.')
            for suffix in ('.txt', '.log', '.log.1'):
                (self.jobs / (ident + suffix)).unlink(missing_ok=True)
            diagnostics = self.jobs / ident
            if diagnostics.is_symlink():
                diagnostics.unlink()
            elif diagnostics.exists():
                shutil.rmtree(diagnostics)
            db.execute('DELETE FROM jobs WHERE id=?', (ident,))

    def action(self, ident, action):
        with self.connect() as db:
            if action == 'cancel':
                result = db.execute("""UPDATE jobs SET status=CASE WHEN status='running'
                    THEN 'cancelling' WHEN status='pausing' THEN 'cancelling' ELSE 'cancelled' END WHERE id=? AND status IN ('running','queued','paused','pausing')""", (ident,))
            elif action == 'pause':
                result = db.execute("""UPDATE jobs SET status=CASE WHEN status='running' THEN 'pausing' ELSE 'paused' END
                    WHERE id=? AND status IN ('queued','running')""", (ident,))
            elif action == 'resume':
                result = db.execute("""UPDATE jobs SET status='queued',started=NULL,finished=NULL,exit_code=NULL,telegram_notified=0
                    WHERE id=? AND status IN ('cancelled','failed','paused')""", (ident,))
            else:
                result = db.execute("""UPDATE jobs SET status='queued',started=NULL,finished=NULL,exit_code=NULL,telegram_notified=0
                    WHERE id=? AND status IN ('cancelled','failed','done','paused')""", (ident,))
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
            return '' if self.status(ident) in ('queued','running','cancelling') else 'Журнал отсутствует или удалён по лимиту хранения.'
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 48000))
            return ANSI.sub('', stream.read().decode('utf-8', errors='replace')).replace('\r', '\n')

    def command(self, job):
        if job['kind'] == 'rename':
            return [sys.executable, str(Path(__file__).with_name('batinapapka.py')), '--recursive',
                    os.environ.get('DOWNLOAD_DIR', '/videos')]
        return [sys.executable, str(Path(__file__).with_name('cyberdrop_runtime.py')), 'download',
                '--input-file', str(self.jobs / f"{job['id']}.txt"),
                '--download-folder', os.environ.get('DOWNLOAD_DIR', '/videos'),
                '--ui', 'simple', '--refresh-rate', '1', '--logs.console-level', 'INFO',
                '--config-file', str(self.root / 'config.yaml'),
                '--database-file', str(self.root / 'cyberdrop.db'),
                '--cache-file', str(self.root / 'cache.json'),
                '--logs.folder', str(self.jobs / job['id']),
                '--log-file', str(self.jobs / job['id'] / 'managed.log'),
                '--logs.no-rotate', '--no-dump-json', '--no-dump-responses', '--logs.no-http-traffic']

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
        native_logs = self.jobs / ident
        native_logs.mkdir(exist_ok=True)
        # Console output is already captured with rotation; avoid a second,
        # unbounded native DEBUG log. CSV diagnostics remain available.
        native_log = native_logs / 'managed.log'
        if not native_log.exists() and not native_log.is_symlink():
            native_log.symlink_to('/dev/null')
        (self.jobs / f'{ident}.txt').write_text('\n'.join(json.loads(job['urls'])) + '\n', encoding='utf-8')
        for filename in ('config.yaml', 'cyberdrop.db', 'cache.json'):
            (self.root / filename).touch(exist_ok=True)
        # The same lock as the renamer: never rename an active download.
        context = (self.state / '.batinapapka.lock').open('a') if job['kind'] == 'download' else nullcontext()
        with context as lock:
            while lock is not None and not self.stop.is_set():
                if self.status(ident) in ('cancelling','pausing'):
                    break
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.stop.wait(.5)
            if self.stop.is_set() or self.status(ident) in ('cancelling','pausing'):
                return
            handler = RotatingFileHandler(self.jobs / f'{ident}.log', maxBytes=5_000_000, backupCount=1)
            output = logging.getLogger(f'job.{ident}')
            output.setLevel(logging.INFO)
            output.propagate = False
            output.addHandler(handler)
            output.info('Starting / resuming %s', job['kind'])
            try:
                self.proc = subprocess.Popen(self.command(job), cwd=self.state if job['kind'] == 'rename' else self.root, stdin=subprocess.DEVNULL,
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
                    self.cleanup_logs()
                    if self.stop.wait(.25) or self.status(ident) in ('cancelling','pausing'):
                        self.terminate()
                        break
                reader.join(timeout=5)
                code = self.proc.wait()
                with self.connect() as db:
                    status = self.status(ident)
                    final = 'paused' if status == 'pausing' else ('cancelled' if status == 'cancelling' else ('queued' if self.stop.is_set() else ('done' if code == 0 else 'failed')))
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

    def cleanup_logs(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_log_cleanup < 5:
            return
        self.last_log_cleanup = now
        try:
            with self.connect() as db:
                active = [row['id'] for row in db.execute("SELECT id FROM jobs WHERE status IN ('running','cancelling','pausing')")]
            clean_job_logs(self.jobs, active)
        except OSError:
            LOG.exception('Log retention cleanup failed')

    def worker(self):
        # Prevent a second service from replaying active jobs on the same volume.
        with (self.root / '.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.recover()
            self.cleanup_logs(force=True)
            while not self.stop.is_set():
                self.cleanup_logs()
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
                        db.execute("UPDATE jobs SET status='paused',finished=? WHERE id=? AND status='pausing'", (time.time(), job['id']))
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

    @app.post('/api/rename')
    def rename():
        if not load_settings(queue.state)['tpdb_api_key'] and not os.environ.get('BRAVE_API_KEY'):
            return jsonify(error='Сначала задайте токен ThePornDB или BRAVE_API_KEY'), 400
        return jsonify(id=queue.add_renamer()), 201

    @app.post('/api/jobs/<ident>/<action>')
    def action(ident, action):
        if action not in ('cancel', 'retry', 'delete', 'pause', 'resume'):
            abort(404)
        try:
            if action == 'delete':
                queue.delete(ident)
            else:
                queue.action(ident, action)
            return jsonify(ok=True)
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

        except OSError:
            LOG.exception('Job cleanup failed')
            return jsonify(error='Не удалось удалить файлы журнала. Попробуйте ещё раз.'), 500

    @app.get('/api/jobs/<ident>/log')
    def log(ident):
        value = queue.log(ident)
        if value is None:
            abort(404)
        return jsonify(log=value)

    @app.get('/api/settings')
    def settings():
        config = load_settings(queue.state)
        return jsonify(enabled=config['enabled'], configured=bool(config['tpdb_api_key']))

    @app.post('/api/settings')
    def save_settings():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get('enabled'), bool):
            return jsonify(error='Некорректные настройки'), 400
        key = data.get('tpdb_api_key', '')
        if not isinstance(key, str) or len(key) > 4096 or any(ord(c) < 32 for c in key):
            return jsonify(error='Некорректный токен'), 400
        current = load_settings(queue.state)
        config = {'enabled': data['enabled'], 'tpdb_api_key':key.strip() or current['tpdb_api_key']}
        path = queue.state / 'metadata-settings.json'
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=queue.state,
                                         prefix='.metadata-settings-', delete=False) as stream:
            json.dump(config, stream)
            stream.flush()
            os.fsync(stream.fileno())
            temp = stream.name
        os.chmod(temp, 0o600)
        os.replace(temp, path)
        return jsonify(enabled=config['enabled'], configured=bool(config['tpdb_api_key']))

    @app.get('/api/telegram')
    def telegram_settings():
        return jsonify(telegram_bot.public_settings(telegram_bot.load_settings(queue.state)))

    @app.post('/api/telegram')
    def save_telegram_settings():
        try:
            return jsonify(telegram_bot.save_settings(queue.state,request.get_json(silent=True)))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

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
    bot = telegram_bot.TelegramBot(queue)
    bot_thread = threading.Thread(target=bot.run, daemon=True)
    bot_thread.start()
    def shutdown(*_):
        queue.stop.set()
        server.close()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.run()
    except OSError:
        if not queue.stop.is_set():
            raise
    finally:
        queue.stop.set()
        worker.join(timeout=25)
        bot_thread.join(timeout=20)
    if failed.is_set():
        raise RuntimeError("Queue worker failed")


if __name__ == '__main__':
    main()
