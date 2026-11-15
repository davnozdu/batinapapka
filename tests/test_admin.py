import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from cyberdrop_admin import Queue, create_app


class AdminTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.queue = Queue(Path(self.temp.name))
        self.client = create_app(self.queue, 'admin', 'password').test_client()
        self.headers = {'Authorization': 'Basic '+base64.b64encode(b'admin:password').decode(), 'X-Admin-Request':'1'}

    def tearDown(self):
        self.queue.stop.set()
        self.temp.cleanup()

    def test_auth_csrf_and_input(self):
        self.assertEqual(self.client.get('/api/jobs').status_code, 401)
        self.assertEqual(self.client.post('/api/jobs', headers={'Authorization':self.headers['Authorization']}).status_code,403)
        for url in ('file:///etc/passwd', '--help', 'https://x.test/\tfoo', 'https://[bad'):
            response = self.client.post('/api/jobs',data={'urls':url},headers=self.headers)
            self.assertEqual(response.status_code,400)
        response = self.client.post('/api/jobs',data={'urls':'https://example.test/a\nhttps://example.test/a'},headers=self.headers)
        self.assertEqual(response.status_code,201)
        self.assertEqual(len(self.client.get('/api/jobs',headers=self.headers).json[0]['urls']),1)
        self.assertIn("frame-ancestors 'none'",self.client.get('/').headers['Content-Security-Policy'])

    def test_upload_and_persistent_queue(self):
        import io
        response = self.client.post('/api/jobs',data={'file':(io.BytesIO(b'\xef\xbb\xbfhttps://example.test/1\n'),'urls.txt')},headers=self.headers)
        self.assertEqual(response.status_code,201)
        restored = Queue(Path(self.temp.name))
        self.assertEqual(restored.list()[0]['id'],response.json['id'])
        self.assertEqual(self.client.get('/api/jobs/not-a-job/log',headers=self.headers).status_code,404)

    def test_recover_cancel_retry(self):
        ident = self.queue.add('https://example.test/1')
        with self.queue.connect() as db:
            db.execute("UPDATE jobs SET status='running'")
        self.queue.recover()
        self.assertEqual(self.queue.status(ident),'queued')
        self.queue.action(ident,'cancel')
        self.assertEqual(self.queue.status(ident),'cancelled')
        self.queue.action(ident,'retry')
        self.assertEqual(self.queue.status(ident),'queued')
        with self.queue.connect() as db:
            db.execute("UPDATE jobs SET status='cancelling'")
        self.queue.recover()
        self.assertEqual(self.queue.status(ident),'cancelled')

    def wait_status(self, ident, status):
        deadline = time.monotonic()+8
        while time.monotonic()<deadline:
            if self.queue.status(ident)==status:
                return
            time.sleep(.05)
        self.fail(f'Expected {status}, got {self.queue.status(ident)}')

    def test_worker_runs_without_browser_and_cancels(self):
        ident = self.queue.add('https://example.test/1')
        with patch.object(self.queue,'command',return_value=[sys.executable,'-u','-c','import time; print("downloading",end="\\r",flush=True); time.sleep(60)']):
            worker = threading.Thread(target=self.queue.worker)
            worker.start()
            try:
                self.wait_status(ident,'running')
                deadline=time.monotonic()+5
                while 'downloading' not in self.queue.log(ident) and time.monotonic()<deadline:
                    time.sleep(.05)
                self.assertIn('downloading',self.queue.log(ident))
                self.queue.action(ident,'cancel')
                self.wait_status(ident,'cancelled')
                self.assertIsNone(self.queue.proc)
            finally:
                self.queue.stop.set()
                worker.join(15)
            self.assertFalse(worker.is_alive())

    def test_worker_stop_preserves_job(self):
        ident=self.queue.add('https://example.test/1')
        with patch.object(self.queue,'command',return_value=[sys.executable,'-c','import time;time.sleep(60)']):
            worker=threading.Thread(target=self.queue.worker)
            worker.start()
            self.wait_status(ident,'running')
            self.queue.stop.set()
            worker.join(15)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.queue.status(ident),'queued')

    def test_worker_completion_and_exit_code(self):
        ident=self.queue.add('https://example.test/1')
        with patch.object(self.queue,'command',return_value=[sys.executable,'-c','print("complete")']):
            worker=threading.Thread(target=self.queue.worker)
            worker.start()
            try:
                self.wait_status(ident,'done')
                self.assertEqual(self.queue.list()[0]['exit_code'],0)
            finally:
                self.queue.stop.set();worker.join(15)


if __name__=='__main__':
    unittest.main()
