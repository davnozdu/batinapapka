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

    def test_delete_history_and_artifacts_preserves_downloads_and_database(self):
        videos=Path(self.temp.name)/'videos';videos.mkdir()
        video=videos/'keep.mp4';video.write_bytes(b'downloaded video')
        history=self.queue.root/'cyberdrop.db';history.write_bytes(b'download history')
        for status in ('queued','done','failed','cancelled'):
            ident=self.queue.add('https://example.test/1')
            with self.queue.connect() as db:
                db.execute('UPDATE jobs SET status=? WHERE id=?',(status,ident))
            for suffix in ('.txt','.log','.log.1'):
                (self.queue.jobs/(ident+suffix)).write_text('job data')
            diagnostics=self.queue.jobs/ident;diagnostics.mkdir()
            (diagnostics/'errors.csv').write_text('diagnostic')
            (diagnostics/'managed.log').symlink_to(video)
            response=self.client.post(f'/api/jobs/{ident}/delete',headers=self.headers)
            self.assertEqual(response.status_code,200)
            self.assertIsNone(self.queue.status(ident))
            self.assertFalse(any(self.queue.jobs.glob(ident+'*')))
            self.assertEqual(self.client.get(f'/api/jobs/{ident}/log',headers=self.headers).status_code,404)
        self.assertEqual(video.read_bytes(),b'downloaded video')
        self.assertEqual(history.read_bytes(),b'download history')
        self.assertEqual(Queue(Path(self.temp.name)).list(),[])

    def test_delete_rejects_active_jobs_invalid_ids_and_cross_site_requests(self):
        ident=self.queue.add('https://example.test/1')
        for status in ('running','cancelling'):
            with self.queue.connect() as db:
                db.execute('UPDATE jobs SET status=? WHERE id=?',(status,ident))
            response=self.client.post(f'/api/jobs/{ident}/delete',headers=self.headers)
            self.assertEqual(response.status_code,409)
            self.assertEqual(self.queue.status(ident),status)
        self.assertEqual(self.client.post(f'/api/jobs/{ident}/delete').status_code,401)
        self.assertEqual(self.client.post(f'/api/jobs/{ident}/delete',headers={'Authorization':self.headers['Authorization']}).status_code,403)
        self.assertEqual(self.client.post('/api/jobs/not-a-uuid/delete',headers=self.headers).status_code,409)
        self.assertEqual(self.client.post('/api/jobs/'+ 'a'*32 +'/delete',headers=self.headers).status_code,409)

    def test_delete_does_not_follow_job_directory_symlink(self):
        ident=self.queue.add('https://example.test/1')
        outside=Path(self.temp.name)/'keep';outside.mkdir()
        (outside/'file').write_text('preserved')
        (self.queue.jobs/ident).symlink_to(outside,target_is_directory=True)
        self.queue.delete(ident)
        self.assertEqual((outside/'file').read_text(),'preserved')

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
                self.assertEqual(self.client.post(f'/api/jobs/{ident}/delete',headers=self.headers).status_code,409)
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

    def test_pause_and_resume_preserve_partial_file_and_job_id(self):
        partial=Path(self.temp.name)/'download.part'
        code="from pathlib import Path; import time; p=Path("+repr(str(partial))+"); existed=p.exists(); p.write_text(p.read_text()+' finished' if existed else 'partial'); print('resumed' if existed else 'downloading',flush=True); time.sleep(0 if existed else 60)"
        ident=self.queue.add('https://example.test/a')
        with patch.object(self.queue,'command',return_value=[sys.executable,'-u','-c',code]):
            worker=threading.Thread(target=self.queue.worker);worker.start()
            try:
                deadline=time.monotonic()+5
                while not partial.exists() and time.monotonic()<deadline:time.sleep(.05)
                self.queue.action(ident,'pause');self.wait_status(ident,'paused')
                self.assertEqual(partial.read_text(),'partial')
                self.assertEqual(Queue(Path(self.temp.name)).status(ident),'paused')
                self.queue.action(ident,'resume');self.wait_status(ident,'done')
                self.assertEqual(partial.read_text(),'partial finished')
                self.assertEqual(len(self.queue.list()),1)
            finally:
                self.queue.stop.set();worker.join(15)

    def test_restart_resumes_unfinished_job_from_partial_file(self):
        partial=Path(self.temp.name)/'download.part'
        code="from pathlib import Path; import time; p=Path("+repr(str(partial))+"); existed=p.exists(); p.write_text(p.read_text()+' finished' if existed else 'partial'); time.sleep(0 if existed else 60)"
        ident=self.queue.add('https://example.test/a')
        with patch.object(self.queue,'command',return_value=[sys.executable,'-c',code]):
            worker=threading.Thread(target=self.queue.worker);worker.start()
            try:
                deadline=time.monotonic()+5
                while not partial.exists() and time.monotonic()<deadline:time.sleep(.05)
            finally:
                self.queue.stop.set();worker.join(15)
        self.assertEqual(self.queue.status(ident),'queued')
        self.queue=Queue(Path(self.temp.name))
        with patch.object(self.queue,'command',return_value=[sys.executable,'-c',code]):
            worker=threading.Thread(target=self.queue.worker);worker.start()
            try:
                self.wait_status(ident,'done')
                self.assertEqual(partial.read_text(),'partial finished')
            finally:
                self.queue.stop.set();worker.join(15)


if __name__=='__main__':
    unittest.main()
