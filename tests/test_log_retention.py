import os
from pathlib import Path
import tempfile
import time
import unittest
from log_retention import clean_job_logs, trim_append_file


class RetentionTests(unittest.TestCase):
    def test_age_and_total_budget_preserve_active_and_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            old=root/('a'*32+'.log');old.write_bytes(b'old')
            os.utime(old,(time.time()-31*86400,)*2)
            inactive=root/('b'*32+'.log');inactive.write_bytes(b'x'*100)
            active=root/('c'*32+'.log');active.write_bytes(b'y'*100)
            inputs=root/('b'*32+'.txt');inputs.write_text('https://example.test')
            db=root/'queue.db';db.write_bytes(b'preserve')
            clean_job_logs(root,['c'*32],max_total_bytes=100)
            self.assertFalse(old.exists());self.assertFalse(inactive.exists())
            self.assertTrue(active.exists());self.assertTrue(inputs.exists());self.assertTrue(db.exists())

    def test_symlink_target_not_touched(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobs=root/'jobs';jobs.mkdir()
            victim=root/'keep.log';victim.write_bytes(b'keep'*100)
            (jobs/('a'*32+'.log')).symlink_to(victim)
            directory=jobs/('b'*32);directory.mkdir()
            (directory/'link').symlink_to(root,target_is_directory=True)
            clean_job_logs(jobs,max_total_bytes=0,max_age_days=0)
            self.assertEqual(victim.read_bytes(),b'keep'*100)

    def test_active_csv_trim_and_append(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'errors.csv';path.write_bytes(b'old\n'*100+b'new\n')
            trim_append_file(path,20)
            self.assertLessEqual(path.stat().st_size,20)
            with path.open('ab') as output:output.write(b'next\n')
            self.assertTrue(path.read_bytes().endswith(b'new\nnext\n'))

    def test_cron_output_rotates_and_preserves_exit_code(self):
        from unittest.mock import patch
        import sys
        import run_scheduled
        with tempfile.TemporaryDirectory() as temp:
            state=Path(temp)
            runner=state/'.run-renamer.sh'
            runner.write_text('#!'+sys.executable+"\nimport sys\nsys.stdout.write('x'*16000000)\nsys.exit(42)\n")
            runner.chmod(0o700)
            with patch.dict(os.environ,{'STATE_DIR':temp}):
                self.assertEqual(run_scheduled.main(),42)
            logs=list(state.glob('cron.log*'))
            self.assertEqual(len(logs),3)
            self.assertTrue(all(path.stat().st_size<=5*1024*1024 for path in logs))
