import json
from contextlib import closing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cyberdrop_runtime as runtime


class UpdateTests(unittest.TestCase):
    def test_success_switches_only_after_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            response=Mock()
            response.read.return_value=b'{"info":{"version":"10.16.0"}}'
            response.__enter__=Mock(return_value=response)
            response.__exit__=Mock(return_value=False)
            def create_venv(args,timeout):
                if '-m' in args:
                    binary=Path(args[-1])/'bin/cyberdrop-dl'
                    binary.parent.mkdir(parents=True)
                    binary.touch()
                self.assertFalse((root/'current').exists())
            with patch.dict(os.environ,{'CYBERDROP_AUTO_UPDATE':'1'}), patch.object(runtime.urllib.request,'urlopen',return_value=response), patch.object(runtime.subprocess,'check_output',side_effect=['10.15.3','--input-file --download-folder --database-file --ui --logs.console-level --log-file --logs.no-rotate --no-dump-json --no-dump-responses --logs.no-http-traffic']),patch.object(runtime,'run_checked',side_effect=create_venv):
                runtime.update(root)
            self.assertTrue((root/'current/bin/cyberdrop-dl').exists())
            self.assertEqual(json.loads((root/'status.json').read_text())['version'],'10.16.0')

    def test_failed_update_retains_current(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            old=root/'venv-old/bin/cyberdrop-dl'
            old.parent.mkdir(parents=True)
            old.touch()
            (root/'current').symlink_to('venv-old',target_is_directory=True)
            with patch.dict(os.environ,{'CYBERDROP_AUTO_UPDATE':'1'}),patch.object(runtime.subprocess,'check_output',return_value='10.15.3'),patch.object(runtime.urllib.request,'urlopen',side_effect=OSError('offline')):
                runtime.update(root)
            self.assertEqual((root/'current').readlink(),Path('venv-old'))
            self.assertIn('сохранена рабочая версия',json.loads((root/'status.json').read_text())['message'])

    def test_interval_avoids_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            import time
            (root/'status.json').write_text(json.dumps({'checked':time.time(),'version':'10.15.3'}))
            with patch.object(runtime.urllib.request,'urlopen') as fetch:
                runtime.update(root)
                fetch.assert_not_called()

    def test_legacy_database_is_copied_and_never_overwrites_current(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as temp:
            state=Path(temp)
            legacy=state/'cyberdrop/AppData/Cache/cyberdrop.db';legacy.parent.mkdir(parents=True)
            with closing(sqlite3.connect(legacy)) as db, db:
                db.execute('CREATE TABLE history (name TEXT)')
                db.execute("INSERT INTO history VALUES ('downloaded')")
            runtime.preserve_legacy_database(state)
            current=state/'cyberdrop/cyberdrop.db'
            with closing(sqlite3.connect(current)) as db, db:
                self.assertEqual(db.execute('SELECT name FROM history').fetchone()[0],'downloaded')
                db.execute("INSERT INTO history VALUES ('new')")
            runtime.preserve_legacy_database(state)
            with closing(sqlite3.connect(current)) as db, db:
                self.assertEqual(db.execute('SELECT count(*) FROM history').fetchone()[0],2)
            self.assertTrue(legacy.exists())

    def test_new_image_uses_its_bundle_instead_of_older_persistent_venv(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            bundled=root/'bundled';bundled.touch()
            old=root/'venv-10.15.3-1234abcd/bin/cyberdrop-dl'
            old.parent.mkdir(parents=True);old.touch()
            (root/'current').symlink_to(old.parent.parent.name,target_is_directory=True)
            with patch.dict(os.environ,{'CYBERDROP_BUNDLED':str(bundled),'CYBERDROP_BUNDLED_VERSION':'10.16.0'}):
                self.assertEqual(runtime.executable(root),str(bundled))
                newer=root/'venv-10.17.0-1234abcd/bin/cyberdrop-dl'
                newer.parent.mkdir(parents=True);newer.touch()
                (root/'current').unlink();(root/'current').symlink_to(newer.parent.parent.name,target_is_directory=True)
                self.assertEqual(runtime.executable(root),str(root/'current/bin/cyberdrop-dl'))

    def test_new_bundle_bypasses_old_interval_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            import time
            root=Path(temp)
            (root/'status.json').write_text(json.dumps({'checked':time.time(),'version':'10.15.3'}))
            with patch.dict(os.environ,{'CYBERDROP_BUNDLED_VERSION':'10.16.0','CYBERDROP_AUTO_UPDATE':'0'}), \
                 patch.object(runtime.subprocess,'check_output',return_value='10.16.0'), \
                 patch.object(runtime.urllib.request,'urlopen') as network:
                runtime.update(root)
                self.assertEqual(json.loads((root/'status.json').read_text())['version'],'10.16.0')
                network.assert_not_called()
