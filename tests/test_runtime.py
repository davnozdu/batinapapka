import json
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
            with patch.dict(os.environ,{'CYBERDROP_AUTO_UPDATE':'1'}), patch.object(runtime.urllib.request,'urlopen',return_value=response), patch.object(runtime.subprocess,'check_output',side_effect=['10.15.3','--input-file --download-folder --database-file --ui --logs.console-level']),patch.object(runtime,'run_checked',side_effect=create_venv):
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
