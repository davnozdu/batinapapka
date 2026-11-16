import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from scripts import check_cyberdrop_release as check


class ReleaseCheckTests(unittest.TestCase):
    def test_resolves_newest_numeric_stable_and_excludes_yanked_prereleases(self):
        data={'releases':{'10.9.0':[{}], '10.16.0':[{'yanked':False}],
                          '10.17.0rc1':[{}], '10.18.0':[{'yanked':True}], '10.19.0':[]}}
        self.assertEqual(check.latest_stable(data),'10.16.0')

    def test_no_rebuild_when_current_and_no_automatic_downgrade(self):
        self.assertTrue(check.needs_update('10.16.0','10.15.3'))
        self.assertTrue(check.needs_update('10.16.0',''))
        self.assertFalse(check.needs_update('10.16.0','10.16.0'))
        self.assertFalse(check.needs_update('10.16.0','10.17.0'))
        self.assertTrue(check.needs_update('10.16.0','10.16.0',True))

    def test_ghcr_index_reads_bundled_version_label(self):
        responses=[{'token':'read-only-test-token'},
                   {'manifests':[{'platform':{'os':'unknown','architecture':'unknown'},'digest':'attestation'},
                                 {'platform':{'os':'linux','architecture':'amd64'},'digest':'sha256:image'}]},
                   {'config':{'digest':'sha256:config'}},
                   {'config':{'Labels':{check.LABEL:'10.16.0'}}}]
        with patch.object(check,'get_json',side_effect=responses) as fetch:
            self.assertEqual(check.published_version('ghcr.io/davnozdu/batinapapka-cyberdrop'),'10.16.0')
            self.assertTrue(fetch.call_args.args[0].endswith('/blobs/sha256:config'))

    def test_unlabelled_old_image_and_missing_latest_need_rebuild(self):
        with patch.object(check,'get_json',side_effect=[{'token':'test'}, {'config':{'digest':'config'}}, {'config':{}}]):
            self.assertEqual(check.published_version('ghcr.io/davnozdu/batinapapka-cyberdrop'),'')
        error=urllib.error.HTTPError('https://ghcr.io/test',404,'not found',{},None)
        with patch.object(check,'get_json',side_effect=[{'token':'test'},error]):
            self.assertEqual(check.published_version('ghcr.io/davnozdu/batinapapka-cyberdrop'),'')

    def test_auth_and_network_failures_are_not_mistaken_for_new_release(self):
        error=urllib.error.HTTPError('https://ghcr.io/test',403,'forbidden',{},None)
        with patch.object(check,'get_json',side_effect=[{'token':'test'},error]):
            with self.assertRaises(urllib.error.HTTPError):
                check.published_version('ghcr.io/davnozdu/batinapapka-cyberdrop')

    def test_explicit_version_validation_and_resolve_only_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'outputs'
            with patch.dict(os.environ,{'GITHUB_OUTPUT':str(output),'GITHUB_STEP_SUMMARY':''}), \
                 patch('sys.argv',['check','--resolve-only','--version','10.16.0']), \
                 patch.object(check,'get_json') as fetch,contextlib.redirect_stdout(io.StringIO()):
                check.main()
                fetch.assert_not_called()
            self.assertIn('version=10.16.0\n',output.read_text())
            for value in ('10.16.0\ninjected=true','10.17.0rc1','$(command)','latest'):
                with self.assertRaises(ValueError):check.version_key(value)
