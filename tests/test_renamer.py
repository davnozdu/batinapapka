import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import batinapapka as bp


class RenamerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.previous=os.getcwd()
        os.chdir(self.temp.name)
        self.client=Mock()
        self.client.search.return_value=[]
        self.client.session.headers.get.return_value="test-key"
        self.renamer=bp.VideoFileRenamer(self.client,bp.CompressedCache('cache.gz'))

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()

    def video(self,name='Some.Sample.mp4'):
        path=Path(name)
        path.write_bytes(b'original video')
        os.utime(path,(1000000000,1000000000))
        return path

    @patch.object(bp,'ffprobe_title',return_value=None)
    def test_rename_mapping_and_repeat(self,_):
        self.video()
        self.renamer.process('.')
        files=list(Path('.').glob('*.mp4'))
        self.assertEqual(len(files),1)
        self.assertEqual(files[0].read_bytes(),b'original video')
        mapping=json.loads(Path('.filename_mapping.json').read_text())
        self.assertEqual(mapping[files[0].name],'Some.Sample.mp4')
        self.renamer.process('.')
        self.assertEqual(self.renamer.stats.renamed,1)

    @patch.object(bp,'ffprobe_title',return_value=None)
    def test_api_failure_leaves_original(self,_):
        path=self.video()
        self.client.search.side_effect=RuntimeError('API unavailable')
        self.renamer.process('.')
        self.assertTrue(path.exists())
        self.assertEqual(self.renamer.stats.errors,1)

    @patch.object(bp,'ffprobe_title',return_value=None)
    def test_corrupt_mapping_preserves_original(self,_):
        path=self.video()
        Path('.filename_mapping.json').write_text('invalid')
        self.renamer.process('.')
        self.assertTrue(path.exists())
        self.assertEqual(len(list(Path('.').glob('*.mp4'))),1)
        self.assertEqual(Path('.filename_mapping.json').read_text(),'invalid')

    @patch.object(bp,'ffprobe_title',return_value=None)
    def test_changed_file_not_renamed(self,_):
        path=self.video()
        def search(*args):
            path.write_bytes(b'still downloading')
            return []
        self.client.search.side_effect=search
        self.renamer.process('.')
        self.assertTrue(path.exists())
        self.assertEqual(self.renamer.stats.renamed,0)

    def test_utf8_filename_and_suffix_limit(self):
        name=self.renamer._safe('Ж'*250+'.mp4')
        self.assertLessEqual(len(name.encode()),255)
        self.video(name)
        unique=self.renamer._unique('.',name)
        self.assertNotEqual(unique,name)
        self.assertLessEqual(len(unique.encode()),255)
        self.assertNotIn('\n',self.renamer._safe('test\nname.mp4'))

    def test_corrupt_cache_types_and_timestamps(self):
        for payload in ({'cache':[],'timestamps':[]},{'cache':{'a':1},'timestamps':{'a':'bad'}}, {'cache':{},'timestamps':{'a':12}}):
            with gzip.open('cache.gz','wt') as out:
                json.dump(payload,out)
            cache=bp.CompressedCache('cache.gz')
            self.assertIsNone(cache.get('a'))

    def test_instance_lock(self):
        lock=bp.acquire_single_instance_lock('test.lock')
        try:
            self.assertIsNone(bp.acquire_single_instance_lock('test.lock'))
        finally:
            lock.close()
        self.assertIs(bp.acquire_single_instance_lock('/not-existing-dir/lock'),False)

    def test_same_name_in_other_directory_not_skipped(self):
        os.mkdir('other')
        self.renamer._mark_renamed(os.path.abspath('name.mp4'))
        self.assertTrue(self.renamer._should_process('name.mp4',False,'other'))

    def test_dates_and_name_tokens(self):
        processor=bp.TitleProcessor()
        date,title,_=processor.choose_best('Big Buck Bunny 2008',[
            {'title':'Big Buck Bunny 2008', 'page_age':'2026-07-29', 'rank':0}],year_hint='2008')
        self.assertIsNone(date)
        self.assertIsNotNone(title)
        _,title,_=processor.choose_best('Ann Example',[
            {'title':'Joanna Example Movie', 'page_age':'2020-01-01'}],required_tokens={'Ann','Example'})
        self.assertIsNone(title)
        self.assertIsNone(bp._parse_page_age('2026-99-99'))
        self.assertIsNone(bp._parse_page_age(123))

    def test_brave_auth_does_not_trigger_fallback(self):
        client=bp.BraveSearchClient('bad',bp.CompressedCache('cache.gz'))
        client.session.get=Mock(return_value=Mock(status_code=401))
        with self.assertRaises(RuntimeError):
            client.search('title')


if __name__=='__main__':
    unittest.main()
