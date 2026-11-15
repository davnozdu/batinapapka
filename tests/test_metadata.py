import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from metadata_lookup import MetadataClient, oshash
import batinapapka as bp
from cyberdrop_admin import Queue, create_app
import base64


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.client=MetadataClient(settings={'enabled':True,'tpdb_api_key':'test-token'})
        self.scene={'id':'one','title':'A Sample Movie Title','date':'2020-03-04','duration':100}

    def test_hash_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'video.mp4'
            path.write_bytes(b'\x01'+b'\0'*131063+b'\x02'+b'\0'*7)
            self.assertEqual(oshash(path),'0000000000020003')
            path.write_bytes(b'small')
            self.assertIsNone(oshash(path))

    def test_hash_wins_before_name(self):
        self.client.query=Mock(return_value=[self.scene|{'hashes':[{'type':'OSHASH','hash':'1234'}]}])
        with patch('metadata_lookup.oshash',return_value='1234'):
            result=self.client.lookup('video.mp4','random filename')
        self.assertEqual(result,('2020-03-04','A Sample Movie Title','tpdb-oshash'))
        self.assertEqual(self.client.query.call_count,1)

    def test_wrong_hash_and_ambiguous_name_rejected(self):
        self.client.query=Mock(return_value=[self.scene|{'hashes':[{'type':'OSHASH','hash':'wrong'}]}, self.scene|{'id':'two'}])
        with patch('metadata_lookup.oshash',return_value='1234'):
            self.assertIsNone(self.client.lookup('file','A Sample Movie Title'))

    def test_exact_metadata_and_unrelated_title(self):
        self.client.query=Mock(return_value=[self.scene])
        with patch('metadata_lookup.oshash',return_value=None):
            self.assertEqual(self.client.lookup('file','A Sample Movie Title')[2],'tpdb-metadata')
            self.assertIsNone(self.client.lookup('file','Entirely Different Film Name'))

    def test_invalid_auth_falls_back_without_cache(self):
        self.client.session.get=Mock(return_value=Mock(status_code=401))
        self.assertEqual(self.client.query('scenes',{'parse':'test'}),[])
        self.assertFalse(self.client.available)
        self.client.query('scenes',{'parse':'test'})
        self.assertEqual(self.client.session.get.call_count,1)

    def test_metadata_precedes_brave_then_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            old=os.getcwd();os.chdir(temp)
            try:
                metadata=Mock()
                metadata.lookup.return_value=('2020-03-04','A Sample Movie Title','tpdb-oshash')
                brave=Mock()
                brave.session.headers.get.return_value='brave-test'
                brave.search.return_value=[]
                cache=bp.CompressedCache('cache.gz')
                renamer=bp.VideoFileRenamer(brave,cache,metadata_client=metadata)
                path=Path('source.mp4');path.write_bytes(b'video');os.utime(path,(1000,1000))
                with patch.object(bp,'ffprobe_title',return_value=None):
                    renamer.process('.')
                self.assertTrue(Path('2020-03-04 A Sample Movie Title.mp4').exists())
                brave.search.assert_not_called()
                metadata.lookup.return_value=None
                path=Path('another.mp4');path.write_bytes(b'video');os.utime(path,(1000,1000))
                with patch.object(bp,'ffprobe_title',return_value=None):
                    renamer.process('.')
                brave.search.assert_called()
            finally:
                os.chdir(old)

    def test_browser_settings_hide_token_and_enqueue_renamer(self):
        with tempfile.TemporaryDirectory() as temp:
            queue=Queue(Path(temp));client=create_app(queue,'user','pass').test_client()
            headers={'Authorization':'Basic '+base64.b64encode(b'user:pass').decode(),'X-Admin-Request':'1'}
            response=client.post('/api/settings',json={'enabled':True,'tpdb_api_key':'secret-test'},headers=headers)
            self.assertEqual(response.status_code,200)
            self.assertTrue(response.json['configured'])
            self.assertNotIn('secret-test',client.get('/api/settings',headers=headers).text)
            self.assertEqual((Path(temp)/'metadata-settings.json').stat().st_mode & 0o777,0o600)
            response=client.post('/api/rename',headers=headers)
            self.assertEqual(response.status_code,201)
            self.assertEqual(queue.list()[0]['kind'],'rename')
            self.assertIn('batinapapka.py',' '.join(queue.command(queue.list()[0])) )
