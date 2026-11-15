import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from cyberdrop_admin import Queue, create_app
from telegram_bot import TelegramBot, load_settings, save_settings, TelegramError


class TelegramTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.queue=Queue(Path(self.temp.name))
        self.bot=TelegramBot(self.queue)
        self.bot.config={'token':'12345:'+'test_token_'*4,'allowed_user_ids':['123456789'],'enabled':True}
        self.bot.send=Mock()

    def tearDown(self):
        self.bot.session.close();self.temp.cleanup()

    def message(self,text,ident=1,user=123456789,chat_type='private'):
        return {'update_id':ident,'message':{'chat':{'id':user,'type':chat_type},'from':{'id':user},'text':text}}

    def test_only_allowed_private_sender_can_enqueue(self):
        self.bot.handle(self.message('https://example.test/a',user=999))
        self.bot.handle(self.message('/start',user=999))
        self.bot.handle(self.message('https://example.test/a',chat_type='group'))
        self.assertEqual(self.queue.list(),[]);self.bot.send.assert_not_called()
        self.bot.handle(self.message('https://example.test/a\nhttps://example.test/b'))
        jobs=self.queue.list()
        self.assertEqual(len(jobs),1);self.assertEqual(len(jobs[0]['urls']),2)
        self.assertEqual(jobs[0]['telegram_chat_id'],'123456789')
        self.bot.send.assert_called_once()

    def test_replayed_update_does_not_duplicate_download_after_restart(self):
        update=self.message('https://example.test/a')
        self.bot.handle(update)
        ident=self.queue.list()[0]['id']
        self.bot.queue=Queue(Path(self.temp.name))
        self.bot.handle(update)
        self.assertEqual(len(self.queue.list()),1)
        self.assertEqual(self.queue.list()[0]['id'],ident)
        self.bot.set_offset(25)
        self.assertEqual(self.bot.get_offset(),25)

    def test_single_completion_for_whole_job_not_each_file(self):
        self.bot.handle(self.message('https://example.test/album'))
        ident=self.queue.list()[0]['id']
        self.bot.send.reset_mock()
        self.bot.notify_finished();self.bot.send.assert_not_called()
        with self.queue.connect() as db:
            db.execute("UPDATE jobs SET status='done',finished=123 WHERE id=?",(ident,))
        (self.queue.jobs/(ident+'.log')).write_text('Downloaded: 50 files\nErrors: 0 files')
        self.bot.notify_finished();self.bot.notify_finished()
        self.bot.send.assert_called_once()
        text=self.bot.send.call_args.args[1]
        self.assertIn('Обработка ссылки закончена',text)
        self.assertIn('50 files',text)

    def test_failed_notification_is_retried_and_does_not_leak_token(self):
        self.bot.handle(self.message('https://example.test/a'))
        with self.queue.connect() as db:db.execute("UPDATE jobs SET status='failed',finished=123")
        self.bot.send.side_effect=TelegramError('offline')
        with self.assertRaises(TelegramError):self.bot.notify_finished()
        with self.queue.connect() as db:self.assertEqual(db.execute('SELECT telegram_notified FROM jobs').fetchone()[0],0)
        self.bot.send.side_effect=None;self.bot.notify_finished()
        with self.queue.connect() as db:self.assertEqual(db.execute('SELECT telegram_notified FROM jobs').fetchone()[0],1)

    def test_active_progress_and_pause_resume_from_bot(self):
        self.bot.handle(self.message('https://example.test/a'))
        ident=self.queue.list()[0]['id']
        with self.queue.connect() as db:db.execute("UPDATE jobs SET status='running'")
        (self.queue.jobs/(ident+'.log')).write_text('Downloading video.mp4 42%')
        self.bot.handle(self.message('/active',ident=2))
        self.assertIn('video.mp4 42%',self.bot.send.call_args.args[1])
        self.bot.handle(self.message('/pause '+ident[:8],ident=3))
        self.assertEqual(self.queue.status(ident),'pausing')
        self.queue.recover();self.assertEqual(self.queue.status(ident),'paused')
        self.bot.handle(self.message('/resume '+ident[:8],ident=4))
        self.assertEqual(self.queue.status(ident),'queued')

    def test_settings_are_authenticated_and_token_is_never_returned(self):
        client=create_app(self.queue,'user','pass').test_client()
        headers={'Authorization':'Basic '+base64.b64encode(b'user:pass').decode(),'X-Admin-Request':'1'}
        self.assertEqual(client.get('/api/telegram').status_code,401)
        response=client.post('/api/telegram',headers=headers,json={'enabled':True,'token':self.bot.config['token'],'allowed_user_ids':['123456789']})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['allowed_user_ids'],['123456789'])
        self.assertNotIn(self.bot.config['token'],client.get('/api/telegram',headers=headers).text)
        self.assertEqual((self.queue.state/'telegram-settings.json').stat().st_mode & 0o777,0o600)
        self.assertEqual(client.post('/api/telegram',headers=headers,json={'enabled':True,'token':'','allowed_user_ids':[]}).status_code,400)
        save_settings(self.queue.state,{'enabled':False,'token':'','allowed_user_ids':[]})
        self.assertFalse(load_settings(self.queue.state)['enabled'])

    def test_menu_buttons_and_inline_callbacks_need_no_typed_commands(self):
        self.bot.handle(self.message('https://example.test/a'))
        ident=self.queue.list()[0]['id']
        self.bot.handle(self.message('📋 Очередь',ident=2))
        keyboard=self.bot.send.call_args.kwargs['keyboard']
        self.assertTrue(any(button['callback_data']=='pause:'+ident for row in keyboard['inline_keyboard'] for button in row))
        self.bot.call=Mock(return_value=True)
        callback={'update_id':3,'callback_query':{'id':'button-click','from':{'id':123456789},
                  'message':{'chat':{'id':123456789,'type':'private'}},'data':'pause:'+ident}}
        self.bot.handle(callback)
        self.assertEqual(self.queue.status(ident),'paused')
        self.bot.call.assert_called_once_with('answerCallbackQuery',callback_query_id='button-click')
        callback['callback_query']['data']='resume:'+ident
        self.bot.handle(callback)
        self.assertEqual(self.queue.status(ident),'queued')
        self.bot.call.reset_mock()
        callback['callback_query']['from']['id']=999
        self.bot.handle(callback)
        self.bot.call.assert_not_called()

    def test_pending_update_does_not_recreate_deleted_job(self):
        update=self.message('https://example.test/a')
        self.bot.handle(update)
        self.queue.delete(self.queue.list()[0]['id'])
        self.bot.handle(update)
        self.assertEqual(self.queue.list(),[])
        self.assertIn('уже было удалено',self.bot.send.call_args.args[1])
        self.bot.set_offset(2)
        with self.queue.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM telegram_state WHERE key LIKE 'receipt:%'").fetchone()[0],0)
