"""Private Telegram front-end for the same persistent browser download queue."""
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time

import requests

LOG = logging.getLogger(__name__)
MENU = {'keyboard':[[{'text':'📥 Скачать'},{'text':'▶️ Активное задание'}],
                     [{'text':'📋 Очередь'},{'text':'⏸ Пауза'}],
                     [{'text':'▶️ Продолжить'},{'text':'⏹ Отменить'}],
                     [{'text':'❓ Помощь'}]],'resize_keyboard':True,'is_persistent':True}

STATES = {'queued':'в очереди','running':'скачивается','pausing':'приостанавливается',
          'paused':'на паузе','cancelling':'отменяется','cancelled':'отменено','done':'завершено','failed':'ошибка'}


def load_settings(state):
    try:
        data = json.loads((Path(state)/'telegram-settings.json').read_text())
        if not isinstance(data,dict):
            data = {}
    except (OSError,ValueError):
        data = {}
    token = data.get('token') or os.environ.get('TELEGRAM_BOT_TOKEN','')
    if not isinstance(token,str):
        token=''
    raw_ids = data.get('allowed_user_ids', os.environ.get('TELEGRAM_ALLOWED_USER_IDS','').split(','))
    if not isinstance(raw_ids,list):
        raw_ids=[]
    ids = [str(value).strip() for value in raw_ids if re.fullmatch(r'[1-9]\d*',str(value).strip())]
    return {'token':token,'allowed_user_ids':ids,'enabled':data.get('enabled',bool(token and ids)) is True}


def public_settings(config):
    return {'enabled':config['enabled'],'configured':bool(config['token']),
            'allowed_user_ids':config['allowed_user_ids']}


def save_settings(state,data):
    if not isinstance(data,dict) or not isinstance(data.get('enabled'),bool):
        raise ValueError('Некорректные настройки Telegram.')
    token = data.get('token','')
    if not isinstance(token,str) or (token and not re.fullmatch(r'\d+:[A-Za-z0-9_-]{20,}',token)):
        raise ValueError('Некорректный токен Telegram-бота.')
    if not isinstance(token,str):
        token=''
    raw_ids = data.get('allowed_user_ids',[])
    if not isinstance(raw_ids,list) or len(raw_ids)>20 or any(not re.fullmatch(r'[1-9]\d*',str(value)) for value in raw_ids):
        raise ValueError('Задайте до 20 числовых Telegram ID пользователей.')
    config = {'token':token or load_settings(state)['token'], 'enabled':data['enabled'],
              'allowed_user_ids':list(dict.fromkeys(str(value) for value in raw_ids))}
    if config['enabled'] and (not config['token'] or not config['allowed_user_ids']):
        raise ValueError('Для включения задайте токен и хотя бы один разрешённый ID.')
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=state,prefix='.telegram-',delete=False) as stream:
        json.dump(config,stream)
        stream.flush();os.fsync(stream.fileno())
        temp=stream.name
    os.chmod(temp,0o600)
    os.replace(temp,Path(state)/'telegram-settings.json')
    return public_settings(config)


class TelegramError(Exception):
    pass


class TelegramBot:
    def __init__(self,queue):
        self.queue=queue
        self.session=requests.Session()
        self.config=None

    def call(self,method,**payload):
        # Never log exception URLs: Telegram places the secret token in the path.
        try:
            response=self.session.post('https://api.telegram.org/bot'+self.config['token']+'/'+method,
                                       json=payload,timeout=(5,18))
            data=response.json()
            if response.status_code!=200 or not data.get('ok'):
                raise TelegramError('Telegram API request failed')
            return data.get('result')
        except (requests.RequestException,ValueError,AttributeError):
            raise TelegramError('Telegram API unavailable') from None

    def send(self,chat,text,keyboard=None):
        return self.call('sendMessage',chat_id=chat,text=text[:4000],reply_markup=keyboard or MENU,
                         link_preview_options={'is_disabled':True})

    @staticmethod
    def job_keyboard(row):
        ident=row['id'];state=row['status'];buttons=[]
        if state in ('queued','running'):
            buttons.append({'text':'⏸ Пауза','callback_data':'pause:'+ident})
        if state in ('paused','cancelled','failed'):
            buttons.append({'text':'▶️ Продолжить','callback_data':'resume:'+ident})
        if state in ('queued','running','paused'):
            buttons.append({'text':'⏹ Отменить','callback_data':'cancel:'+ident})
        buttons.append({'text':'📄 Журнал','callback_data':'log:'+ident})
        return {'inline_keyboard':[buttons]}

    def offset_key(self):
        return hashlib.sha256(self.config['token'].encode()).hexdigest()[:16]

    def get_offset(self):
        with self.queue.connect() as db:
            row=db.execute('SELECT value FROM telegram_state WHERE key=?',(self.offset_key(),)).fetchone()
            return int(row['value']) if row else 0

    def set_offset(self,value):
        with self.queue.connect() as db:
            db.execute('INSERT INTO telegram_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                       (self.offset_key(),str(value)))
            db.execute('DELETE FROM telegram_state WHERE key LIKE ?',('receipt:'+self.offset_key()+':%',))

    def active(self,chat):
        with self.queue.connect() as db:
            rows=db.execute("""SELECT * FROM jobs WHERE (telegram_chat_id=? OR telegram_chat_id IS NULL) AND status IN
                ('running','pausing','cancelling','queued','paused') ORDER BY
                CASE WHEN status IN ('running','pausing','cancelling') THEN 0 ELSE 1 END,created LIMIT 10""",(str(chat),)).fetchall()
        if not rows:
            return 'Активных заданий нет.'
        result=[]
        for row in rows:
            urls=json.loads(row['urls'])
            label=urls[0] if urls else 'Переименование видео'
            result.append(f"{row['id'][:8]} · {STATES[row['status']]} · {label}")
            if row['status'] in ('running','pausing','cancelling'):
                log=self.queue.log(row['id']) or ''
                lines=[line.strip() for line in log.splitlines() if line.strip()]
                result.append('Текущие операции Cyberdrop:\n'+'\n'.join(lines[-8:])[-1800:])
        return '\n\n'.join(result)

    def find_job(self,chat,prefix):
        if not re.fullmatch('[0-9a-f]{8,32}',prefix):
            raise ValueError('Укажите ID задания (не менее 8 символов).')
        with self.queue.connect() as db:
            rows=db.execute('SELECT id FROM jobs WHERE (telegram_chat_id=? OR telegram_chat_id IS NULL) AND id LIKE ?', (str(chat),prefix+'%')).fetchall()
        if len(rows)!=1:
            raise ValueError('Задание не найдено или ID неоднозначен.')
        return rows[0]['id']

    def show_jobs(self,chat,action=None):
        params=[str(chat)]
        condition='(telegram_chat_id=? OR telegram_chat_id IS NULL)'
        if action:
            eligible={'pause':('queued','running'), 'resume':('paused','failed','cancelled'),
                      'cancel':('queued','running','paused')}[action]
            condition+=' AND status IN ('+','.join('?' for _ in eligible)+')'
            params.extend(eligible)
        with self.queue.connect() as db:
            rows=db.execute("SELECT * FROM jobs WHERE "+condition+"""
                ORDER BY CASE WHEN status IN ('running','pausing','cancelling') THEN 0
                WHEN status='queued' THEN 1 WHEN status='paused' THEN 2 ELSE 3 END,created DESC LIMIT 10""",params).fetchall()
        if not rows:
            self.send(chat,'Нет заданий для этого действия.' if action else 'Очередь и история пусты.')
            return
        for row in rows[:5]:
            urls=json.loads(row['urls'])
            text=f"{row['id'][:8]} · {STATES[row['status']]}\n"+(urls[0] if urls else 'Переименование видео')
            if action:
                keyboard={'inline_keyboard':[[{'text':{'pause':'⏸ Пауза','resume':'▶️ Продолжить','cancel':'⏹ Отменить'}[action],
                                               'callback_data':action+':'+row['id']}]]}
            else:
                keyboard=self.job_keyboard(row)
            self.send(chat,text,keyboard=keyboard)

    def handle(self,update):
        callback=update.get('callback_query') or {}
        message=update.get('message') or callback.get('message') or {}
        chat=message.get('chat') or {}
        sender=callback.get('from') or message.get('from') or {}
        # Private chats only; group members cannot trigger downloads via an allowed chat.
        if chat.get('type')!='private' or str(sender.get('id')) not in self.config['allowed_user_ids']:
            return
        ident=chat.get('id')
        if str(ident)!=str(sender.get('id')):
            return
        if callback:
            self.call('answerCallbackQuery',callback_query_id=callback['id'])
            action,_,prefix=callback.get('data','').partition(':')
            if action not in ('pause','resume','cancel','log'):
                return
            try:
                job=self.find_job(ident,prefix)
                if action=='log':
                    text=self.queue.log(job) or 'Ожидание запуска…'
                    self.send(ident,'Журнал '+job[:8]+'\n'+text[-3400:])
                else:
                    self.queue.action(job,action)
                    self.send(ident,'Команда принята. Скачанные файлы сохранены.')
                    self.show_jobs(ident)
            except ValueError as exc:
                self.send(ident,str(exc))
            return
        text=message.get('text','').strip()
        if not text:
            return
        menu={'📥 Скачать':'/download','▶️ Активное задание':'/active','📋 Очередь':'/queue',
              '⏸ Пауза':'/choose_pause','▶️ Продолжить':'/choose_resume','⏹ Отменить':'/choose_cancel','❓ Помощь':'/help'}
        command=menu.get(text,text.split()[0].split('@')[0].lower())
        if command in ('/start','/help'):
            self.send(ident,'Отправьте HTTP/HTTPS ссылку или несколько ссылок, по одной в строке. Это будет одно задание.\n'
                      '/active или /status — текущее задание и операции\n/queue — ваша очередь\n'
                      '/pause ID — пауза\n/resume ID — продолжить\n/cancel ID — отменить\n/id — ваш Telegram ID\n'
                      'После обработки всей ссылки придёт итоговое уведомление. По каждому файлу сообщений не будет.')
        elif command=='/id':
            self.send(ident,'Ваш Telegram ID: '+str(sender['id']))
        elif command in ('/active','/status'):
            self.send(ident,self.active(ident))
        elif command=='/download':
            self.send(ident,'Отправьте ссылку на файл/альбом. Для нескольких ссылок — по одной в строке. После всей обработки придёт одно итоговое уведомление.')
        elif command=='/queue':
            self.show_jobs(ident)
        elif command in ('/choose_pause','/choose_resume','/choose_cancel'):
            self.show_jobs(ident,command.removeprefix('/choose_'))
        elif command in ('/pause','/resume','/cancel'):
            try:
                prefix=text.split(maxsplit=1)[1].strip()
                job=self.find_job(ident,prefix)
                self.queue.action(job,command[1:])
                self.send(ident,f'Задание {job[:8]}: команда {command} принята. Скачанные файлы сохранены.')
            except (ValueError,IndexError) as exc:
                self.send(ident,str(exc) if isinstance(exc,ValueError) else 'Укажите ID после команды.')
        else:
            try:
                key=self.offset_key()+':'+str(update['update_id'])
                job=self.queue.add(text,telegram_chat_id=str(ident),telegram_update_key=key)
                if self.queue.status(job) is None:
                    self.send(ident,'Задание из этого сообщения уже было удалено. Для новой загрузки отправьте ссылку новым сообщением.')
                    return
                self.send(ident,f'Задание {job[:8]} добавлено в очередь. Сообщу, когда обработка всей ссылки закончится.')
            except ValueError as exc:
                self.send(ident,str(exc))

    def notify_finished(self):
        with self.queue.connect() as db:
            rows=db.execute("""SELECT * FROM jobs WHERE telegram_chat_id IS NOT NULL
                AND telegram_notified=0 AND status IN ('done','failed','cancelled') ORDER BY finished LIMIT 20""").fetchall()
        for row in rows:
            if self.queue.stop.is_set():
                break
            if self.queue.status(row['id']) != row['status']:
                continue
            if row['telegram_chat_id'] not in self.config['allowed_user_ids']:
                continue
            if row['status']=='done':
                title='Обработка ссылки закончена. Задание завершено.'
            elif row['status']=='failed':
                title='Задание завершилось с ошибкой. Можно продолжить: /resume '+row['id'][:8]
            else:
                title='Задание отменено. Скачанные файлы сохранены. Продолжить: /resume '+row['id'][:8]
            log=self.queue.log(row['id']) or ''
            # Include final statistics, not one notification per individual download.
            lines=[line.strip() for line in log.splitlines() if line.strip()]
            summary='\n'.join(lines[-18:])[-2000:]
            self.send(row['telegram_chat_id'],title+'\nID: '+row['id'][:8]+'\n'+
                      '\n'.join(json.loads(row['urls'])[:3])+'\n\nИтог Cyberdrop:\n'+summary,keyboard=self.job_keyboard(row))
            with self.queue.connect() as db:
                db.execute('UPDATE jobs SET telegram_notified=1 WHERE id=? AND status=? AND finished IS ?',
                           (row['id'],row['status'],row['finished']))

    def run(self):
        try:
            while not self.queue.stop.is_set():
                self.config=load_settings(self.queue.state)
                if not self.config['enabled'] or not self.config['token'] or not self.config['allowed_user_ids']:
                    self.queue.stop.wait(3)
                    continue
                try:
                    self.notify_finished()
                    updates=self.call('getUpdates',offset=self.get_offset(),timeout=8,limit=20,allowed_updates=['message','callback_query']) or []
                    for update in updates:
                        latest=load_settings(self.queue.state)
                        if latest['token']!=self.config['token'] or not latest['enabled']:
                            break
                        self.config=latest
                        if self.queue.stop.is_set():
                            break
                        self.handle(update)
                        self.set_offset(update['update_id']+1)
                    self.queue.stop.wait(.5)
                except TelegramError:
                    LOG.warning('Telegram unavailable or bot already has another poller/webhook; retrying in 15 seconds')
                    self.queue.stop.wait(15)
                except Exception:
                    LOG.error('Telegram processing failed; retrying without exposing message contents or credentials')
                    self.queue.stop.wait(15)
        finally:
            self.session.close()
