"""Read-only ThePornDB matching: OSHASH first, conservative metadata second."""
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import struct
from datetime import datetime

import requests
from rapidfuzz import fuzz

LOG = logging.getLogger(__name__)
API = 'https://api.theporndb.net'


def load_settings(state=None):
    path = Path(state or os.environ.get('STATE_DIR', '.')) / 'metadata-settings.json'
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    return {'enabled': data.get('enabled', True) is True,
            'tpdb_api_key': data.get('tpdb_api_key') or os.environ.get('TPDB_API_KEY', '')}


def oshash(path):
    """OpenSubtitles hash: size + uint64 little-endian sums of first/last 64 KiB."""
    with open(path, 'rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        if size < 131072:
            return None
        first = stream.read(65536)
        stream.seek(-65536, os.SEEK_END)
        last = stream.read(65536)
    if len(first) != 65536 or len(last) != 65536:
        return None
    total = size + sum(x[0] for x in struct.iter_unpack('<Q', first + last))
    return f'{total & 0xffffffffffffffff:016x}'


def date_value(raw):
    try:
        return datetime.strptime(raw[:10], '%Y-%m-%d').strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def words(value):
    text = re.sub(r'\b(?:19|20)\d{2}[-.]\d{2}[-.]\d{2}\b', ' ', str(value).casefold())
    text = re.sub(r'\b(?:2160p|1080p|720p|480p|x264|x265|h264|h265|hevc|webdl|webrip)\b', ' ', text)
    return ' '.join(re.findall(r'\w+', text))


class MetadataClient:
    def __init__(self, cache=None, settings=None):
        self.settings = settings if settings is not None else load_settings()
        self.enabled = self.settings['enabled'] and bool(self.settings.get('tpdb_api_key'))
        self.available = self.enabled
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({'Authorization': 'Bearer '+self.settings.get('tpdb_api_key', ''),
                                     'Accept': 'application/json', 'User-Agent':'BatinaPapka/1.2'})

    def query(self, endpoint, params):
        if not self.available:
            return []
        key = 'tpdb:' + hashlib.sha256((self.settings['tpdb_api_key'] + endpoint +
                                       json.dumps(params, sort_keys=True)).encode()).hexdigest()
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        try:
            response = self.session.get(API + '/' + endpoint, params=params, timeout=(5, 15), allow_redirects=False)
            if response.status_code in (401,403,429):
                self.available = False
            if response.status_code != 200:
                LOG.warning('ThePornDB HTTP %s; falling back to search',response.status_code)
                return []
            data = response.json().get('data', [])
            if not isinstance(data, list):
                return []
            data = [item for item in data if isinstance(item, dict)]
            if self.cache:
                self.cache.set(key, data)
            return data
        except (requests.RequestException, ValueError, AttributeError):
            LOG.warning('ThePornDB unavailable; falling back to search')
            self.available = False
            return []

    @staticmethod
    def result(item, source):
        title = item.get('title')
        date = date_value(item.get('date'))
        if not isinstance(title,str) or len(title.strip()) < 3 or not date:
            return None
        return date, title.strip(), source

    def lookup(self, path, original, embedded=None, duration=None):
        if not self.enabled:
            return None
        digest = oshash(path)
        if digest:
            for endpoint in ('scenes','movies'):
                matches = self.query(endpoint, {'hash':digest,'hashType':'OSHASH','limit':10})
                valid = []
                for item in matches:
                    hashes = item.get('hashes')
                    # Require the returned fingerprint to match, even if an API filter
                    # changes behavior or returns an unfiltered result list.
                    if not isinstance(hashes, list) or not any(isinstance(h,dict) and h.get('type') == 'OSHASH' and
                                                     str(h.get('hash','')).lower() == digest for h in hashes):
                        continue
                    result = self.result(item, 'tpdb-oshash')
                    if result:
                        valid.append(result)
                valid = list(dict.fromkeys(valid))
                if len(valid) == 1:
                    return valid[0]
        # No hash match: metadata names must be nearly exact and unambiguous.
        for query in dict.fromkeys(q for q in (embedded, original) if q and len(words(q)) >= 8):
            candidates = []
            for item in self.query('scenes', {'parse':query,'limit':10}):
                result = self.result(item, 'tpdb-metadata')
                if not result:
                    continue
                title = item['title']
                cast = ' '.join(p.get('name','') for p in item.get('performers',[]) if isinstance(p,dict))
                studio = item.get('site') or {}
                studio = studio.get('name','') if isinstance(studio,dict) else ''
                comparisons = (title, cast+' '+title, studio+' '+cast+' '+title)
                score = max(fuzz.token_sort_ratio(words(query),words(value)) for value in comparisons)
                if score < 94:
                    continue
                dates = re.findall(r'\b\d{4}[-.]\d{2}[-.]\d{2}\b',query)
                if dates and result[0] not in [date.replace('.','-') for date in dates]:
                    continue
                try:
                    if duration and item.get('duration') and abs(float(item['duration'])-duration)>max(10,duration*.03):
                        continue
                except (ValueError,TypeError):
                    continue
                candidates.append((score,result))
            candidates.sort(key=lambda value:value[0],reverse=True)
            if candidates and (len(candidates)==1 or candidates[0][0]-candidates[1][0]>=5):
                return candidates[0][1]
        return None
