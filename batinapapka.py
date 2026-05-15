"""Rename adult video files based on online search results (Brave Search).

Reads filenames from a directory, queries Brave Search, and renames each file
to "<YYYY-MM-DD> <cleaned-title>.<ext>". Falls back to the file's mtime and a
sanitized version of the original name when the search yields nothing usable.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from rapidfuzz import fuzz
from unidecode import unidecode

LOG_FILE = "file_renamer.log"
RENAMED_FILES_LOG = "renamed_files.txt"
CACHE_FILE = "search_cache.json.gz"
CACHE_TTL_DAYS = 30
SIMILARITY_THRESHOLD = 0.60
# Stricter threshold for filenames with no strong signal (no Capitalized
# names, no year) — we don't trust a borderline match when there's nothing
# in the input to disambiguate it.
SIMILARITY_THRESHOLD_WEAK_SIGNAL = 0.70
MIN_FREE_SPACE_BYTES = 100 * 1024 * 1024
REQUEST_TIMEOUT = 10
FFPROBE_TIMEOUT = 5
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_RESULTS_PER_QUERY = 20  # Brave Search API caps `count` at 20.

# Telegram media filenames are 18-digit integers — searching them is pointless.
NUMERIC_FILENAME_LEN = 18

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpeg", ".mpg",
})

# Tube/aggregator/cam hostings — their names in a title are SEO boilerplate
# ("Big Buck Bunny - Pornhub"), not part of the actual content, so we strip
# them from both filenames-to-search and chosen titles.
TUBE_AND_CAM_NAMES = (
    # Mainstream — almost always noise when present in an adult-video search.
    "YouTube", "Vimeo", "Dailymotion", "Twitch", "Facebook", "Instagram",
    "Twitter", "TikTok",
    # Major adult tubes.
    "Pornhub", "PornhubPremium", "Pornhub Premium", "Xvideos", "XNXX",
    "YouPorn", "RedTube", "Porn.com", "XHamster", "XHamster Premium",
    "xHamsterLive", "SpankBang", "TNAFlix", "Tube8", "JizzBunker",
    "KeezMovies", "Nuvid", "DrTuber", "EPORNER", "Beeg", "SunPorno",
    "Porn300", "PornOne", "MegaPorn", "EMPFlix", "Txxx", "HDZog",
    "AlphaPorno", "PornDoe", "PornTrex", "PornHD", "Spankwire", "HClips",
    "HQporner", "Fapality", "AnyPorn", "AnySex", "HotMovs", "BravoTube",
    "HellPorno", "HotShame", "Pichunter", "PornHat", "Vjav", "Vporn",
    "4Tube", "Fux", "PornHub", "Fapcat", "TubeGalore", "PornDig",
    "Yespornplease", "Shooshtime", "24porn", "IcePorn", "GotPorn",
    "BoyFriendTV", "EmpFlix",
    # JAV-focused tubes.
    "JavHD", "JavBus", "JavLibrary", "JavGuru",
    # Cam sites.
    "Chaturbate", "CamSoda", "MyFreeCams", "LiveJasmin", "BongaCams",
    "Stripchat", "Cam4", "Camster", "ImLive",
)

# Studio/brand names — these ARE meaningful parts of titles ("Brazzers", "Blacked",
# "OnlyFans") and must NOT be stripped. We still want a host bonus for hits on
# their domains, so they show up in KNOWN_HOSTS below.
STUDIO_NAMES = (
    "Brazzers", "BangBros", "Mofos", "Reality Kings", "RealityKings",
    "Naughty America", "NaughtyAmerica", "Digital Playground",
    "DigitalPlayground", "Evil Angel", "EvilAngel", "Jules Jordan",
    "JulesJordan", "Hard X", "HardX", "Babes", "Twistys", "Blacked",
    "Tushy", "Vixen", "Deeper", "Slayed", "Kink", "BangBus", "TeamSkeet",
    "Mile High Media", "Wicked", "Burning Angel", "BurningAngel",
    "Met-Art", "Met Art", "MetArt", "ATKgalleria", "Mr. Skin",
    # VR studios.
    "VRBangers", "WankzVR", "BadoinkVR", "Naughty America VR",
    "NaughtyAmericaVR", "VRConk", "SLR", "SexLikeReal", "POVR",
    "VirtualRealPorn", "VRPorn", "AdultTime", "Adult Time",
    # Creator platforms — treated as brands because creators are usually
    # identified as e.g. "Mia Khalifa OnlyFans".
    "OnlyFans", "ManyVids", "Manyvids", "Fansly", "ModelHub", "Fancentro",
    "LoyalFans", "JustForFans", "IsMyGirl", "iWantClips", "iWantEmpire",
    "Clips4Sale", "Pocketstars",
)

# Backwards-compat alias for any external reference.
VIDEO_HOSTINGS = TUBE_AND_CAM_NAMES + STUDIO_NAMES

# Authoritative hosts (tubes + studios + creator platforms). A result hit on
# any of these gets a +0.10 confidence bonus in choose_best.
KNOWN_HOSTS = frozenset({
    # Tubes.
    "pornhub.com", "xvideos.com", "xnxx.com", "xhamster.com", "youporn.com",
    "redtube.com", "spankbang.com", "eporner.com", "porntrex.com",
    "tnaflix.com", "tube8.com", "drtuber.com", "beeg.com", "hclips.com",
    "hqporner.com", "spankwire.com", "4tube.com", "hotmovs.com",
    "bravotube.com", "anyporn.com", "anysex.com", "vjav.com", "vporn.com",
    "hellporno.com", "hotshame.com", "pichunter.com", "pornhat.com",
    "fapality.com", "txxx.com", "hdzog.com", "alphaporno.com",
    "sunporno.com", "porn300.com", "pornone.com", "empflix.com",
    "porndig.com", "yespornplease.com", "shooshtime.com", "gotporn.com",
    "iceporn.com", "javhd.com", "javbus.com", "jav.guru",
    # Cam.
    "chaturbate.com", "camsoda.com", "myfreecams.com", "livejasmin.com",
    "bongacams.com", "stripchat.com", "cam4.com",
    # Creator / paywalled.
    "onlyfans.com", "manyvids.com", "fansly.com", "fancentro.com",
    "loyalfans.com", "justfor.fans", "ismygirl.com", "iwantclips.com",
    "clips4sale.com",
    # Studios.
    "brazzers.com", "realitykings.com", "bangbros.com", "mofos.com",
    "naughtyamerica.com", "digitalplayground.com", "evilangel.com",
    "julesjordan.com", "hardx.com", "babes.com", "twistys.com",
    "blacked.com", "tushy.com", "vixen.com", "deeper.com", "slayed.com",
    "kink.com", "teamskeet.com", "adulttime.com", "wicked.com",
    "burningangel.com", "met-art.com", "atkgalleria.com",
    # VR.
    "vrbangers.com", "wankzvr.com", "badoinkvr.com", "vrconk.com",
    "sexlikereal.com", "povr.com", "virtualrealporn.com", "vrporn.com",
})

# Exact hostnames (or their subdomains) to drop from search results.
EXCLUDED_HOSTS = frozenset({
    "wikipedia.org", "imdb.com",
    "kinopoisk.ru", "ivi.ru", "megogo.net", "okko.tv", "more.tv", "tvzavr.ru",
    "reddit.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "pinterest.com", "linkedin.com", "tumblr.com",
})

_CLEAN_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(360p|480p|720p|1080p|2160p|4K|8K|HD|FHD|UHD)\b",
        r"\b(HD|HQ|HDRip|BRRip|DVDRip|WEBRip|BluRay)\b",
        r"\b(x264|h264|x265|h265|hevc|avc|mp3|aac|ac3|dts|flac)\b",
        r"\b(MP4|MKV|AVI|WMV|FLV|MPEG|MPG)\b",
        r"\b(Official|Video|Full|Complete|Scene|Version|Edit|Cut)\b",
        r"[\[\(\{].*?[\]\)\}]",
        r"\b\d{2,4}[-/.]\d{2}[-/.]\d{2,4}\b",
        r"\b(com|net|org|xxx)\b",
        r"\b\d{3,4}x\d{3,4}\b",
        r"\b\d+(\.\d+)?\s*(MB|GB|TB)\b",
        # SEO boilerplate from adult-video site titles. These are noise added
        # by the host, not part of the actual video name.
        r"\b(Porn|Sex|Adult|XXX)\s+(Videos?|Movies?|Tube|Clips?|Films?)\b",
        r"\b(Free|Watch|Download|Best|Top|Hot|New|Latest|Premium)\b",
        r"\b(Online|Streaming|HD|Quality)\b",
    )
]

_HOSTINGS_RE = re.compile(
    r"\b(" + "|".join(re.escape(h) for h in TUBE_AND_CAM_NAMES) + r")\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NON_WORD_RE = re.compile(r"[^\w\s-]")
_WS_RE = re.compile(r"\s+")
_LONG_NUM_RE = re.compile(r"\b\d{6,}\b")
_SINGLE_LETTER_RE = re.compile(r"\b[a-zA-Z]\b")
_SEPARATOR_RE = re.compile(r"[_\-]+")
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*]')
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Two or more consecutive Capitalized tokens — a strong cue for a person/studio
# name in a filename (e.g. "Mia.Khalifa.full.mp4" → "Mia", "Khalifa").
_NAME_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

logger = logging.getLogger(__name__)


class CompressedCache:
    """On-disk cache: gzipped JSON, TTL-checked, atomic save with file lock."""

    def __init__(self, path: str, ttl_days: int = CACHE_TTL_DAYS):
        self.path = Path(path)
        self.ttl_days = ttl_days
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with gzip.open(self.path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("cache root is not a dict")
            self.cache = data.get("cache", {})
            self.timestamps = data.get("timestamps", {})
            self._cleanup_expired()
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.error("Corrupt cache (%s) — starting fresh", e)
            self.cache, self.timestamps = {}, {}

    def save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "wb") as raw:
                fcntl.flock(raw.fileno(), fcntl.LOCK_EX)
                with gzip.GzipFile(fileobj=raw, mode="wb") as gz:
                    payload = json.dumps(
                        {"cache": self.cache, "timestamps": self.timestamps},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    gz.write(payload)
            os.replace(tmp, self.path)
        except OSError as e:
            logger.error("Cache save failed: %s", e)
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def _cleanup_expired(self) -> None:
        now = datetime.now()
        expired = [
            k for k, ts in self.timestamps.items()
            if (now - datetime.fromisoformat(ts)).days > self.ttl_days
        ]
        for k in expired:
            self.cache.pop(k, None)
            self.timestamps.pop(k, None)

    def get(self, key: str) -> Optional[Any]:
        ts = self.timestamps.get(key)
        if not ts:
            return None
        if (datetime.now() - datetime.fromisoformat(ts)).days > self.ttl_days:
            return None
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = value
        self.timestamps[key] = datetime.now().isoformat()


def _normalized_host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _host_excluded(host: str) -> bool:
    for blocked in EXCLUDED_HOSTS:
        if host == blocked or host.endswith("." + blocked):
            return True
    return False


def _host_known(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in KNOWN_HOSTS)


def _parse_page_age(raw: str) -> Optional[str]:
    """Return YYYY-MM-DD if `raw` looks like one, else None.

    Brave occasionally returns ISO 8601 with a T suffix, sometimes a relative
    string ("3 days ago"), sometimes year-only. We only trust an explicit
    YYYY-MM-DD substring; anything else falls back to mtime in the caller.
    """
    if not raw:
        return None
    m = _ISO_DATE_RE.search(raw)
    return m.group() if m else None


def _normalize_results(items: Iterable[dict]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rank, item in enumerate(items):
        url = item.get("url", "") or ""
        host = _normalized_host(url)
        if _host_excluded(host):
            continue
        out.append({
            "title": item.get("title", "") or "",
            "page_age": item.get("page_age") or "",
            "url": url,
            "host": host,
            "rank": rank,
        })
    return out


class BraveSearchClient:
    def __init__(self, api_key: str, cache: CompressedCache):
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        })
        self._last_request_ts = 0.0
        self.retry_count = 3

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_ts
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_ts = time.time()

    def _key(self, full_query: str) -> str:
        return hashlib.sha1(f"brave:{full_query}".encode("utf-8")).hexdigest()

    def search(self, full_query: str) -> List[Dict[str, Any]]:
        """Run one Brave search. `full_query` must already be the literal text
        sent as `q=` (templating happens in the caller)."""
        if not full_query.strip():
            return []
        key = self._key(full_query)
        cached = self.cache.get(key)
        if cached is not None:
            logger.info("Brave cache hit: %s", full_query)
            return cached

        for attempt in range(1, self.retry_count + 1):
            try:
                self._throttle()
                resp = self.session.get(
                    BRAVE_SEARCH_API_URL,
                    params={"q": full_query, "count": BRAVE_RESULTS_PER_QUERY, "safesearch": "off"},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code in {401, 403}:
                    logger.error("Brave: invalid API key (HTTP %s)", resp.status_code)
                    return []
                if resp.status_code == 429:
                    delay = float(resp.headers.get("Retry-After", attempt * 2))
                    logger.warning("Brave rate-limited, sleeping %.1fs", delay)
                    time.sleep(delay)
                    continue
                if resp.status_code >= 500:
                    logger.warning("Brave server error %s (attempt %d/%d)",
                                   resp.status_code, attempt, self.retry_count)
                    time.sleep(attempt * 2)
                    continue
                resp.raise_for_status()
                results = _normalize_results(
                    resp.json().get("web", {}).get("results", [])
                )
                self.cache.set(key, results)
                return results
            except requests.RequestException as e:
                logger.warning("Brave request failed (%d/%d): %s",
                               attempt, self.retry_count, e)
                time.sleep(attempt * 2)
        return []


def ffprobe_title(path: str) -> Optional[str]:
    """Return the embedded `title` tag from the media container, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format_tags=title",
             "-of", "json", path],
            capture_output=True, text=True, timeout=FFPROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("ffprobe failed on %s: %s", path, e)
        return None
    if out.returncode != 0:
        return None
    try:
        tags = json.loads(out.stdout or "{}").get("format", {}).get("tags", {}) or {}
    except json.JSONDecodeError:
        return None
    title = (tags.get("title") or tags.get("TITLE") or "").strip()
    return title or None


class TitleProcessor:
    @lru_cache(maxsize=4096)
    def clean_title(self, title: str, is_original_file: bool = False) -> str:
        title = title.split("|", 1)[0].strip()
        title = _HOSTINGS_RE.sub("", title)
        for pat in _CLEAN_PATTERNS:
            title = pat.sub("", title)

        year_match = _YEAR_RE.search(title)
        year = year_match.group() if year_match else None

        title = _NON_WORD_RE.sub(" ", title)
        title = _WS_RE.sub(" ", title)
        title = unicodedata.normalize("NFKD", title)
        title = unidecode(title).encode("ascii", errors="ignore").decode()

        if is_original_file:
            title = _LONG_NUM_RE.sub("", title)
            title = _SINGLE_LETTER_RE.sub("", title)
            title = _SEPARATOR_RE.sub(" ", title)

        if year and year not in title:
            title = f"{title} {year}"

        title = _WS_RE.sub(" ", title).strip()
        if len(title) < 3:
            return "unnamed_video" if is_original_file else title
        return title

    def _similarity(self, a: str, b: str) -> float:
        ca = self.clean_title(a)
        cb = self.clean_title(b)
        if not ca or not cb:
            return 0.0
        # Blend three rapidfuzz metrics — each catches different mismatches.
        # token_set_ratio: extra/reordered tokens; partial_ratio: substring
        # contained in a longer title; WRatio: rapidfuzz's own blended score.
        return max(
            fuzz.token_set_ratio(ca, cb),
            fuzz.partial_ratio(ca, cb),
            fuzz.WRatio(ca, cb) * 0.9,
        ) / 100.0

    def choose_best(
        self,
        original: str,
        results: List[Dict[str, Any]],
        year_hint: Optional[str] = None,
        required_tokens: Optional[Set[str]] = None,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """Pick the most likely title/date from a result list.

        Returns (date, title, score). When no result clears SIMILARITY_THRESHOLD,
        returns (None, None, best_observed_score) — the score is exposed so the
        caller can decide whether to try a wider query.
        """
        best_score, best_title, best_date = 0.0, None, None
        required_lc = {t.lower() for t in (required_tokens or set())}

        for r in results:
            title = r.get("title", "")
            if not title:
                continue
            title_lc = title.lower()
            # Required tokens (e.g. names extracted from the filename) must
            # appear in the candidate title — otherwise it's a different video.
            if required_lc and not all(t in title_lc for t in required_lc):
                continue

            score = self._similarity(original, title)

            # Length sanity — very short or very long titles are usually junk.
            if len(title) < 10 or len(title) > 100:
                score *= 0.8

            # Bonus for top-of-page rank: +0..+0.10 over the top 10 hits.
            rank = r.get("rank", 0)
            score += max(0.0, (10 - rank)) * 0.01

            # Bonus for an authoritative adult-video host.
            if _host_known(r.get("host", "")):
                score += 0.10

            # Year sanity: if the filename embeds a year and the page age has
            # one too, reward agreement and lightly penalize a big mismatch.
            page_date = _parse_page_age(r.get("page_age", ""))
            if year_hint and page_date:
                page_year = page_date[:4]
                if page_year == year_hint:
                    score += 0.05
                elif abs(int(page_year) - int(year_hint)) > 2:
                    score -= 0.10

            if score > best_score:
                best_score = score
                best_title = self.clean_title(title)
                best_date = page_date

        # Filenames with no Capitalized name tokens and no year are
        # "low-signal" — they can match almost anything, so demand a higher
        # threshold to avoid false positives like IMG_20240101_xxx.
        has_signal = bool(required_tokens) or bool(year_hint)
        threshold = SIMILARITY_THRESHOLD if has_signal else SIMILARITY_THRESHOLD_WEAK_SIGNAL
        if best_score < threshold:
            return None, None, best_score
        return best_date, best_title, best_score


@dataclass
class Stats:
    processed: int = 0
    renamed: int = 0
    skipped: int = 0
    errors: int = 0
    brave_queries_hit: int = 0
    brave_queries_sent: int = 0
    no_search_results: int = 0
    using_original_name: int = 0
    used_embedded_title: int = 0


class ProgressReporter:
    """Live, single-line progress display for interactive runs.

    Enabled only when stderr is a TTY (so cron and `docker logs` get plain
    multi-line output instead of a stream of carriage returns). announce()
    prints a permanent line above the rotating status bar; status() updates
    the bar in place.
    """

    _ANSI_CLEAR_LINE = "\r\x1b[2K"

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.total = 0
        self.current = 0
        self._last_status = ""

    def start(self, total: int) -> None:
        self.total = total
        self.current = 0
        if self.enabled:
            sys.stderr.write(f"Found {total} candidate file(s)\n")
            sys.stderr.flush()

    def tick(self) -> None:
        self.current += 1

    def _term_width(self) -> int:
        try:
            return shutil.get_terminal_size((80, 20)).columns
        except OSError:
            return 80

    def status(self, msg: str) -> None:
        if not self.enabled:
            return
        line = f"[{self.current}/{self.total}] {msg}"
        width = self._term_width()
        if len(line) > width:
            line = line[: width - 3] + "..."
        sys.stderr.write(self._ANSI_CLEAR_LINE + line)
        sys.stderr.flush()
        self._last_status = line

    def announce(self, msg: str, error: bool = False) -> None:
        prefix = f"[{self.current}/{self.total}] " if self.total else ""
        line = prefix + msg
        if self.enabled:
            sys.stderr.write(self._ANSI_CLEAR_LINE + line + "\n")
            sys.stderr.flush()
            if self._last_status:
                sys.stderr.write(self._last_status)
                sys.stderr.flush()
        else:
            stream = sys.stderr if error else sys.stdout
            stream.write(line + "\n")
            stream.flush()

    def finish(self) -> None:
        if self.enabled:
            sys.stderr.write(self._ANSI_CLEAR_LINE)
            sys.stderr.flush()


# Query variants are tried in order; we stop as soon as choose_best returns a
# title that clears the threshold. Each variant produces a different `q=` text
# (and therefore a different cache key), so caching stays consistent.
_QUERY_VARIANTS = (
    '"{q}" adult video',  # strict phrase + genre bias
    "{q} adult video",    # loose, still genre-biased
    '"{q}"',              # strict phrase, generic
    "{q}",                # loose, generic
)


def _extract_year(text: str) -> Optional[str]:
    m = _YEAR_RE.search(text)
    return m.group() if m else None


def _required_tokens(name: str) -> Set[str]:
    """Heuristic: if a filename has two or more Capitalized tokens of length>=3,
    treat them as a strong signal (person/studio name) and require the chosen
    title to contain all of them. Returns an empty set otherwise."""
    tokens = _NAME_TOKEN_RE.findall(name.replace("_", " ").replace(".", " "))
    deduped = []
    seen = set()
    for t in tokens:
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        deduped.append(t)
    return set(deduped) if len(deduped) >= 2 else set()


class VideoFileRenamer:
    def __init__(
        self,
        brave_client: BraveSearchClient,
        cache: CompressedCache,
        progress: Optional[ProgressReporter] = None,
    ):
        self.cache = cache
        self.brave_client = brave_client
        self.title = TitleProcessor()
        self.renamed_files = self._load_renamed()
        self.stats = Stats()
        self.progress = progress or ProgressReporter(enabled=False)

    def _load_renamed(self) -> Set[str]:
        if not os.path.exists(RENAMED_FILES_LOG):
            return set()
        try:
            with open(RENAMED_FILES_LOG, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except OSError as e:
            logger.error("Cannot load %s: %s", RENAMED_FILES_LOG, e)
            return set()

    def _mark_renamed(self, name: str) -> None:
        try:
            with open(RENAMED_FILES_LOG, "a", encoding="utf-8") as f:
                f.write(name + "\n")
            self.renamed_files.add(name)
        except OSError as e:
            logger.error("Cannot append to %s: %s", RENAMED_FILES_LOG, e)

    @staticmethod
    def _safe(name: str) -> str:
        name = _UNSAFE_FILENAME_RE.sub("_", name)
        if len(name) > 255:
            base, ext = os.path.splitext(name)
            name = base[: 255 - len(ext)] + ext
        return name

    @staticmethod
    def _unique(directory: str, name: str) -> str:
        base, ext = os.path.splitext(name)
        candidate, counter = name, 1
        while os.path.exists(os.path.join(directory, candidate)):
            candidate = f"{base}_{counter}{ext}"
            counter += 1
        return candidate

    def _should_process(self, filename: str, force: bool) -> bool:
        if filename.startswith("."):
            return False
        ext = os.path.splitext(filename)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            return False
        if not force and filename in self.renamed_files:
            self.stats.skipped += 1
            return False
        stem = os.path.splitext(filename)[0]
        if len(stem) == NUMERIC_FILENAME_LEN and stem.isdigit():
            self.stats.skipped += 1
            return False
        return True

    def _dedup(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen, unique = set(), []
        for r in results:
            key = (r.get("url", ""), r.get("title", "").lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        # Preserve original rank for the bonus in choose_best.
        for idx, r in enumerate(unique):
            r["rank"] = idx
        return unique

    def _candidate_queries(self, original_base: str, embedded: Optional[str]) -> List[str]:
        """Produce a deduplicated list of `q=` strings to try in order.

        - Start with the embedded title (most authoritative), if present.
        - Then the cleaned filename (the high-recall form).
        - Then the raw filename, which keeps any tokens our cleaner stripped.
        Each base is run through _QUERY_VARIANTS (strict→loose, genre→generic).
        """
        bases: List[str] = []
        if embedded:
            bases.append(embedded.strip())
        cleaned = self.title.clean_title(original_base, is_original_file=True)
        if cleaned and cleaned != "unnamed_video":
            bases.append(cleaned)
        # Raw base may still help when our cleaner is over-aggressive.
        if original_base.strip() and original_base.strip() not in bases:
            bases.append(original_base.strip())

        queries: List[str] = []
        for base in bases:
            for tmpl in _QUERY_VARIANTS:
                q = tmpl.format(q=base)
                if q not in queries:
                    queries.append(q)
        return queries

    def _cascade_search(
        self,
        original_base: str,
        embedded: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
        """Try queries in order. Return the first (date, title) that clears the
        threshold, plus the merged result list for diagnostics."""
        year_hint = _extract_year(original_base)
        required = _required_tokens(original_base)
        # The "original" against which similarity is measured stays the same
        # across all variants — we're trying different queries, not different
        # references for similarity.
        merged: List[Dict[str, Any]] = []
        for q in self._candidate_queries(original_base, embedded):
            self.stats.brave_queries_sent += 1
            results = self.brave_client.search(q)
            if not results:
                continue
            merged.extend(results)
            merged = self._dedup(merged)
            date, title, score = self.title.choose_best(
                original_base, merged,
                year_hint=year_hint,
                required_tokens=required,
            )
            logger.info("Query %r: %d results, best score %.2f", q, len(results), score)
            if title:
                self.stats.brave_queries_hit += 1
                return date, title, merged
        return None, None, merged

    def _backup_mapping(self, directory: str, old: str, new: str) -> None:
        backup = Path(directory) / ".filename_mapping.json"
        mapping: Dict[str, str] = {}
        if backup.exists():
            try:
                mapping = json.loads(backup.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                logger.error("Cannot read %s: %s", backup, e)
        mapping[new] = old
        try:
            backup.write_text(
                json.dumps(mapping, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error("Cannot write %s: %s", backup, e)

    def process(self, directory: str, force: bool = False) -> None:
        if not os.path.isdir(directory):
            raise ValueError(f"Not a directory: {directory}")
        if shutil.disk_usage(directory).free < MIN_FREE_SPACE_BYTES:
            raise OSError(f"Less than {MIN_FREE_SPACE_BYTES // (1024 * 1024)}MB free")

        logger.info("Starting rename in %s", directory)
        started = time.time()

        # Pre-flight: enumerate the candidate videos once so the progress
        # reporter has a real total to show ("[3/42] processing X…").
        candidates: List[str] = []
        for filename in sorted(os.listdir(directory)):
            if filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                candidates.append(filename)

        self.progress.start(len(candidates))

        for filename in candidates:
            self.progress.tick()
            try:
                if not self._should_process(filename, force):
                    self.progress.announce(f"skip {filename}")
                    continue
                self.stats.processed += 1

                path = os.path.join(directory, filename)
                base = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]

                try:
                    with open(path, "rb"):
                        pass
                except PermissionError:
                    self.progress.announce(f"ERROR locked: {filename}", error=True)
                    logger.error("File locked: %s", filename)
                    self.stats.errors += 1
                    continue

                file_date = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")

                self.progress.status(f"{filename}")
                logger.info("Processing %s", filename)

                embedded = ffprobe_title(path)
                if embedded:
                    logger.info('Embedded title: "%s"', embedded)
                    self.stats.used_embedded_title += 1

                self.progress.status(f"searching: {filename}")
                date, new_title, merged = self._cascade_search(base, embedded)

                if not new_title:
                    if embedded and len(embedded) >= 3:
                        new_title = self.title.clean_title(embedded)
                        if not new_title:
                            new_title = self.title.clean_title(base, is_original_file=True)
                    else:
                        new_title = self.title.clean_title(base, is_original_file=True)
                    date = file_date
                    if merged:
                        self.stats.using_original_name += 1
                    else:
                        self.stats.no_search_results += 1
                elif not date:
                    date = file_date

                new_filename = self._unique(
                    directory, self._safe(f"{date} {new_title}{ext}")
                )
                new_path = os.path.join(directory, new_filename)
                self._backup_mapping(directory, filename, new_filename)
                shutil.move(path, new_path)
                self._mark_renamed(new_filename)
                logger.info('Renamed "%s" -> "%s"', filename, new_filename)
                self.progress.announce(f"renamed {filename} -> {new_filename}")
                self.stats.renamed += 1

            except Exception as e:  # noqa: BLE001 — never let one bad file kill the run
                logger.exception("Error processing %s: %s", filename, e)
                self.progress.announce(f"ERROR {filename}: {e}", error=True)
                self.stats.errors += 1

        self.progress.finish()
        self.cache.save()
        logger.info(
            "Done in %.2fs. Stats: %s",
            time.time() - started, json.dumps(asdict(self.stats)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename adult video files based on Brave Search results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "directories", nargs="+",
        help="One or more directories with video files. Pass --recursive to "
             "descend into subdirectories.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Descend into every subdirectory of each given path.",
    )
    parser.add_argument(
        "--api-key", default=os.getenv("BRAVE_API_KEY"),
        help="Brave Search API key (or BRAVE_API_KEY env var)",
    )
    parser.add_argument("--clean-cache", action="store_true",
                        help="Drop the on-disk cache before processing")
    parser.add_argument("--force", action="store_true",
                        help="Re-process files already listed in renamed_files.txt")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Stream every INFO event to stderr (disables the "
                             "live progress bar)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Silence the progress bar even on a TTY")
    parser.add_argument("--debug", action="store_true",
                        help="Maximum verbosity: DEBUG-level events to stderr")
    return parser.parse_args()


def _expand_targets(roots: List[str], recursive: bool) -> List[str]:
    """Resolve each input path into a list of directories to process.

    Non-existent or unreadable paths are dropped with a logged warning so the
    rest of the run still completes. With --recursive every (sub)directory
    that contains at least one file becomes a target; without it we only
    process the paths given on the command line.
    """
    targets: List[str] = []
    seen: Set[str] = set()
    for root in roots:
        if not os.path.isdir(root):
            logger.warning("Skipping %s — not a directory", root)
            continue
        if not os.access(root, os.W_OK):
            logger.warning("Skipping %s — no write access", root)
            continue
        if recursive:
            for current, _, files in os.walk(root):
                if not files:
                    continue
                real = os.path.realpath(current)
                if real in seen:
                    continue
                seen.add(real)
                targets.append(current)
        else:
            real = os.path.realpath(root)
            if real not in seen:
                seen.add(real)
                targets.append(root)
    return targets


def configure_logging(debug: bool, verbose: bool, show_progress: bool) -> None:
    """Wire up the logger.

    - File handler (rotating) always gets every INFO+ event for the archive.
    - Console handler streams to stderr; level depends on flags:
        --debug         → DEBUG
        --verbose       → INFO
        progress on     → WARNING (keeps the progress line uncluttered;
                                   the file log still has every detail)
        default         → INFO
    """
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=7)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    if debug:
        console_level = logging.DEBUG
    elif verbose:
        console_level = logging.INFO
    elif show_progress:
        console_level = logging.WARNING
    else:
        console_level = logging.INFO
    console_handler.setLevel(console_level)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)


def main() -> int:
    args = parse_args()

    # Live progress bar only on interactive terminals, and only when the
    # user hasn't asked for the noisier alternatives.
    show_progress = (
        sys.stderr.isatty()
        and not args.quiet
        and not args.verbose
        and not args.debug
    )
    configure_logging(args.debug, args.verbose, show_progress)

    if not args.api_key:
        logger.error("Brave API key missing (use --api-key or BRAVE_API_KEY env var)")
        return 2

    targets = _expand_targets(args.directories, args.recursive)
    if not targets:
        logger.error("No usable directories among: %s", args.directories)
        return 2

    if args.clean_cache:
        try:
            Path(CACHE_FILE).unlink(missing_ok=True)
            logger.info("Cache cleared")
        except OSError as e:
            logger.error("Cannot clear cache: %s", e)

    cache = CompressedCache(CACHE_FILE, CACHE_TTL_DAYS)
    brave = BraveSearchClient(args.api_key, cache)
    progress = ProgressReporter(enabled=show_progress)
    renamer = VideoFileRenamer(brave, cache, progress=progress)
    try:
        for target in targets:
            logger.info("Target directory: %s", target)
            if show_progress:
                sys.stderr.write(f"\n=== {target} ===\n")
                sys.stderr.flush()
            try:
                renamer.process(target, force=args.force)
            except (ValueError, OSError) as e:
                logger.error("Skipping %s: %s", target, e)
    except KeyboardInterrupt:
        progress.finish()
        logger.info("Interrupted by user")
        cache.save()
        return 130

    print("\nRenaming completed:")
    for k, v in asdict(renamer.stats).items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
