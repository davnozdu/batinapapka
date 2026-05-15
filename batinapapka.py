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
SIMILARITY_THRESHOLD = 0.40
MIN_FREE_SPACE_BYTES = 100 * 1024 * 1024
REQUEST_TIMEOUT = 10
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_RESULTS_PER_QUERY = 20  # Brave Search API caps `count` at 20.

# Telegram media filenames are 18-digit integers — searching them is pointless.
NUMERIC_FILENAME_LEN = 18

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpeg", ".mpg",
})

VIDEO_HOSTINGS = (
    "YouTube", "Vimeo", "Dailymotion", "Twitch", "Facebook", "Instagram", "Twitter",
    "Pornhub", "Xvideos", "YouPorn", "RedTube", "Porn.com", "XHamster", "Brazzers",
    "SpankBang", "TNAFlix", "Tube8", "JizzBunker", "KeezMovies", "Nuvid", "DrTuber",
    "BangBros", "Mofos", "Reality Kings", "PornHD", "ManyVids", "PornTrex", "EPORNER",
    "xHamsterLive", "Chaturbate", "CamSoda", "MyFreeCams", "LiveJasmin",
    "VRBangers", "WankzVR", "AdultTime", "PornDoe", "Beeg", "SunPorno",
    "Porn300", "PornOne", "MegaPorn", "EMPFlix", "Txxx", "HDZog", "AlphaPorno",
    "OnlyFans", "Manyvids", "ModelHub", "XHamster Premium", "PornhubPremium",
)

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
    )
]

_HOSTINGS_RE = re.compile(
    r"\b(" + "|".join(re.escape(h) for h in VIDEO_HOSTINGS) + r")\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NON_WORD_RE = re.compile(r"[^\w\s-]")
_WS_RE = re.compile(r"\s+")
_LONG_NUM_RE = re.compile(r"\b\d{6,}\b")
_SINGLE_LETTER_RE = re.compile(r"\b[a-zA-Z]\b")
_SEPARATOR_RE = re.compile(r"[_\-]+")
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*]')

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


def _host_excluded(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    for blocked in EXCLUDED_HOSTS:
        if host == blocked or host.endswith("." + blocked):
            return True
    return False


def _normalize_results(items: Iterable[dict]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in items:
        url = item.get("url", "")
        if _host_excluded(url):
            continue
        out.append({
            "title": item.get("title", "") or "",
            "page_age": item.get("page_age") or "",
            "url": url,
        })
    return out


class BraveSearchClient:
    QUERY_TEMPLATE = '"{q}" adult video'

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

    def search(self, query: str) -> List[Dict[str, str]]:
        full_query = self.QUERY_TEMPLATE.format(q=query)
        key = self._key(full_query)
        cached = self.cache.get(key)
        if cached is not None:
            logger.info("Brave cache hit: %s", query)
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
        # token_set_ratio handles word reordering, duplicates and partial overlap
        # well enough that the previous TF-IDF + multi-metric blend isn't needed.
        return fuzz.token_set_ratio(ca, cb) / 100.0

    def choose_best(
        self, original: str, results: List[Dict[str, str]]
    ) -> Tuple[Optional[str], Optional[str]]:
        best_score, best_title, best_date = 0.0, None, None
        for r in results:
            title = r.get("title", "")
            score = self._similarity(original, title)
            if len(title) < 10 or len(title) > 100:
                score *= 0.8
            if score > best_score:
                best_score = score
                best_title = self.clean_title(title)
                best_date = (r.get("page_age") or "")[:10] or None
        if best_score < SIMILARITY_THRESHOLD:
            return None, None
        return best_date, best_title


@dataclass
class Stats:
    processed: int = 0
    renamed: int = 0
    skipped: int = 0
    errors: int = 0
    brave_queries_hit: int = 0
    no_search_results: int = 0
    using_original_name: int = 0


class VideoFileRenamer:
    def __init__(self, brave_client: BraveSearchClient, cache: CompressedCache):
        self.cache = cache
        self.brave_client = brave_client
        self.title = TitleProcessor()
        self.renamed_files = self._load_renamed()
        self.stats = Stats()

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

    def _search(self, query: str) -> List[Dict[str, str]]:
        results = self.brave_client.search(query)
        if results:
            self.stats.brave_queries_hit += 1
            logger.info("Brave returned %d results", len(results))
        seen, unique = set(), []
        for r in results:
            key = (r.get("url", ""), r.get("title", "").lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        return unique

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

        for filename in os.listdir(directory):
            try:
                if not self._should_process(filename, force):
                    continue
                self.stats.processed += 1

                path = os.path.join(directory, filename)
                base = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]

                try:
                    with open(path, "rb"):
                        pass
                except PermissionError:
                    logger.error("File locked: %s", filename)
                    self.stats.errors += 1
                    continue

                file_date = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")

                logger.info("Processing %s", filename)
                results = self._search(base)

                if results:
                    date, new_title = self.title.choose_best(base, results)
                    if not new_title:
                        new_title = self.title.clean_title(base, is_original_file=True)
                        date = file_date
                        self.stats.using_original_name += 1
                    elif not date:
                        date = file_date
                else:
                    new_title = self.title.clean_title(base, is_original_file=True)
                    date = file_date
                    self.stats.no_search_results += 1

                new_filename = self._unique(
                    directory, self._safe(f"{date} {new_title}{ext}")
                )
                new_path = os.path.join(directory, new_filename)
                self._backup_mapping(directory, filename, new_filename)
                shutil.move(path, new_path)
                self._mark_renamed(new_filename)
                logger.info('Renamed "%s" -> "%s"', filename, new_filename)
                self.stats.renamed += 1

            except Exception as e:  # noqa: BLE001 — never let one bad file kill the run
                logger.exception("Error processing %s: %s", filename, e)
                self.stats.errors += 1

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
    parser.add_argument("directory", help="Path to the directory with video files")
    parser.add_argument(
        "--api-key", default=os.getenv("BRAVE_API_KEY"),
        help="Brave Search API key (or BRAVE_API_KEY env var)",
    )
    parser.add_argument("--clean-cache", action="store_true",
                        help="Drop the on-disk cache before processing")
    parser.add_argument("--force", action="store_true",
                        help="Re-process files already listed in renamed_files.txt")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=7)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)


def main() -> int:
    args = parse_args()
    configure_logging(args.debug)

    if not args.api_key:
        logger.error("Brave API key missing (use --api-key or BRAVE_API_KEY env var)")
        return 2
    if not os.path.isdir(args.directory):
        logger.error("Not a directory: %s", args.directory)
        return 2
    if not os.access(args.directory, os.W_OK):
        logger.error("No write access to %s", args.directory)
        return 2
    if shutil.disk_usage(args.directory).free < MIN_FREE_SPACE_BYTES:
        logger.error("Less than %dMB free on %s",
                     MIN_FREE_SPACE_BYTES // (1024 * 1024), args.directory)
        return 2

    if args.clean_cache:
        try:
            Path(CACHE_FILE).unlink(missing_ok=True)
            logger.info("Cache cleared")
        except OSError as e:
            logger.error("Cannot clear cache: %s", e)

    cache = CompressedCache(CACHE_FILE, CACHE_TTL_DAYS)
    brave = BraveSearchClient(args.api_key, cache)
    renamer = VideoFileRenamer(brave, cache)
    try:
        renamer.process(args.directory, force=args.force)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        cache.save()
        return 130

    print("\nRenaming completed:")
    for k, v in asdict(renamer.stats).items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
