import sys
import os
import shutil
import requests
import argparse
import time
import re
import pickle
import hashlib
import logging
import json
import zlib
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from unidecode import unidecode
from rapidfuzz import fuzz
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

# Конфигурация логирования
LOG_FILE = 'file_renamer.log'
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 10MB для каждого лог файла, храним 7 файлов
handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=7)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# API конфигурация
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
API_KEY = None
RENAMED_FILES_LOG = "renamed_files.txt"
CACHE_FILE = "search_cache.pkl"
CACHE_TTL = 30  # дней для хранения кэша

# Поддерживаемые видео форматы
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpeg", ".mpg"}

# Расширенный список видео хостингов
VIDEO_HOSTINGS = [
    # Общие видео хостинги
    "YouTube", "Vimeo", "Dailymotion", "Twitch", "Facebook", "Instagram", "Twitter",
    # Взрослые видео хостинги (расширенный список)
    "Pornhub", "Xvideos", "YouPorn", "RedTube", "Porn.com", "XHamster", "Brazzers",
    "SpankBang", "TNAFlix", "Tube8", "JizzBunker", "KeezMovies", "Nuvid", "DrTuber",
    "BangBros", "Mofos", "Reality Kings", "PornHD", "ManyVids", "PornTrex", "EPORNER",
    "xHamsterLive", "Chaturbate", "CamSoda", "MyFreeCams", "LiveJasmin",
    "VRBangers", "WankzVR", "AdultTime", "PornDoe", "Beeg", "SunPorno",
    "Porn300", "PornOne", "MegaPorn", "EMPFlix", "Txxx", "HDZog", "AlphaPorno",
    "OnlyFans", "Manyvids", "ModelHub", "XHamster Premium", "PornhubPremium",
]

# Расширенный список исключаемых сайтов
EXCLUDED_SITES = [
    "Wikipedia", "IMDb", "news", "kino", "film", "serial", "afisha",
    "kinopoisk", "ivi.ru", "megogo", "okko", "more.tv", "tvzavr",
    "reddit.com", "facebook.com", "twitter.com", "instagram.com",
    "tiktok.com", "pinterest.com", "linkedin.com", "tumblr.com",
]

# Паттерны для очистки (расширенные)
COMMON_PATTERNS = {
    'resolution': r'\b(360p|480p|720p|1080p|2160p|4K|8K|HD|FHD|UHD)\b',
    'video_quality': r'\b(HD|HQ|HDRip|BRRip|DVDRip|WEBRip|BluRay)\b',
    'codec': r'\b(x264|h264|x265|h265|hevc|avc|mp3|aac|ac3|dts|flac)\b',
    'format': r'\b(MP4|MKV|AVI|WMV|FLV|MPEG|MPG)\b',
    'extra': r'\b(Official|Video|Full|Complete|Scene|Version|Edit|Cut)\b',
    'brackets': r'[\[\(\{].*?[\]\)\}]',
    'date': r'\b\d{2,4}[-/.]\d{2}[-/.]\d{2,4}\b',
    'website_tags': r'\b(com|net|org|xxx)\b',
    'resolution_numbers': r'\b\d{3,4}x\d{3,4}\b',
    'file_size': r'\b\d+(\.\d+)?\s*(MB|GB|TB)\b',
}
class CompressedCache:
    """Кэш с поддержкой сжатия и TTL"""
    def __init__(self, filename: str, ttl_days: int = 30):
        self.filename = filename
        self.ttl_days = ttl_days
        self.cache = {}
        self.timestamps = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'rb') as f:
                    compressed_data = f.read()
                    if len(compressed_data) > 100 * 1024 * 1024:  # 100MB limit
                        logger.warning("Cache file too large, creating new cache")
                        return
                    try:
                        decompressed_data = zlib.decompress(compressed_data)
                        data = pickle.loads(decompressed_data)
                        if not isinstance(data, dict):
                            raise ValueError("Invalid cache format")
                        self.cache = data.get('cache', {})
                        self.timestamps = data.get('timestamps', {})
                        self._cleanup_expired()
                    except (zlib.error, pickle.UnpicklingError) as e:
                        logger.error(f"Corrupted cache file: {e}")
                        self.cache = {}
                        self.timestamps = {}
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self.cache = {}
                self.timestamps = {}

    def save(self) -> None:
        try:
            data = {
                'cache': self.cache,
                'timestamps': self.timestamps
            }
            compressed_data = zlib.compress(pickle.dumps(data))
            with open(self.filename, 'wb') as f:
                f.write(compressed_data)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _cleanup_expired(self) -> None:
        current_time = datetime.now()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if (current_time - timestamp).days > self.ttl_days
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            timestamp = self.timestamps.get(key)
            if timestamp and (datetime.now() - timestamp).days <= self.ttl_days:
                return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = value
        self.timestamps[key] = datetime.now()

class BraveSearchClient:
    """Клиент для работы с Brave Search API"""
    def __init__(self, api_key: str, cache: CompressedCache):
        self.api_key = api_key
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "X-Subscription-Token": api_key
        })
        self.retry_count = 3
        self.retry_delay = 5

    def _generate_cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def search(self, query: str) -> List[Dict]:
        initial_delay = self.retry_delay
        cache_key = self._generate_cache_key(query)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {query}")
            return cached_result

        # Пробуем разные варианты поиска
        results = self._do_search(f'"{query}" adult video')  # Сначала точный поиск
        
        if not results:
            results = self._do_search(f'{query} adult video')  # Затем без кавычек
            
        if not results:
            words = query.split()
            if len(words) > 2:
                simplified_query = ' '.join(words[:3])  # Берем только первые 3 слова
                results = self._do_search(f'{simplified_query} adult video')

        return results

    def _do_search(self, query: str) -> List[Dict]:
        for attempt in range(self.retry_count):
            try:
                response = self.session.get(
                    BRAVE_SEARCH_API_URL,
                    params={
                        "q": query,
                        "count": 50,  # Увеличили количество результатов
                        "safesearch": "off",
                    },
                    timeout=10
                )

                if response.status_code == 429:  # Too Many Requests
                    logger.warning(f"Rate limit hit, waiting {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2
                    continue
                elif response.status_code in {403, 401}:
                    logger.error("API key invalid or expired")
                    return []
                elif response.status_code >= 500:
                    logger.error(f"Server error: {response.status_code}")
                    time.sleep(self.retry_delay)
                    continue

                response.raise_for_status()
                results = response.json().get("web", {}).get("results", [])
                
                filtered_results = []
                for result in results:
                    url_lower = result.get("url", "").lower()
                    if not any(site.lower() in url_lower for site in EXCLUDED_SITES):
                        filtered_results.append({
                            "title": result.get("title", ""),
                            "page_age": result.get("page_age"),
                            "url": url_lower
                        })

                self.cache.set(self._generate_cache_key(query), filtered_results)
                self.retry_delay = initial_delay  # Сбрасываем задержку
                return filtered_results

            except requests.exceptions.RequestException as e:
                logger.error(f"API request error (attempt {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2
                else:
                    logger.error(f"Failed to get results for query: {query}")
                    self.retry_delay = initial_delay
                    return []

        return []
class TitleProcessor:
    """Класс для обработки и очистки названий файлов"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            stop_words='english'
        )

    def clean_title(self, title: str, is_original_file: bool = False) -> str:
        """
        Улучшенная очистка названия
        is_original_file: флаг, указывающий, что это оригинальное имя файла
        """
        # Базовая очистка
        title = title.split('|')[0].strip()
        
        # Удаляем имена видео хостингов
        for host in VIDEO_HOSTINGS:
            title = re.sub(rf'\b{re.escape(host)}\b', '', title, flags=re.IGNORECASE)

        # Применяем все паттерны очистки
        for pattern in COMMON_PATTERNS.values():
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)

        # Специальная обработка дат
        date_match = re.search(r'\b\d{4}\b', title)
        year = date_match.group() if date_match else None

        # Очищаем от специальных символов, но сохраняем пробелы между словами
        title = re.sub(r'[^\w\s-]', ' ', title)
        
        # Удаляем множественные пробелы
        title = re.sub(r'\s+', ' ', title)

        # Нормализация Unicode и конвертация в ASCII
        title = unicodedata.normalize('NFKD', title)
        title = unidecode(title)
        title = title.encode('ascii', errors='ignore').decode()

        # Дополнительная очистка для оригинальных имен файлов
        if is_original_file:
            # Удаляем слишком длинные числовые последовательности
            title = re.sub(r'\b\d{6,}\b', '', title)
            # Удаляем одиночные буквы
            title = re.sub(r'\b[a-zA-Z]\b', '', title)
            # Удаляем специфичные разделители
            title = re.sub(r'[_\-]+', ' ', title)

        # Возвращаем год в название если он был
        if year and year not in title:
            title = f"{title} {year}"

        # Финальная очистка от множественных пробелов
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Если после очистки название стало пустым или слишком коротким
        if len(title) < 3:
            if is_original_file:
                return "unnamed_video"  # Для оригинальных файлов даем generic название
            return title  # Для найденных названий возвращаем как есть

        return title

    def extract_meaningful_parts(self, title: str) -> List[str]:
        """Извлекает значимые части из названия"""
        # Разбиваем на части по разделителям
        parts = re.split(r'[-_\s]+', title)
        
        # Фильтруем короткие части и числа
        return [part for part in parts if len(part) > 2 and not part.isdigit()]

    def calculate_similarity(self, title1: str, title2: str) -> float:
        """Рассчитывает схожесть названий с учетом разных факторов"""
        # Очищаем названия
        clean_title1 = self.clean_title(title1)
        clean_title2 = self.clean_title(title2)

        if not clean_title1 or not clean_title2:
            return 0.0

        # Получаем значимые части
        parts1 = self.extract_meaningful_parts(clean_title1)
        parts2 = self.extract_meaningful_parts(clean_title2)

        if not parts1 or not parts2:
            return 0.0

        # Рассчитываем различные метрики схожести
        
        # TF-IDF similarity
        try:
            tfidf_matrix = self.vectorizer.fit_transform([clean_title1, clean_title2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_sim = 0.0

        # Fuzzy ratio для полных строк
        fuzzy_sim = fuzz.ratio(clean_title1, clean_title2) / 100.0

        # Token sort ratio для учета перестановок слов
        token_sort_sim = fuzz.token_sort_ratio(clean_title1, clean_title2) / 100.0

        # Учитываем длину названий (пенализируем сильно различающиеся длины)
        len_ratio = min(len(clean_title1), len(clean_title2)) / max(len(clean_title1), len(clean_title2))

        # Проверяем наличие общих значимых частей
        common_parts = set(parts1) & set(parts2)
        parts_sim = len(common_parts) / max(len(parts1), len(parts2))

        # Взвешенная комбинация всех метрик
        weights = {
            'cosine': 0.3,
            'fuzzy': 0.2,
            'token_sort': 0.2,
            'length': 0.1,
            'parts': 0.2
        }

        final_similarity = (
            weights['cosine'] * cosine_sim +
            weights['fuzzy'] * fuzzy_sim +
            weights['token_sort'] * token_sort_sim +
            weights['length'] * len_ratio +
            weights['parts'] * parts_sim
        )

        return final_similarity

    def choose_best_title(self, original_title: str, search_results: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """Выбирает лучшее название из результатов поиска"""
        if not search_results:
            return None, None

        best_score = 0
        best_title = None
        best_date = None

        for result in search_results:
            title = result.get('title', '')
            page_age = result.get('page_age')
            
            # Рассчитываем схожесть
            similarity = self.calculate_similarity(original_title, title)

            # Учитываем дополнительные факторы
            # Пенализируем слишком короткие или длинные названия
            length_penalty = 1.0
            if len(title) < 10 or len(title) > 100:
                length_penalty = 0.8

            # Получаем финальный скор
            final_score = similarity * length_penalty

            if final_score > best_score:
                best_score = final_score
                best_title = self.clean_title(title)
                best_date = page_age[:10] if page_age else None

        # Снизили порог схожести
        if best_score < 0.15:
            return None, None

        return best_date, best_title
class VideoFileRenamer:
    """Класс для переименования видео файлов"""
    def __init__(self, api_key: str):
        self.cache = CompressedCache(CACHE_FILE, CACHE_TTL)
        self.search_client = BraveSearchClient(api_key, self.cache)
        self.title_processor = TitleProcessor()
        self.renamed_files = self._load_renamed_files()
        self.stats = {
            'processed': 0,
            'renamed': 0,
            'skipped': 0,
            'errors': 0,
            'no_search_results': 0,
            'using_original_name': 0
        }

    def _load_renamed_files(self) -> Set[str]:
        """Загружает список переименованных файлов"""
        try:
            if os.path.exists(RENAMED_FILES_LOG):
                with open(RENAMED_FILES_LOG, "r", encoding='utf-8') as f:
                    return set(line.strip() for line in f)
        except Exception as e:
            logger.error(f"Error loading renamed files log: {e}")
        return set()

    def _save_renamed_file(self, filename: str) -> None:
        """Сохраняет информацию о переименованном файле"""
        try:
            with open(RENAMED_FILES_LOG, "a", encoding='utf-8') as f:
                f.write(f"{filename}\n")
            self.renamed_files.add(filename)
        except Exception as e:
            logger.error(f"Error saving renamed file log: {e}")

    def _get_safe_filename(self, filename: str) -> str:
        """Создает безопасное имя файла"""
        # Заменяем недопустимые символы
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Ограничиваем длину имени файла (учитываем файловую систему)
        max_length = 255
        if len(safe_name) > max_length:
            base, ext = os.path.splitext(safe_name)
            safe_name = base[:max_length-len(ext)] + ext
            
        return safe_name

    def _get_unique_filename(self, directory: str, desired_name: str) -> str:
        """Получает уникальное имя файла в случае конфликта"""
        base, ext = os.path.splitext(desired_name)
        counter = 1
        result_name = desired_name
        
        while os.path.exists(os.path.join(directory, result_name)):
            result_name = f"{base}_{counter}{ext}"
            counter += 1
            
        return result_name

    def _should_process_file(self, filename: str) -> bool:
        """Проверяет, нужно ли обрабатывать файл"""
        # Получаем расширение файла
        ext = os.path.splitext(filename)[1].lower()
        
        # Проверяем условия и логируем причину пропуска
        if ext not in VIDEO_EXTENSIONS:
            logger.info(f'File "{filename}" skipped: not a video file (extension: {ext})')
            return False
            
        if filename in self.renamed_files:
            logger.info(f'File "{filename}" skipped: already renamed')
            self.stats['skipped'] += 1
            return False
            
        if filename.startswith('.'):
            logger.info(f'File "{filename}" skipped: hidden file')
            self.stats['skipped'] += 1
            return False

        if len(os.path.splitext(filename)[0]) == 18 and os.path.splitext(filename)[0].isdigit():
            logger.info(f'File "{filename}" skipped: numeric sequence')
            self.stats['skipped'] += 1
            return False
            
        return True

    def _backup_original_filename(self, directory: str, original_name: str, new_name: str) -> None:
        """Создает бэкап оригинального имени файла"""
        backup_file = os.path.join(directory, ".filename_mapping.json")
        mapping = {}
        
        if os.path.exists(backup_file):
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
            except Exception as e:
                logger.error(f"Error loading backup mapping: {e}")

        mapping[new_name] = original_name
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving backup mapping: {e}")

    def rename_files_in_directory(self, directory: str) -> None:
        """Переименовывает видео файлы в указанной директории"""
        try:
            # Проверяем существование директории
            if not os.path.exists(directory):
                raise ValueError(f"Directory does not exist: {directory}")

            # Проверяем свободное место
            free_space = shutil.disk_usage(directory).free
            if free_space < 1024 * 1024 * 100:  # 100MB minimum
                raise OSError("Not enough free disk space")

            logger.info(f"Starting video file renaming in: {directory}")
            start_time = time.time()

            for filename in os.listdir(directory):
                self.stats['processed'] += 1
                
                try:
                    if not self._should_process_file(filename):
                        continue

                    file_path = os.path.join(directory, filename)
                    base_name = os.path.splitext(filename)[0]

                    # Проверяем, не заблокирован ли файл
                    try:
                        with open(file_path, 'rb') as f:
                            pass
                    except PermissionError:
                        logger.error(f"File {filename} is locked by another process")
                        self.stats['errors'] += 1
                        continue

                    # Получаем дату модификации файла
                    file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d')

                    # Поиск подходящего названия
                    logger.info(f"Processing file: {filename}")
                    search_results = self.search_client.search(base_name)

                    if search_results:
                        # Если есть результаты поиска, пытаемся найти лучшее название
                        date, new_name = self.title_processor.choose_best_title(base_name, search_results)
                        if new_name:
                            # Используем найденное название
                            date = date or file_date
                        else:
                            # Если название не подошло, используем очищенное оригинальное с датой
                            new_name = self.title_processor.clean_title(base_name, is_original_file=True)
                            date = file_date
                            self.stats['using_original_name'] += 1
                    else:
                        # Если поиск не дал результатов, используем очищенное оригинальное название с датой
                        new_name = self.title_processor.clean_title(base_name, is_original_file=True)
                        date = file_date
                        self.stats['no_search_results'] += 1

                    # Формируем новое имя файла
                    new_filename = f"{date} {new_name}{os.path.splitext(filename)[1]}"
                    safe_filename = self._get_safe_filename(new_filename)
                    unique_filename = self._get_unique_filename(directory, safe_filename)
                    new_path = os.path.join(directory, unique_filename)

                    # Создаем бэкап и переименовываем
                    self._backup_original_filename(directory, filename, unique_filename)
                    os.rename(file_path, new_path)
                    self._save_renamed_file(unique_filename)
                    
                    logger.info(f'Renamed "{filename}" to "{unique_filename}"')
                    self.stats['renamed'] += 1

                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    self.stats['errors'] += 1

            # Сохраняем кэш
            self.cache.save()

            # Логируем статистику
            end_time = time.time()
            duration = end_time - start_time
            logger.info(
                f"Renaming completed. Duration: {duration:.2f}s. "
                f"Stats: {json.dumps(self.stats, indent=2)}"
            )

        except Exception as e:
            logger.error(f"Fatal error during directory processing: {e}")
            raise
def setup_environment() -> bool:
    """Проверяет и настраивает окружение"""
    try:
        # Проверяем наличие необходимых директорий для логов и кэша
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Проверяем права на запись
        if os.path.exists(LOG_FILE):
            if not os.access(LOG_FILE, os.W_OK):
                raise PermissionError(f"No write access to log file: {LOG_FILE}")
        
        # Проверяем доступ к API
        if not API_KEY:
            raise ValueError("API key is not set")

        # Проверяем доступность временной директории для кэша
        cache_dir = os.path.dirname(CACHE_FILE)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        if os.path.exists(CACHE_FILE) and not os.access(CACHE_FILE, os.W_OK):
            raise PermissionError(f"No write access to cache file: {CACHE_FILE}")

        return True

    except Exception as e:
        print(f"Error setting up environment: {e}")
        return False

def parse_arguments() -> argparse.Namespace:
    """Парсит аргументы командной строки"""
    parser = argparse.ArgumentParser(
        description="Script for renaming adult video files based on online search results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory with video files"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Brave Search API key (can also be set via BRAVE_API_KEY environment variable)",
        required=False
    )

    parser.add_argument(
        '--clean-cache',
        action='store_true',
        help='Clean the search cache before processing'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force processing of already renamed files'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()

def clean_cache() -> None:
    """Очищает кэш поиска"""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            logger.info("Search cache cleaned successfully")
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")

def main() -> int:
    """Основная функция программы"""
    try:
        # Парсим аргументы
        args = parse_arguments()

        # Устанавливаем уровень логирования
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        # Устанавливаем API ключ
        global API_KEY
        API_KEY = args.api_key or os.getenv('BRAVE_API_KEY')

        if not API_KEY:
            logger.error('Brave Search API key not provided. Please specify it via --api-key argument or BRAVE_API_KEY environment variable.')
            return 1

        # Проверяем окружение
        if not setup_environment():
            return 1

        # Очищаем кэш если требуется
        if args.clean_cache:
            clean_cache()

        # Проверяем существование директории
        if not os.path.exists(args.directory):
            logger.error(f"Directory does not exist: {args.directory}")
            return 1

        if not os.path.isdir(args.directory):
            logger.error(f"Path is not a directory: {args.directory}")
            return 1

        # Проверяем права на запись в директорию
        if not os.access(args.directory, os.W_OK):
            logger.error(f"No write access to directory: {args.directory}")
            return 1

        try:
            # Проверяем свободное место
            free_space = shutil.disk_usage(args.directory).free
            if free_space < 1024 * 1024 * 100:  # 100MB minimum
                logger.error("Not enough free disk space")
                return 1
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return 1

        # Создаем и запускаем переименователь файлов
        renamer = VideoFileRenamer(API_KEY)
        
        # Логируем начало работы
        logger.info(f"Starting video file renaming in directory: {args.directory}")
        logger.info(f"Cache TTL set to {CACHE_TTL} days")
        
        # Запускаем переименование
        renamer.rename_files_in_directory(args.directory)

        # Выводим итоговую статистику в консоль
        print("\nRenaming completed:")
        print(f"Processed files: {renamer.stats['processed']}")
        print(f"Successfully renamed: {renamer.stats['renamed']}")
        print(f"Skipped: {renamer.stats['skipped']}")
        print(f"Files with no search results: {renamer.stats['no_search_results']}")
        print(f"Files using original name: {renamer.stats['using_original_name']}")
        print(f"Errors: {renamer.stats['errors']}")

        logger.info("File renaming completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
