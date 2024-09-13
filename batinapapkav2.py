import os
import requests
import argparse
import time
import re
import pickle
import hashlib
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from unidecode import unidecode
from rapidfuzz import fuzz
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

# Logging configuration
LOG_FILE = 'file_renamer.log'
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)  # 5MB per file
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# API configuration
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
API_KEY = None  # Will be set from command line arguments or environment variables
RENAMED_FILES_LOG = "renamed_files.txt"
CACHE_FILE = "search_cache.pkl"

# Supported video formats
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpeg", ".mpg"]

# List of popular video hosting sites to remove from titles
VIDEO_HOSTINGS = [
    # General video hosting sites
    "YouTube", "Vimeo", "Dailymotion", "Twitch", "Facebook", "Instagram", "Twitter",
    "TikTok", "Metacafe", "Vevo", "Hulu", "Netflix",
    # Adult video hosting sites
    "Pornhub", "Xvideos", "YouPorn", "RedTube", "Porn.com", "XHamster", "Brazzers",
    "Naughty America", "SpankBang", "TNAFlix", "YouJizz", "Tube8", "JizzBunker",
    "KeezMovies", "Nuvid", "DrTuber", "Yuvutu", "Xtube", "BangBros", "Mofos",
    "Reality Kings", "BadoinkVR", "PornHD", "ManyVids", "PornTrex", "EPORNER",
    "xHamsterLive", "Chaturbate", "CamSoda", "MyFreeCams", "LiveJasmin",
    "Streamate", "Cam4", "Camster", "Camversity", "Flirt4Free", "Cams.com",
    "Stripchat", "BongaCams", "LivePrivates", "ImLive", "CamContacts", "FireCams",
    "Cherry.tv", "FapHouse", "VRBangers", "WankzVR", "NaughtyAmericaVR",
    "SexLikeReal", "AdultTime", "PornDoe", "SpankWire", "Beeg", "SunPorno",
    "Porn300", "PornOne", "MegaPorn", "EMPFlix", "Txxx", "HDZog", "AlphaPorno",
    "Xbabe", "FapVid", "Vid123", "PornDig", "xVideosX", "PornFlip", "Cliplips",
    "Vid2C", "Xnxx", "HClips", "X18", "YesPornPlease", "ExtremeTube", "NuPorn",
    "BravoTube", "PornRabbit", "PornHeed", "Lubed", "fapster",
]

EXCLUDED_SITES = [
    "Wikipedia", "IMDb", "Rotten Tomatoes", "Metacritic", "AllMusic", "Fandom",
    "news", "kino", "film", "serial", "afisha", "rambler", "kinopoisk", "ivi.ru",
    "megogo", "okko", "more.tv", "tvzavr", "premier.one"
]

# Precompile regular expressions
VIDEO_HOSTINGS_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, VIDEO_HOSTINGS)) + r')\b', re.IGNORECASE)
BRACKETED_CONTENT_PATTERN = re.compile(r'[\[\(\{].*?[\]\)\}]')
RESOLUTION_PATTERN = re.compile(r'\b(360p|480p|720p|1080p|2160p|4K|8K)\b', re.IGNORECASE)
CODEC_PATTERN = re.compile(r'\b(x264|h264|x265|h265|hevc|avc|mp3|aac|ac3|dts|flac)\b', re.IGNORECASE)
EXTRA_PHRASES_PATTERN = re.compile(r'\b(Official Video|Music Video|Lyrics Video|Lyric Video|Full Movie|HD|HQ)\b', re.IGNORECASE)
NON_ALPHANUMERIC_PATTERN = re.compile(r'[^a-zA-Z0-9\s]')
MULTIPLE_SPACES_PATTERN = re.compile(r'\s+')
DATE_PREFIX_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}')

EXCLUDED_SITES_LOWER = [site.lower() for site in EXCLUDED_SITES]

def is_numeric_sequence(filename):
    base_name = os.path.splitext(filename)[0]
    return len(base_name) == 18 and base_name.isdigit()

def load_cache():
    """Loads the cache from a file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f'Error loading cache: {e}')
    return {}

def save_cache(cache):
    """Saves the cache to a file."""
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

def generate_cache_key(query):
    """Generates a unique key for the cache based on the query."""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def search_brave(query, cache):
    """Search function using the Brave Search API with caching."""
    cache_key = generate_cache_key(query)
    if cache_key in cache:
        logger.info(f'Using cached result for query: {query}')
        return cache[cache_key], False  # False means API call was not made

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": API_KEY
    }
    params = {
        "q": f'"{query}" video',  # Add quotes and "video" keyword for specificity
        "count": 20,
        "safesearch": "off",
    }
    try:
        response = requests.get(BRAVE_SEARCH_API_URL, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get("web", {}).get("results", [])
            filtered_results = []
            for result in results:
                url_lower = result.get("url", "").lower()
                if not any(excluded_site in url_lower for excluded_site in EXCLUDED_SITES_LOWER):
                    filtered_results.append(
                        (result.get("title", ""), result.get("page_age"), result.get("url"))
                    )
            cache[cache_key] = filtered_results
            return filtered_results, True  # True means API call was made
        else:
            logger.error(f'Error requesting Brave Search API with status code {response.status_code}')
            return [], False
    except requests.exceptions.RequestException as e:
        logger.error(f'Network error when requesting API: {e}')
        return [], False

def clean_title(title):
    """Cleans the title by removing special characters and video hosting names."""
    title = title.split("|")[0].strip()
    title = VIDEO_HOSTINGS_PATTERN.sub('', title)
    title = BRACKETED_CONTENT_PATTERN.sub('', title)
    title = RESOLUTION_PATTERN.sub('', title)
    title = CODEC_PATTERN.sub('', title)
    title = EXTRA_PHRASES_PATTERN.sub('', title)
    title = NON_ALPHANUMERIC_PATTERN.sub('', title)
    title = MULTIPLE_SPACES_PATTERN.sub(' ', title).strip()
    title = unicodedata.normalize('NFKD', title)
    title = unidecode(title)  # Convert Unicode to ASCII
    # Remove any non-ASCII characters
    title = title.encode('ascii', errors='ignore').decode()
    return title.strip()

def extract_title_from_metadata(file_path):
    """Attempts to extract the title from video file metadata."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format_tags=title', '-of', 'default=nw=1:nk=1', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        metadata_title = result.stdout.strip()
        if metadata_title:
            logger.info(f'Title extracted from metadata: {metadata_title}')
            return metadata_title
    except Exception as e:
        logger.error(f'Error extracting metadata from file "{file_path}": {e}')
    return None

def choose_best_title(base_name, titles_with_dates, file_path):
    """Selects the most appropriate title from the list."""
    best_match = None
    highest_similarity = 0
    best_date = None

    base_name_clean = clean_title(base_name)

    # Try to extract title from metadata
    metadata_title = extract_title_from_metadata(file_path)
    if metadata_title:
        metadata_title_clean = clean_title(metadata_title)
        similarity = fuzz.token_set_ratio(base_name_clean, metadata_title_clean)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = metadata_title_clean

    # List for comparison
    candidate_titles = []

    for result in titles_with_dates:
        title, page_age, url = result
        clean_title_str = clean_title(title)
        if not clean_title_str:
            continue  # Skip if title is empty after cleaning
        candidate_titles.append((clean_title_str, page_age))

    # Add base_name_clean to candidate list
    candidate_titles.append((base_name_clean, None))

    # Use TF-IDF vectorization and cosine similarity
    documents = [base_name_clean] + [title for title, _ in candidate_titles]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:])[0]

    for i, (similarity_score, (title, page_age)) in enumerate(zip(cosine_similarities, candidate_titles)):
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            best_match = title
            best_date = page_age[:10] if page_age else None

    return best_date, best_match

def get_file_modification_date(file_path):
    """Returns the file modification date in YYYY-MM-DD format."""
    modification_time = os.path.getmtime(file_path)
    return datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d')

def load_renamed_files_log():
    """Loads the list of already renamed files from the log."""
    if os.path.exists(RENAMED_FILES_LOG):
        with open(RENAMED_FILES_LOG, "r") as file:
            return set(line.strip() for line in file)
    return set()

def save_renamed_file_log(filename):
    """Saves the renamed file to the log."""
    with open(RENAMED_FILES_LOG, "a") as file:
        file.write(filename + "\n")

def has_date_prefix(filename):
    """Checks if the file name starts with a date in the format YYYY-MM-DD."""
    return DATE_PREFIX_PATTERN.match(filename) is not None

def rename_video_files_in_directory(directory):
    """Renames video files in the directory based on search results."""
    renamed_files = load_renamed_files_log()
    cache = load_cache()
    cache_modified = False

    for filename in os.listdir(directory):
        try:
            file_path = os.path.join(directory, filename)
            file_extension = os.path.splitext(filename)[1].lower()

            if os.path.isfile(file_path) and file_extension in VIDEO_EXTENSIONS:
                if filename in renamed_files:
                    logger.info(f'File "{filename}" has already been renamed, skipping.')
                    continue

                if is_numeric_sequence(filename):
                    logger.info(f'File "{filename}" is a numeric sequence, skipping.')
                    continue

                base_name = os.path.splitext(filename)[0]

                if has_date_prefix(base_name):
                    logger.info(f'File "{filename}" already contains a date, skipping.')
                    continue

                titles_with_dates, api_called = search_brave(base_name, cache)
                if api_called:
                    cache_modified = True
                    time.sleep(1)  # Delay to comply with API rate limits

                best_date, new_name = choose_best_title(base_name, titles_with_dates, file_path)

                # Use the date from the internet if found, otherwise use the file modification date
                if not best_date:
                    best_date = get_file_modification_date(file_path)

                if new_name:
                    new_file_name = f"{best_date} {new_name}{file_extension}"

                    new_file_path = os.path.join(directory, new_file_name)

                    # Check if a file with the same name already exists
                    if os.path.exists(new_file_path):
                        base, ext = os.path.splitext(new_file_name)
                        counter = 1
                        while os.path.exists(new_file_path):
                            new_file_name = f"{base}_{counter}{ext}"
                            new_file_path = os.path.join(directory, new_file_name)
                            counter += 1

                    os.rename(file_path, new_file_path)
                    logger.info(f'File "{filename}" was renamed to "{new_file_name}"')
                    save_renamed_file_log(new_file_name)
                else:
                    logger.warning(f'Could not find a suitable title for file "{filename}".')
        except Exception as e:
            logger.error(f'Error processing file "{filename}": {e}')

    if cache_modified:
        save_cache(cache)
        logger.info('Cache saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for renaming video files based on search results.")
    parser.add_argument("directory", type=str, help="Path to the directory with video files")
    parser.add_argument("--api-key", type=str, help="Brave Search API key", required=False)

    args = parser.parse_args()

    API_KEY = args.api_key or os.getenv('BRAVE_API_KEY')

    if not API_KEY:
        logger.error('Brave Search API key not provided. Please specify it via the --api-key argument or the BRAVE_API_KEY environment variable.')
        exit(1)

    rename_video_files_in_directory(args.directory)
