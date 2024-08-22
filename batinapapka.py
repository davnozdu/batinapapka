import os
import requests
import argparse
from difflib import SequenceMatcher
import time
import re
import pickle
import hashlib
import logging
from datetime import datetime

# Logging configuration
LOG_FILE = 'file_renamer.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# API Configuration
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"
API_KEY = "YOUR_API_KEY"
RENAMED_FILES_LOG = "renamed_files.txt"
CACHE_FILE = "search_cache.pkl"

# List of supported video formats
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

# List of popular video hosting sites to remove from titles
VIDEO_HOSTINGS = [
    "YouTube", "Vimeo", "Dailymotion", "Twitch", "Facebook", "Instagram", "Twitter",
    "TikTok", "Metacafe", "Vevo", "Hulu", "Netflix", "Pornhub", "Xvideos", "YouPorn",
    "RedTube", "Porn.com", "XHamster", "Brazzers", "Naughty America", "SpankBang", 
    "TNAFlix", "YouJizz", "Tube8", "JizzBunker", "KeezMovies", "Nuvid", "DrTuber",
    "Yuvutu", "Xtube", "BangBros", "Mofos", "Reality Kings", "BadoinkVR", "PornHD", 
    "ManyVids"
]

EXCLUDED_SITES = ["Wikipedia", "IMDb", "Rotten Tomatoes", "Metacritic", "AllMusic", "Fandom", "news"]

def is_numeric_sequence(filename):
    base_name = os.path.splitext(filename)[0]
    return len(base_name) == 18 and base_name.isdigit()

def load_cache():
    """Loads the cache from a file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
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
        logging.info(f'Using cached result for query: {query}')
        return cache[cache_key]

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": API_KEY
    }
    params = {
        "q": query,
        "count": 20,
        "safesearch": "off",
    }
    response = requests.get(BRAVE_SEARCH_API_URL, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json().get("web", {}).get("results", [])
        filtered_results = [
            (result["title"], result.get("page_age"), result.get("url")) for result in results
            if not any(excluded_site.lower() in result["url"].lower() for excluded_site in EXCLUDED_SITES)
        ]
        cache[cache_key] = filtered_results
        save_cache(cache)
        return filtered_results
    return []

def clean_title(title):
    """Cleans the title by removing special characters and video hosting names."""
    title = title.split("|")[0].strip()
    for host in VIDEO_HOSTINGS:
        title = re.sub(r'\b{}\b'.format(re.escape(host)), '', title, flags=re.IGNORECASE)
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def choose_best_title(base_name, titles_with_dates):
    """Selects the most appropriate title from the list."""
    best_match = None
    highest_similarity = 0
    best_date = None

    for result in titles_with_dates:
        title, page_age, url = result
        clean_title_str = clean_title(title)
        similarity = similar(base_name.lower(), clean_title_str.lower())
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = clean_title_str
            if page_age:
                best_date = page_age[:10]  # Extract only YYYY-MM-DD

    return best_date, best_match

def similar(a, b):
    """Calculates the similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()

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

def trim_log_file():
    """Trims the log file to the last 100 lines."""
    try:
        with open(LOG_FILE, 'r') as file:
            lines = file.readlines()
        if len(lines) > 100:
            with open(LOG_FILE, 'w') as file:
                file.writelines(lines[-100:])
    except Exception as e:
        logging.error(f'Error trimming log file: {e}')

def has_date_prefix(filename):
    """Checks if the file name starts with a date in the format YYYY-MM-DD."""
    return re.match(r'^\d{4}-\d{2}-\d{2}', filename) is not None

def rename_video_files_in_directory(directory):
    """Renames video files in the directory based on search results."""
    renamed_files = load_renamed_files_log()
    cache = load_cache()
    
    for filename in os.listdir(directory):
        try:
            file_path = os.path.join(directory, filename)
            file_extension = os.path.splitext(filename)[1].lower()

            if os.path.isfile(file_path) and file_extension in VIDEO_EXTENSIONS:
                if filename in renamed_files:
                    logging.info(f'File "{filename}" has already been renamed, skipping.')
                    continue

                if is_numeric_sequence(filename):
                    logging.info(f'File "{filename}" is a numeric sequence, skipping.')
                    continue

                base_name = os.path.splitext(filename)[0]

                if has_date_prefix(base_name):
                    logging.info(f'File "{filename}" already contains a date, skipping.')
                    continue

                titles_with_dates = search_brave(base_name, cache)
                best_date, new_name = choose_best_title(base_name, titles_with_dates)

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
                    logging.info(f'File "{filename}" was renamed to "{new_file_name}"')
                    save_renamed_file_log(new_file_name)
                else:
                    logging.warning(f'Could not find a suitable title for file "{filename}".')
            
            time.sleep(1)
            trim_log_file()  # Trim the log file after each cycle
        except Exception as e:
            logging.error(f'Error processing file "{filename}": {e}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for renaming video files based on search results.")
    parser.add_argument("directory", type=str, help="Path to the directory with video files")
    
    args = parser.parse_args()
    
    rename_video_files_in_directory(args.directory)
