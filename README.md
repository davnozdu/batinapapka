
# BatinaPapka

**BatinaPapka** is a Python script designed to automatically rename video files in a specified directory based on search results from the Brave Search API. The script cleans up filenames by removing special characters, video hosting site names, and adds the publication date from search results. If the date cannot be found online, it defaults to the file's modification date.

## Features

- **Automatic Renaming**: Renames video files using titles and dates found via Brave Search API.
- **Date Handling**: Prioritizes online publication dates but uses file modification dates if none are found.
- **Cache Implementation**: Caches API responses to reduce redundant API calls.
- **File Extension Support**: Supports common video formats such as `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, and `.wmv`.
- **Log Management**: Maintains a log file of renamed files and trims the log to the last 100 lines to save space.
- **Exclusion Filters**: Filters out irrelevant results like Wikipedia, IMDb, and other non-video hosting sites.

## Installation

### Download and Run the Script

1. **Download the Script:**
   - Download the `batinapapka.py` file from the repository.
   - Save it to a directory on your computer.

2. **Install Required Python Packages:**
   Ensure you have Python 3 installed. Install the necessary dependencies using pip:
   ```bash
   pip install requests
   ```

3. **Set Your Brave API Key:**
   Open the `batinapapka.py` script in your favorite text editor and replace `"YOUR_API_KEY"` with your actual Brave API key.

   ```python
   API_KEY = "YOUR_API_KEY"
   ```

## Run Using Docker with Automatic Script Download

If you prefer to run the script using Docker and want the script to be downloaded automatically from your GitHub repository when the container is created, follow these steps:

1. **Set Up the Docker Environment:**
   Ensure Docker and Docker Compose are installed on your system.

2. **Prepare the Docker Compose File:**
   Use the following Docker Compose configuration:

   ```yaml
   version: '3.7'

   services:
     rename_script:
       image: python:3.12-slim-bookworm
       container_name: batinapapka
       volumes:
         - /path/to/your/video/files:/videos
       command: >
         sh -c "
         apt-get update &&
         apt-get install -y ffmpeg cron curl &&
         pip install requests unidecode rapidfuzz scikit-learn &&
         curl -o /usr/src/app/batinapapka.py https://raw.githubusercontent.com/davnozdu/batinapapka/main/batinapapkav3.py &&
         echo '0 3 * * * /usr/local/bin/python /usr/src/app/batinapapka.py --api-key YOUR_API_KEY /videos >> /var/log/cron.log 2>&1' > /etc/cron.d/rename_cron &&
         chmod 0644 /etc/cron.d/rename_cron &&
         crontab /etc/cron.d/rename_cron &&
         touch /var/log/cron.log &&
         cron && tail -f /var/log/cron.log
         "
       working_dir: /usr/src/app
       stdin_open: true
       tty: true
       restart: always
       networks:
         - rename_net

   networks:
     rename_net:
       driver: bridge

3. **Build and Run the Docker Container:**
   - Replace `/share/YOUR_VIDEO_FOLDER` with the path to the directory containing your video files.
   - Deploy the stack or use the following command to start the container:
     ```bash
     docker-compose up -d
     ```

   The script will run daily at 3:00 AM, renaming any new video files according to the Brave Search results.

## How to Get Your Brave API Key

1. **Sign up for the Brave Search API:**
   - Visit the [Brave Search API](https://brave.com/search/api/) page.
   - Sign up for an account if you don't have one.
   - Follow the instructions to obtain your API key.

2. **Add the API Key to the Script:**
   - Replace the placeholder in the script with your API key.

## Usage

1. **Run the Script Manually:**
   To rename video files in a directory, run the following command:

   ```bash
   python batinapapka.py /path/to/your/video/files
   ```

2. **Run the Script via Docker:**
   Once the Docker container is running, it will automatically rename files in the specified directory based on the schedule set in the Docker Compose file.

## Example

Before running the script:
```
/path/to/your/video/files
├── video123.mp4
├── movie_trailer.avi
```

After running the script:
```
/path/to/your/video/files
├── 2024-01-01 Example Video Title.mp4
├── 2023-12-31 Another Video Title.avi
```

## Contributing

If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are welcome.

## License

This project is open-source and available under the MIT License.

## Tags

- Python
- Video Renaming
- Brave Search API
- Automation
- Docker
- Cron Jobs
