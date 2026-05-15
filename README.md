# BatinaPapka

**BatinaPapka** is a Python script that renames video files in a directory based on results from the Brave Search API. It strips junk (resolutions, codecs, hosting names, bracketed tags), normalizes Unicode, and prefixes the chosen title with the publication date — or, when nothing is found, with the file's modification date.

## Features

- **Automatic Renaming** — picks the best-matching online title using fuzzy similarity (`rapidfuzz`).
- **Date Handling** — prefers the page's publication date; falls back to mtime.
- **Cached search** — gzipped JSON cache with TTL, atomic writes, file-locked.
- **Format support** — `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`, `.mpeg`, `.mpg`.
- **Rotating log** — `file_renamer.log`, 10 MB × 7 files.
- **Host-level filter** — drops results from Wikipedia/IMDb/social platforms by exact-hostname match (no false-positive substring filtering).
- **Reversible** — every rename is recorded in `.filename_mapping.json` next to the videos.

## Install (bare metal)

Requires Python 3.10+.

```bash
git clone https://github.com/davnozdu/batinapapka.git
cd batinapapka
pip install -r requirements.txt
```

Get a Brave API key at <https://brave.com/search/api/>, then:

```bash
export BRAVE_API_KEY=...your key...
python batinapapka.py /path/to/your/video/files
```

CLI flags:

| Flag | Description |
| --- | --- |
| `--api-key KEY` | Brave Search API key (or set `BRAVE_API_KEY`). |
| `--clean-cache` | Drop the on-disk cache before processing. |
| `--force` | Re-process files already listed in `renamed_files.txt`. |
| `--debug` | Verbose logging. |

## Run with Docker (recommended)

The repo ships with a `Dockerfile` and `docker-compose.yml`. Secrets stay in `.env`, the image is built from pinned dependencies, and a cron job re-runs the renamer on a schedule.

```bash
cp .env.example .env
# edit .env — at minimum set BRAVE_API_KEY and VIDEO_DIR

docker compose up -d --build
docker compose logs -f batinapapka
```

The container does one immediate run on startup, then runs again on `CRON_SCHEDULE` (default: `0 3 * * *` — every day at 03:00). Cache, log and the renamed-files index live in the named volume `batinapapka_state`, so they survive recreates.

## Run combined with cyberdrop-dl

`batinapapka_cyberdrop-dl.yaml` adds `cyberdrop-dl-patched` and an SSH side-car for interactive use. Same `.env` plus two extra variables:

```env
SHELL_USER=youruser
SHELL_PASSWORD=...
```

Then:

```bash
docker compose -f batinapapka_cyberdrop-dl.yaml up -d
ssh -p 2222 youruser@host
```

The script URL is pinned to `BATINAPAPKA_REF` (default `main`). For production set it to a tagged release so a container restart can never silently pull a breaking change.

## Example

Before:

```
/videos
├── video123.mp4
├── movie_trailer.avi
```

After:

```
/videos
├── 2024-01-01 Example Video Title.mp4
├── 2023-12-31 Another Video Title.avi
├── .filename_mapping.json   # reverse-lookup of new → original names
```

## Contributing

Fork, branch, PR. Style: standard library + `requests`/`rapidfuzz`/`Unidecode`; no heavy ML deps.

## License

MIT — see `LICENSE`.
