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
# or several at once, recursively:
python batinapapka.py --recursive /path/A /path/B
```

CLI flags:

| Flag | Description |
| --- | --- |
| `directories…` | One or more directories. With `--recursive`, each is walked into. |
| `-r`, `--recursive` | Descend into every subdirectory of each given path. |
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

## Deploy the prebuilt image from GHCR

Every push to `main` and every `vX.Y.Z` tag triggers a GitHub Actions workflow that builds a multi-arch image (linux/amd64, linux/arm64) and publishes it to **`ghcr.io/davnozdu/batinapapka`**. No need to clone the repo or build locally on the target host — Docker is the only dependency.

Available tags:

| Tag | When you'd use it |
| --- | --- |
| `latest` | Tracks the tip of `main`. Convenient, but moves under you. |
| `1.2.3`, `1.2`, `1` | Published for every `vX.Y.Z` git tag. Pin one for production. |
| `main` | Same content as `latest`, but explicit branch ref. |
| `sha-<short>` | Immutable pointer to a specific commit. Bulletproof rollback. |

### One-time setup

```bash
# 1. Grab the deploy compose and env template (no git clone needed)
mkdir -p ~/batinapapka && cd ~/batinapapka
curl -fsSL https://raw.githubusercontent.com/davnozdu/batinapapka/main/docker-compose.deploy.yml -o docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/davnozdu/batinapapka/main/.env.example       -o .env

# 2. Fill BRAVE_API_KEY in .env
$EDITOR .env

# 3. Mount your video folder(s) — edit docker-compose.yml's `volumes:` section
$EDITOR docker-compose.yml

# 4. Pull + run
docker compose pull
docker compose up -d
docker compose logs -f batinapapka
```

If the GHCR package is still private you'll see `unauthorized` on `pull`. Either flip the package to public once (GitHub → repo → Packages → batinapapka → Package settings → Change visibility → Public), or `docker login ghcr.io -u <github_user>` using a PAT with `read:packages`.

### Mounting video folders

The image's default command is **`--recursive /videos`**, so everything under `/videos` (including subdirectories) is processed. To point one or more host paths at it, edit the `volumes:` section of `docker-compose.yml`:

```yaml
volumes:
  # one folder:
  - ./videos:/videos

  # multiple folders — keep them on distinct mount points:
  - /mnt/media/movies:/videos/movies
  - /mnt/nas/clips:/videos/clips
  - /home/me/downloads:/videos/downloads

  - batinapapka_state:/state    # leave this line alone — it's persistent state
```

Each mounted subfolder is visited recursively by the same single run, and they share the cache + renamed-files index, so a video already renamed on one mount won't be re-processed when it shows up via another path.

### `.env` reference

```env
# Required
BRAVE_API_KEY=...                       # https://brave.com/search/api/

# Optional
CRON_SCHEDULE=0 3 * * *                 # cron line for the recurring run.
                                        # Set CRON_SCHEDULE= (empty) to make
                                        # the container a one-shot run.
IMAGE_TAG=1.2.3                         # pin a specific image (defaults to latest)
```

### Updating

```bash
docker compose pull && docker compose up -d
```

If you pinned `IMAGE_TAG=1.2.3` in `.env`, bump it and rerun — `pull` will only fetch the new tag.

### Run once, ad-hoc (no compose)

```bash
docker run --rm \
  -e BRAVE_API_KEY=... \
  -e CRON_SCHEDULE= \
  -v /path/to/videos:/videos \
  -v "$PWD/state":/state \
  ghcr.io/davnozdu/batinapapka:latest
```

`CRON_SCHEDULE=` (empty) makes the container exit after one pass; otherwise it would stay attached, tailing the cron log. The default `CMD` is `--recursive /videos`, so a single mount under `/videos` is enough; for several folders, mount them as `/videos/A`, `/videos/B`, … and they get processed in one go.

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
