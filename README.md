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
```

The compose files pin `image:` to `:latest`. If you want to lock onto a specific version (e.g. `1.0.0`), edit the `image:` line in the compose file directly.

### Updating

```bash
docker compose pull && docker compose up -d
```

### Run the renamer by hand (outside the cron schedule)

The cron job is convenient but sometimes you want a run right now — e.g. after a fresh batch of `cyberdrop-dl` downloads. The image installs a short wrapper, `/usr/local/bin/batinapapka`, that calls the script with whatever args you pass:

**From the host (uses the running container's env automatically):**

```bash
docker compose exec batinapapka            batinapapka --recursive /videos
# or, on the cyberdrop stack:
docker compose -f batinapapka_cyberdrop-dl.yaml exec batinapapka_cyberdrop  batinapapka --recursive /videos
```

**Inside an SSH session (cyberdrop image only).** `BRAVE_API_KEY` is exported to `/etc/environment` by the entrypoint, so it's in your shell env after login:

```bash
ssh -p 2222 "$SHELL_USER"@host
batinapapka --recursive /videos        # default — live progress on the terminal
batinapapka -v --recursive /videos     # stream every INFO line instead of progress
batinapapka --debug /videos/movies     # DEBUG verbosity (HTTP traces, etc.)
batinapapka -q --recursive /videos     # silence the progress bar, only print final stats
batinapapka --clean-cache --recursive /videos  # drop cache first
```

When you run interactively (stderr is a TTY) the terminal shows:

```
=== /videos/movies ===
Found 42 candidate file(s)
[ 1/42] renamed Big.Buck.Bunny.1080p.mp4 -> 2024-08-12 Big Buck Bunny.mp4
[ 2/42] renamed sintel_trailer_2010.mkv  -> 2026-05-15 Sintel.mkv
[ 3/42] ERROR locked: in_progress_download.mp4
[ 4/42] searching: Mia.Khalifa.example.mp4_       ← live rotating line
```

Each completed file (success, skip, or error) prints a permanent line above; the bottom line is the rotating status of the file currently being looked up against Brave. Errors are also written to stderr so they survive a piped log too. The full INFO/DEBUG trace is in `/state/file_renamer.log` regardless of which mode you picked.

If you really want to bypass the container entirely:

```bash
docker run --rm \
  -e BRAVE_API_KEY=... \
  -e CRON_SCHEDULE= \
  -v /path/to/videos:/videos \
  -v "$PWD/state":/state \
  ghcr.io/davnozdu/batinapapka:latest --recursive /videos
```

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

A second image, **`ghcr.io/davnozdu/batinapapka-cyberdrop`**, ships the renamer plus `cyberdrop-dl-patched` and an SSH side-car. Same workflow as the slim deploy, just a different compose file:

```bash
mkdir -p ~/batinapapka-cyberdrop && cd ~/batinapapka-cyberdrop
curl -fsSL https://raw.githubusercontent.com/davnozdu/batinapapka/main/batinapapka_cyberdrop-dl.yaml -o docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/davnozdu/batinapapka/main/.env.example                 -o .env

$EDITOR .env       # set BRAVE_API_KEY, SHELL_USER, SHELL_PASSWORD
$EDITOR docker-compose.yml   # mount your video folder(s)

docker compose pull
docker compose up -d
ssh -p 2222 "${SHELL_USER}"@host    # cyberdrop-dl is on PATH inside
```

This image is amd64-only (some `cyberdrop-dl-patched` transitive deps have no prebuilt arm64 wheels). The slim `batinapapka` image stays multi-arch.

**Persistent cyberdrop state.** On startup the entrypoint sets up symlinks inside the SSH user's home directory so cyberdrop-dl reads/writes a location that survives `docker compose down && up`:

| In the container | Backed by (on the host) |
| --- | --- |
| `~/Downloads` → `/videos` | the bind-mounted videos folder |
| `~/AppData`  → `/state/cyberdrop/AppData` | `batinapapka_state` named volume |
| `~/URLs.txt` → `/state/cyberdrop/URLs.txt` | `batinapapka_state` named volume |

So `cyberdrop-dl` keeps its config (`AppData/Configs/Default/settings.yaml`), its cache DB (`AppData/Cache/cyberdrop.db`) and your `URLs.txt` between container recreates. The first run after a fresh install adopts any real files the user already wrote into `~/AppData` / `~/URLs.txt` before the symlinks existed and moves them into `/state/cyberdrop/` once.

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
