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

Requires Python 3.11+ (3.12+ for the Cyberdrop image).

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

When you run interactively (stderr is a TTY) the terminal shows a **two-line dashboard** updated in place:

```
[ 47/420] searching: Big.Buck.Bunny.1080p.mp4
     last renamed: Mia.Khalifa.example.mp4 → 2026-03-13 Mia Khalifa.mp4
```

Top line — counter + the file currently being looked up against Brave. Bottom line — the most recent successful rename. Both lines repaint in place; nothing scrolls until an **error** happens, which prints permanently above the dashboard so a piped log still has it:

```
[ 53/420] ERROR locked: in_progress_download.mp4
[ 54/420] searching: Mia.Khalifa.example.mp4
     last renamed: sintel_trailer_2010.mkv → 2026-05-15 Sintel.mkv
```

Every per-file detail (matched title, score, cache hits, …) still lands in `/state/file_renamer.log`; the dashboard is just a "I'm alive" indicator.

In **`-v / --verbose`** the dashboard is replaced with the full INFO stream — every `Processing …`, `Brave returned N results`, `Renamed X -> Y` event printed on its own line. **Off-TTY** runs (cron, `docker logs`) always behave like `--verbose`: full line-by-line output, no carriage returns.

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

## Browser administration (Cyberdrop image, v1.2.0+)

Open **http://localhost:8080** on the Docker host. The login defaults to
`SHELL_USER` / `SHELL_PASSWORD`; set `ADMIN_USER` and `ADMIN_PASSWORD` to use
separate browser credentials. No Brave key is needed for downloads; without
it the automatic renamer is disabled.

The page supports:

- One or many HTTP/HTTPS links, one per line.
- Uploading a UTF-8 `.txt` file with links (up to 1 MiB, 1000 links per job).
- Persistent queue/history, a live Cyberdrop output log, cancel and retry.

You can close the browser or SSH terminal: queued/browser-managed downloads
continue in the container. Reopening the page restores their status. If the
container restarts, unfinished jobs are requeued and Cyberdrop uses its history
and partial files to resume where the source host permits. Cancellation stops
the download process and its children, retains partial files and never deletes
finished downloads. A zero exit code means the process finished; check the log
for unsupported URLs or individual download errors.

The port binds to loopback by default. For remote access:

```bash
ssh -L 8080:127.0.0.1:8080 user@docker-host
# Then open http://localhost:8080 on your computer.
```

Alternatively set `ADMIN_BIND_IP` to the host's private LAN/VPN IP in `.env`.
Use an HTTPS reverse proxy if exposing the UI beyond a trusted connection.
`ADMIN_PORT` changes the published host port. Existing deployments must update
the compose file as well as pulling the image to publish the new port.

```bash
curl -fsSL https://raw.githubusercontent.com/davnozdu/batinapapka/main/batinapapka_cyberdrop-dl.yaml -o docker-compose.new.yml
# Copy the new environment/ports/stop_grace_period settings into your existing
# compose file, preserving your volume mounts and existing state volume name.
docker compose pull
docker compose up -d
```

Cyberdrop updates automatically **before starting a download or SSH CLI**, at
most once per `CYBERDROP_UPDATE_INTERVAL` seconds (default 86400). It builds
and verifies a separate venv before switching; a failed update retains the
working version. Set `CYBERDROP_AUTO_UPDATE=0` to disable. To check immediately,
remove `/state/cyberdrop/runtime/status.json` and run `cyberdrop-dl --update-only`.
Updates never replace a running download. The bundled baseline is 10.15.3.

The queue and current V10 config/database/cache/logs live in
`/state/cyberdrop/`; downloads go into `/videos`. V10 uses
`/state/cyberdrop/config.yaml`; legacy AppData remains preserved for migration.
SSH remains available, including `cyberdrop-dl download URL` and `batinapapka`.
Use the browser to manage a download if you want browser cancellation/history;
manual SSH CLI processes are separate from the web queue.

The renamer skips files modified in the last 60 seconds and waits for managed
downloads to release their shared lock. Run it again after downloads finish if
a scheduled run was skipped. The renamer uses atomic hard links for safe
renaming, so the video filesystem must support hard links. See [AUDIT.md](AUDIT.md)
for findings, validation and limits.

## Prefer catalog metadata before web search

In the admin page open **Поиск названий · ThePornDB**, enter a token from
<https://theporndb.net/user/api-tokens> and save. Alternatively set `TPDB_API_KEY`
in `.env`. The token is stored only in `/state/metadata-settings.json` with
permissions 0600 and never returned to the browser. Blank input preserves it;
uncheck the catalog option to disable lookup. Changes apply on the next
renamer run. Use **Переименовать загруженные видео** to enqueue that run from
the browser; its log, cancellation and retry use the same queue.

The renamer checks exact OSHASH matches first, then embedded title/filename
metadata in ThePornDB, then Brave if no confident catalog match exists. A
catalog match uses its **release date and title**: `YYYY-MM-DD Title.ext`.
Ambiguous title matches are rejected. A network/auth error at the catalog also
falls back to Brave. Without a Brave key, unmatched catalog-only files are
left untouched. No video is uploaded; only the fingerprint and search names.
OSHASH is fast (128 KiB read) but does not recognize arbitrary re-encodings.

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

## Log retention and V9 compatibility (v1.2.1+)

- Job output rotates at 5 MB with one backup (about 10 MB per job).
- Managed job logs are cleaned after 30 days and have a combined 200 MiB
  budget. Cleanup runs every 5 seconds; active job output is protected and
  remains bounded by its own rotation. Native diagnostic CSV files are trimmed
  to 1 MiB while active. Limits can be briefly exceeded between cleanup passes.
- Duplicate native Cyberdrop DEBUG output is discarded for browser jobs; the
  rotating browser log retains INFO output, progress, errors and final stats.
- Cron output rotates at 5 MiB with two backups (15 MiB). The renamer's own
  log remains 10 MiB with seven backups (80 MiB).
- Compose configures Docker output rotation at 10 MB with three files. Update
  the compose file to apply this host-side setting to an existing installation.

Cyberdrop V10 checks for `AppData` in the current working directory, even with
an explicit V10 config. The wrapper now starts from a dedicated `work-v10`
directory so a preserved V9 configuration cannot block startup. Old settings
are retained and are not automatically converted to the incompatible V10
schema. A legacy SQLite history is backed up into the V10 location only when
the new database is absent/empty; an existing V10 database is never replaced.
After updating, use **Повторить** on a job that failed with `Migration Error`.

## Queue controls and Telegram bot (v1.3.0+)

The browser has **Удалить** for queued, paused and terminal jobs. Deletion
removes that job's row, submitted URL list and diagnostic logs; downloaded
videos and Cyberdrop's download-history database are preserved. Running jobs
must be paused/cancelled and stopped first. Active jobs stay at the top of the
list even when there is a long history.

**Пауза** stops the current process while retaining the job, database and
partial files. **Продолжить** resumes a paused, cancelled or failed job under
the same ID. During an image update/restart, unfinished running jobs return to
the queue automatically. Explicitly paused/cancelled jobs wait for the user to
continue them. Byte-level resumption depends on the source host and Cyberdrop;
files already in its completed history are normally skipped.

To enable the bot, open **Telegram-бот** in the browser settings, enter the
BotFather token and your numeric user ID, enable it and save. Blank token input
preserves the existing token. Settings apply without restarting the container.
The token is stored in `/state/telegram-settings.json` with mode 0600 and is
never returned by the settings API. No token or personal ID is baked into the
image. The `.env` alternatives are `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_ALLOWED_USER_IDS` (comma-separated).

The bot speaks **only to allowed users in private chats**. Empty allowlists
cannot enable the bot; other users and groups are ignored, including button
callbacks. The bottom menu contains **Скачать**, **Активное задание**,
**Очередь**, **Пауза**, **Продолжить**, **Отменить**, **Помощь**. Individual jobs
also have buttons, so command names and job IDs do not need to be memorized.

Send a link (or several links on separate lines) to enqueue one job. The bot
acknowledges it and sends a final message when the entire job ends, with its
result and final Cyberdrop output; it never sends per-file notifications.
Failures/cancellations are labelled accordingly and provide a resume button.
The active view shows the latest Cyberdrop output, including the current file
when Cyberdrop reports it. Browser-submitted jobs are visible to the permitted
bot administrators as well. Text commands remain available: `/start`, `/id`,
`/active`, `/status`, `/queue`, `/pause ID`, `/resume ID`, `/cancel ID`.

Polling offsets, job origin and notification acknowledgements persist in the
queue database. Replayed Telegram link updates do not create duplicate jobs,
even around restarts. A notification whose delivery fails is retried after
reconnection. A rare crash/network ambiguity between Telegram accepting a
message and the local acknowledgement can cause a repeated final message;
there are no periodic or per-file notification streams.

This uses Telegram long polling and needs outbound HTTPS, not a published
webhook port. Use one running instance with a given token, with no active
webhook for that bot. API outages never stop the local download queue.

## Automatic Cyberdrop image releases

The **Update Cyberdrop image** workflow checks PyPI every hour at minute 17
(UTC; GitHub schedules may be delayed). It resolves the newest non-yanked
stable `X.Y.Z` release and compares it to the
`io.batinapapka.cyberdrop.version` label on the published combined image.
When a newer version exists, it calls the shared build workflow with that
exact version and builds **only** `batinapapka-cyberdrop`.

Unit tests, dependency auditing, and the real-container web/queue/SSH/restart
checks run before publication. The smoke check also runs the bundled CLI with
runtime self-updating disabled and verifies its exact version. Failed checks
leave the previously published image in place. Successful builds update:

- `ghcr.io/davnozdu/batinapapka-cyberdrop:latest`
- `ghcr.io/davnozdu/batinapapka-cyberdrop:main`
- `ghcr.io/davnozdu/batinapapka-cyberdrop:cyberdrop-X.Y.Z`

The dependency version tag identifies the bundled Cyberdrop release; pin a
manifest digest if you need immutable image contents across application code
changes. Scheduled dependency refreshes do not rewrite the application release
semver tags. No extra PAT, repository auto-commit, or third-party service is
needed: publication uses the workflow's scoped `GITHUB_TOKEN`.

You can also run **Update Cyberdrop image → Run workflow** manually. Enable
`force` to rebuild the current version; otherwise a current/newer published
version skips the build. Normal code/release builds resolve the latest stable
Cyberdrop too, preventing a stale Dockerfile default from undoing an upstream
refresh. Local builds default to 10.16.0, overridable with
`--build-arg CYBERDROP_VERSION=...`.

This publishes an image; it does not restart your deployment automatically.
Pull/recreate the container as usual, retaining `/state` and `/videos`:

```bash
docker compose pull
docker compose up -d
```
