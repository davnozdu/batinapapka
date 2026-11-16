# Audit and release 1.2.0 (2026-11-15)

The existing main branch and its history contained SSH/Cyberdrop integration but no browser admin service.

## Changes

- Added an authenticated browser console on port 8080 for HTTP/HTTPS links and UTF-8 TXT lists (1 MiB request limit, at most 1000 unique URLs per job). Durable SQLite queue, live process output, cancellation of the process group, retry and recovery after restart. Only one download job runs at a time.
- Browser disconnects do not affect the worker. Container shutdown stops the process; unfinished jobs resume after startup using the persisted Cyberdrop database and partial files. Actual byte-range resumption depends on the source host/Cyberdrop. Explicitly cancelled jobs stay cancelled.
- Cyberdrop 10.15.3 is bundled. Before downloads and manual CLI invocation the wrapper automatically checks PyPI (default interval 24 hours), installs an update into a separate persistent venv, checks dependencies and CLI compatibility, and atomically selects it. Failed updates keep the working version; existing downloads are never upgraded in place. A compatible CLI does not guarantee every host works after an upstream update. Previous venvs remain available in `/state/cyberdrop/runtime` for recovery; monitor volume usage.
- V10 config, database, cache, logs and XDG directories persist under `/state/cyberdrop`. Legacy `AppData` is retained, but V10 uses a different configuration format; old credentials/settings may need migration into `config.yaml`.
- Updated requests, rapidfuzz and Unidecode; added pinned Flask and Waitress. Python minimum is now 3.12 for the combined image and 3.11 for the standalone renamer.
- Removed sudo from the combined image. SSH host keys are generated per installation and persisted, rather than embedded in a shared image. Browser access requires credentials; POSTs require a custom header with no CORS support; CSP prevents framing and remote scripts. Compose binds the admin port to localhost by default. Use a VPN, SSH tunnel or HTTPS reverse proxy for remote access.
- Compose now respects `CRON_SCHEDULE` from `.env`. The combined image stays running even when cron or Brave is disabled. Renaming requires a Brave key or a configured ThePornDB token. Scheduled credentials and arguments live in a private script instead of unescaped cron text.

## Metadata-first matching

Read-only ThePornDB integration precedes Brave: exact OSHASH matching in scenes/movies, then conservative matching of embedded titles/filenames against scene metadata. Exact returned fingerprints are verified; ambiguous or weak title matches are rejected. A match uses the catalog release date and title in the existing `YYYY-MM-DD Title.ext` format. No match or a catalog/network failure falls back to Brave. Catalog-only mode leaves unmatched files untouched if no Brave key is configured.

The browser can save/disable the catalog token and enqueue a renamer run in the same durable queue. Stored tokens are mode 0600 and never returned by the API. The `.env` alternative is `TPDB_API_KEY`. The video itself is not uploaded; fingerprint and query names are sent. OSHASH identifies matching file versions; re-encoded files may require the name fallback. This does not implement perceptual video hashing.

The supplied test token returned HTTP 200. A live OSHASH lookup returned one record with a verified matching hash, and 20 sampled records contained valid release dates. Test tokens were not saved in project files. Integration is based on [ThePornDB's reference client](https://github.com/ThePornDatabase/namer/blob/main/namer/metadataapi.py) and the [OSHash specification](https://opensubtitles.github.io/oshash/). The old porndb.me API was shut down on 2026-10-16; [migration notice](https://porndb.me/migration/).

## Renamer correctness

- Downloads from the web queue and the SSH wrapper share the renamer lock, avoiding renames during managed downloads. A lock-file access failure stops the renamer instead of proceeding unlocked. Raw `/opt/cyberdrop/venv/bin/cyberdrop-dl` bypasses this wrapper and should not be used while renaming.
- Skip symlinks, non-files, recently modified files (60 seconds, configurable with `BATINAPAPKA_MIN_FILE_AGE`) and files whose inode/size/mtime changed during processing. External writers do not cooperate with the lock; the stability check reduces that risk but cannot eliminate every external race.
- Atomic hard-link creation refuses to overwrite existing destinations. The original name is removed only after an atomic, fsynced mapping write succeeds. Filesystems without hard-link support report an error and retain the original file; this requires a filesystem supporting regular file hard links, including on mounted storage.
- UTF-8 names and collision suffixes obey the 255-byte filename limit; control characters are removed.
- New rename-index entries use absolute paths, so equal names in different directories no longer suppress new work. Legacy basename entries remain recognized for compatibility.
- Invalid cache structures/timestamps are ignored. Invalid mapping JSON is retained and blocks renaming that file rather than destroying history.
- API authentication failures and exhausted request retries leave files untouched. A successful search with no match still uses the existing filename/mtime fallback. The renamer returns failure when any file or target failed.
- Required name tokens match whole words. Dates are validated; page dates more than two years away from an explicit filename year are rejected. Search matching remains heuristic: Brave page publication dates can refer to a webpage update rather than a media release. Review mappings; this is not authoritative identification.

## Validation

26 unit/integration tests cover queue persistence, browser authentication, CSRF rejection, TXT upload, input validation, real subprocess completion/cancellation/shutdown, rename mappings/repeated runs, corrupt state, changing files, UTF-8 collisions, name/date matching and Brave auth failures. Live Brave API returned 19 normalized results with the provided temporary test key; no key was saved in project files. Actual Cyberdrop v10 invocation completed against an unsupported test URL and emitted its final statistics.

`pip-audit -r requirements.txt -r requirements-admin.txt` found no known vulnerabilities in the resolved application dependencies at audit time. A full local environment audit also identified an outdated development-venv pip (26.1.2); the image/updater upgrades pip before installing Cyberdrop. This is not an OS package security scan.

CI runs the tests and dependency audit before publishing. The combined image additionally starts the actual container, checks HTTP/API/CLI integration and restarts it to verify persistence. Slim image: amd64 + arm64; combined image: amd64. Native browser automation was unavailable in the session, so visual browser QA could not be completed.

## Follow-up 1.2.1

A real existing V9 configuration exposed a startup regression that the initial
fresh-install smoke test did not cover: Cyberdrop unconditionally scans cwd
for legacy AppData. The wrapper now resolves input paths before switching to
a dedicated V10 working directory. Old files are retained; SQLite backup
copies legacy history only into an absent/empty V10 database. The regression
is covered by the real Cyberdrop/container checks as well as database tests.

Per-job rotation alone did not bound aggregate log retention, native Cyberdrop
DEBUG files or cron output. Added a 30-day/200-MiB budget for recognized job
log files, protected active logs, bounded native CSV diagnostics, redirected
duplicate native DEBUG logs to `/dev/null`, bounded cron capture, and Docker
logging limits in all compose files. Only recognized diagnostic paths are
pruned; URLs, settings, queue/download history and video files are retained.
The cleanup interval is five seconds, so this is a retention budget rather
than a hard filesystem quota. User-specified raw SSH logging overrides are
outside the browser queue's log budget.

31 tests now cover the previous behavior plus retention budgets, age pruning,
symlink safety, cron rotation and legacy history preservation. The actual V10
process completed with a legacy V9 settings file present and left it intact.

## Follow-up 1.3.0

Added browser deletion of inactive jobs and their own artifacts, with an
atomic SQLite status check against worker claiming/retry and path validation.
Video files, configuration and Cyberdrop's history database are never deleted.
Added pause/resume and persistent recovery tests using real subprocesses and
partial-file fixtures. Running jobs are recovered automatically after image
replacement; explicit pauses/cancellations remain under user control.

Added a Telegram adapter using the same queue. Private sender IDs and chat IDs
must match an explicit allowlist, including callback authorization. Token and
allowlist are configurable in the authenticated browser, secrets are mode
0600 and never exposed by GET APIs, and request exceptions are sanitized to
avoid logging the token embedded in Telegram API URLs. Menu and inline buttons
cover queue/status/pause/resume/cancel/log operations. No per-file messages are
emitted: notification state is persisted per terminal job. Telegram update
offsets and pending-link receipts prevent duplicate job creation on replay.

45 tests cover queue controls, deletion/data preservation, process resumption,
Telegram authorization, settings, button callbacks, replay protection and
whole-job notifications/retry. A live getMe request verified the supplied bot
token; the user opted to enter runtime configuration themselves, so no bot
was activated and no live Telegram messages were sent during development.

## Automatic upstream image refresh (2026-11-16)

Added an hourly PyPI/GHCR comparison workflow and reusable exact-version builds.
The checker skips prereleases/yanked releases, recognizes unlabelled legacy
images, propagates registry/network failures, skips current versions and avoids
scheduled downgrades. Build publication is serialized per image/ref. The
combined image carries a machine-readable bundled-version label and a
`cyberdrop-X.Y.Z` tag; the existing smoke test verifies the installed CLI version
with self-updating disabled. No PAT or bot credentials are required in CI.
Normal builds resolve the current stable version so later application pushes
do not reinstall the old Dockerfile fallback. Local default updated from
10.15.3 to 10.16.0.

53 tests and actionlint validation passed locally. A newer bundled CLI now supersedes an older persistent runtime immediately; newer persisted runtimes remain preferred. Real scheduled-workflow
validation is performed by manual dispatch against the newly released 10.16.0.
