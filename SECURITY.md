# Security notes (Raspberry Pi side only)

Everything here is local to this project on the Pi. Nothing about the SQLite schema,
the JSON sent to the API, the live-view HTTP routes, or anything the C# command center
expects has changed - only how secrets are stored and logged on the Pi.

## What changed

1. **Secrets split out of `config.yaml` into `config/secrets.yaml`.**
   `config.yaml` stays safe to share, screenshot, or zip up for a bug report.
   `secrets.yaml` holds the actual RTSP URL (with password), the live-view auth
   token, and the API endpoint URL. It's listed in `.gitignore` and should be
   `chmod 600`, owned only by the service account the pipeline runs as.
   `config/config.py` merges it in automatically if present (see
   `PipelineConfig.load()`); nothing else needs to change.

2. **Environment-variable overrides.** Any field can now also be set via
   `PIPELINE_<SECTION>_<FIELD>` (e.g. `PIPELINE_CAMERA_RTSP_URL=...`), highest
   precedence of all. This means you can skip having any secret on disk at all
   and inject it via a systemd `EnvironmentFile=` with its own locked-down
   permissions, if you'd rather do that than use `secrets.yaml`.

3. **Credential redaction in logs.** `camera.py` was printing the full RTSP URL
   (password included) to stdout/journald on every connect, reconnect, and
   error. It now strips `user:pass@` before anything gets logged. This was the
   biggest actual leak in the code - locking down the config file alone
   wouldn't have stopped this from leaking straight back out through
   `journalctl -u detection-pipeline`.

4. **Live view authentication turned on.** `live.auth_token` was `null`,
   meaning anyone who could reach the Pi's port 8090 on your LAN could watch
   the camera feed with no login at all. A random token now ships in
   `secrets.yaml` (regenerate anytime with
   `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`).

5. **Dedicated service account + systemd sandboxing.** The unit file now runs
   as a new unprivileged `pipeline` system user instead of `pi` (which
   typically has sudo + SSH access), plus standard systemd hardening
   (`ProtectSystem=strict`, `NoNewPrivileges`, restricted write paths, etc.).
   `secure_setup.sh` creates that user and fixes ownership/permissions for you.

6. **`.gitignore` now also excludes** `config/secrets.yaml`, `*.env`, `data/`,
   `output/`, and `*.db*` - the SQLite buffer and on-disk crop JPEGs contain
   plate photos and detection history, not just config, so they shouldn't end
   up in git or a shared zip either.

## Do this now

The RTSP password in the original `config.yaml` was already sitting in plain
text in a project folder that just got zipped up and shared outside the Pi
(that's how it ended up here). Moving it into `secrets.yaml` fixes it going
forward, but **rotate that camera's password** in the camera's own web UI -
relocating an already-exposed credential doesn't un-expose it.

## What this does and doesn't protect against

Does protect against:
- The config file being casually copied, zipped, screenshotted, or committed
  to git and leaking credentials as a side effect (this project's own history,
  right now, is a live example of exactly that).
- Other local, non-root accounts on the same Pi reading the secrets file.
- Credentials leaking into journald/console logs.
- Unauthenticated viewers on the LAN watching the live camera feed.
- The pipeline process running with the same privileges as your login/SSH user.

Does **not** protect against:
- Anyone with root access to this same Pi - file permissions can't hide
  secrets from root, only from other accounts. If you need that too, the next
  step up is full-disk encryption (LUKS) on the SD card/SSD, which is a bigger
  change and not covered here.
- Someone physically removing the SD card and reading it on another machine,
  for the same reason.
- The RTSP credential itself being weak/reused - rotating it (above) is the
  actual fix for that, not file placement.
