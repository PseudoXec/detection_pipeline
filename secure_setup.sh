#!/usr/bin/env bash
# secure_setup.sh
# ----------------
# Run this ONCE on the Raspberry Pi, as root, from inside the project
# directory, after copying the project over but before starting the
# systemd service. Safe to re-run any time - it only fixes ownership and
# permissions, it never touches pipeline logic or data.
#
#   sudo ./secure_setup.sh
#
# What it does:
#   1. Creates a dedicated, unprivileged "pipeline" system user (no login
#      shell, no home dir) if one doesn't already exist, so the pipeline
#      isn't running as "pi" (which usually has sudo + SSH access).
#   2. Locks config/secrets.yaml down to chmod 600, owned by that user -
#      no other local account (or process running as another user) can
#      read it.
#   3. Locks down data/ and output/ similarly - the SQLite buffer and JPEG
#      crops contain plate photos and detection history, not just config.
#   4. Warns (loudly) if config/secrets.yaml doesn't exist yet or still
#      has its zip/extract-default permissions.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run this with sudo: sudo ./secure_setup.sh" >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="pipeline"

echo "[1/4] service user"
if ! id "$SERVICE_USER" &>/dev/null; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
  echo "  created system user '$SERVICE_USER'"
else
  echo "  '$SERVICE_USER' already exists"
fi

echo "[2/4] project ownership"
chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR"

echo "[3/4] secrets file"
SECRETS_FILE="$PROJECT_DIR/config/secrets.yaml"
if [[ -f "$SECRETS_FILE" ]]; then
  chmod 600 "$SECRETS_FILE"
  echo "  $SECRETS_FILE -> chmod 600, owned by $SERVICE_USER"
else
  echo "  WARNING: $SECRETS_FILE does not exist yet."
  echo "  Copy config/secrets.yaml.example -> config/secrets.yaml and fill it in,"
  echo "  then re-run this script. Until then rtsp_url/auth_token/api endpoint"
  echo "  stay null and the pipeline has no camera to read from."
fi
# config.yaml itself is meant to be shareable, but there's no reason for
# other local accounts to be able to WRITE to it
chmod 644 "$PROJECT_DIR/config/config.yaml" 2>/dev/null || true

echo "[4/4] runtime data directories"
mkdir -p "$PROJECT_DIR/data" "$PROJECT_DIR/output"
chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/data" "$PROJECT_DIR/output"
chmod 750 "$PROJECT_DIR/data" "$PROJECT_DIR/output"

echo
echo "Done. Next steps:"
echo "  sudo cp detection-pipeline.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable --now detection-pipeline"
