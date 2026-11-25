#!/usr/bin/env bash
# Install and start the motion-capture systemd service on the Pi.
# Run from the repo folder ON THE PI:   bash pi/install_service.sh
# It asks for your sudo password, because installing a system service
# (something that runs at boot for everyone) needs admin rights.
set -euo pipefail   # stop at the first error; treat unset variables as errors

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE="motion-capture@${USER}"

if [ ! -f "$REPO/.env" ]; then
    echo "Missing $REPO/.env (needs MAC_SERVER_URL and UPLOAD_TOKEN). Aborting." >&2
    exit 1
fi
if [ "$REPO" != "/home/$USER/motion-alert-agent" ]; then
    echo "The service expects the repo at /home/$USER/motion-alert-agent (found $REPO)." >&2
    exit 1
fi

# The service runs the script itself, so stop any copy started by hand first:
# two copies would fight over the camera. Skip this check if the service is
# already running (re-install/update), since then the running copy IS the
# service. The pattern matches only "python3 ... pi/motion_capture.py", not
# e.g. an editor that has the file open.
if ! systemctl is-active --quiet "$SERVICE" \
   && pgrep -u "$USER" -f '^(/usr/bin/)?python3 .*pi/motion_capture\.py' >/dev/null; then
    echo "A hand-started motion_capture.py is running. Stop it (Ctrl+C in its terminal) and re-run this." >&2
    exit 1
fi

echo "Installing the service file..."
sudo cp "$REPO/pi/motion-capture@.service" /etc/systemd/system/
sudo systemctl daemon-reload                 # make systemd re-read its service files
sudo systemctl enable "$SERVICE"             # start at every boot from now on
sudo systemctl restart "$SERVICE"            # (re)start now, so an updated service file takes effect

echo
systemctl status "$SERVICE" --no-pager --lines=5 || true
echo
echo "Done. Follow the log with:  journalctl -u $SERVICE -f"
