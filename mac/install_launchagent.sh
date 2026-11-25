#!/usr/bin/env bash
# Install and start the upload server as a macOS LaunchAgent.
# Run from the repo folder ON THE MAC:   bash mac/install_launchagent.sh
#
# launchd is macOS's service manager (the equivalent of systemd on the Pi).
# A *LaunchAgent* runs as your user and starts when you log in; it can use
# your Homebrew Ollama, which also runs as a LaunchAgent. (A LaunchDaemon
# would start at boot without login, but runs as root, which this server
# doesn't need and shouldn't have.)
#
# The plist (launchd's config file) is generated here rather than stored in
# the repo, because it must contain absolute paths, which include your
# username, and the repo is public.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="local.motion-alert.server"                       # unique name for this job
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/motion-alert"               # standard place for user app logs
DOMAIN="gui/$(id -u)"                                   # "this user's login session" in launchd terms

[ -x "$REPO/.venv/bin/uvicorn" ] || { echo "No venv at $REPO/.venv (see CLAUDE.md, Mac setup)." >&2; exit 1; }
[ -f "$REPO/.env" ] || { echo "Missing $REPO/.env (needs UPLOAD_TOKEN)." >&2; exit 1; }
if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && ! launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    echo "Something else is already listening on port 8000 (a server started by hand?). Stop it first." >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$PLIST")"

# What each key means:
#   ProgramArguments   the command to run, one argument per <string>
#   WorkingDirectory   run from the repo root (server.py finds .env from there)
#   RunAtLoad          start as soon as the agent is loaded (i.e. at login)
#   KeepAlive          if it ever exits, start it again
#   ThrottleInterval   wait at least 10 s between restarts (no tight crash loops)
#   EnvironmentVariables  PYTHONUNBUFFERED=1: when output goes to a file,
#                      Python holds print() output in a buffer and writes it
#                      later (or never, if killed). This makes it write at once.
#   StandardOutPath /  where print()/log output goes
#   StandardErrorPath
# --no-access-log: uvicorn would otherwise log every HTTP request; our own
# "Queued ..." lines already record each upload, and the file isn't rotated.
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>            <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$REPO/.venv/bin/uvicorn</string>
        <string>--app-dir</string> <string>mac</string>
        <string>server:app</string>
        <string>--host</string>    <string>0.0.0.0</string>
        <string>--port</string>    <string>8000</string>
        <string>--no-access-log</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key> <string>1</string>
    </dict>
    <key>WorkingDirectory</key>  <string>$REPO</string>
    <key>RunAtLoad</key>         <true/>
    <key>KeepAlive</key>         <true/>
    <key>ThrottleInterval</key>  <integer>10</integer>
    <key>StandardOutPath</key>   <string>$LOG_DIR/server.log</string>
    <key>StandardErrorPath</key> <string>$LOG_DIR/server.log</string>
</dict>
</plist>
EOF
plutil -lint "$PLIST" >/dev/null   # check the XML is valid before handing it to launchd

# If an older version is loaded, unload it first so the new settings apply.
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
# Load it (and, because of RunAtLoad, start it). Unloading takes a moment, and
# loading too soon fails with "Input/output error", so retry a few times.
for attempt in 1 2 3 4 5; do
    launchctl bootstrap "$DOMAIN" "$PLIST" 2>/dev/null && break
    [ "$attempt" = 5 ] && { echo "launchctl bootstrap failed" >&2; exit 1; }
    sleep 1
done

echo "Installed $PLIST"
echo "Log:     tail -f $LOG_DIR/server.log"
echo "Status:  launchctl print $DOMAIN/$LABEL | grep -E 'state|pid'"
echo "Restart: launchctl kickstart -k $DOMAIN/$LABEL    (after a git pull)"
echo "Remove:  launchctl bootout $DOMAIN/$LABEL && rm $PLIST"
