"""Take a photo whenever the PIR sensor sees motion, and upload it to the Mac.

Run on the Pi from the repo folder:
    python3 pi/motion_capture.py

Needs a .env file in the repo root with:
    MAC_SERVER_URL=http://<your-mac-name>.local:8000/upload
    UPLOAD_TOKEN=<same value as on the Mac>

Flow for one motion event:
    motion -> rpicam-still saves a JPEG into RAM (/dev/shm) -> POST it to the Mac
      - upload OK (202): delete it. It never touched the SD card at all.
      - upload failed:   move it to ~/captures (SD card) so nothing is lost;
        that folder is capped at MAX_KEPT_PHOTOS so a long Mac outage can't
        fill the card.

Golden rule: nothing that goes wrong with one photo may stop the sensor loop.
A crashed script means a silent security system.
"""

import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests
from gpiozero import MotionSensor

# --- Settings ---------------------------------------------------------------
PIR_PIN = 17            # GPIO pin the PIR sensor's OUT wire is on (also in pir_test.py)
COOLDOWN_SECONDS = 10   # ignore new motion for this long after a capture
SAVE_DIR = Path.home() / "captures"   # on the SD card: only photos whose upload failed
CAMERA_TIMEOUT = 30     # seconds; if rpicam-still hangs, give up instead of freezing forever
UPLOAD_TIMEOUT = (3, 15)  # (connect, read) seconds; the Mac replies 202 in ~2 ms when healthy
MAX_KEPT_PHOTOS = 100   # photos kept on the SD card after failed uploads (~2 MB each -> ~200 MB max)

# /dev/shm is a folder that lives in RAM, not on the SD card. Writing there is
# fast, doesn't wear out the card (SD cards survive a limited number of
# writes), and private photos vanish on reboot. Fall back to the SD card if
# it doesn't exist (it always does on Raspberry Pi OS / Debian).
TEMP_DIR = Path("/dev/shm/motion-capture") if Path("/dev/shm").is_dir() else SAVE_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent   # pi/ -> repo root


def load_env(path: Path) -> None:
    """Minimal .env reader: copies KEY=value lines into os.environ.

    Why not python-dotenv like on the Mac? It isn't installed on the Pi, and
    Debian blocks `pip install` into the system Python (PEP 668). A few lines
    of our own code are simpler than a virtualenv for one small feature.
    Like python-dotenv, it never overrides variables that are already set.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue                          # skip blanks, comments, junk
        if line.startswith("export "):        # allow shell-style "export KEY=value"
            line = line[len("export "):]
        key, value = line.split("=", 1)       # split on the FIRST "=" only (values may contain "=")
        # Strip matching quotes: KEY="abc" should mean abc, not "abc" with the
        # quote characters, which would silently make the token wrong.
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


load_env(REPO_ROOT / ".env")
MAC_SERVER_URL = os.environ.get("MAC_SERVER_URL", "").strip()
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN", "").strip()


def capture_photo() -> Path:
    """Take a photo into TEMP_DIR (RAM) and return its path."""
    filename = TEMP_DIR / f"motion_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    subprocess.run(
        ["rpicam-still", "-n", "-t", "500", "-o", str(filename)],
        check=True,             # raise CalledProcessError if the camera command fails
        capture_output=True,    # keep its chatty output off our screen (but available in errors)
        timeout=CAMERA_TIMEOUT, # raise TimeoutExpired if it hangs
    )
    if not filename.is_file():  # exit code 0 but no photo: treat as a failure, not a crash later
        raise FileNotFoundError(f"rpicam-still wrote no file ({filename.name})")
    return filename


def upload_photo(path: Path) -> bool:
    """Send one photo to the Mac. Returns True if the Mac accepted it.

    The photo is the raw request body (not a multipart form): the Mac's
    server checks the token *before* reading any of the body, which only
    works with a raw body. `data=f` streams the open file, so the whole
    photo never has to be loaded into the Pi's memory at once, and requests
    sets the Content-Length header from the file's size (the Mac requires it).
    """
    with open(path, "rb") as f:
        response = requests.post(
            MAC_SERVER_URL,
            data=f,
            headers={"Authorization": f"Bearer {UPLOAD_TOKEN}", "Content-Type": "image/jpeg"},
            timeout=UPLOAD_TIMEOUT,
        )
    if response.status_code == 202:
        return True
    # Show the Mac's short error reason (e.g. "bad or missing token"), never our token.
    print(f"  Mac refused the upload: {response.status_code} {response.text[:100]}")
    return False


def keep_for_later(path: Path) -> None:
    """Upload failed: move the photo from RAM to the SD card so it isn't lost."""
    SAVE_DIR.mkdir(exist_ok=True)
    shutil.move(str(path), SAVE_DIR / path.name)
    trim_saved_photos()


def trim_saved_photos() -> None:
    """Delete the oldest kept photos so at most MAX_KEPT_PHOTOS remain.

    Sorted by the file's modification time, not its name: the Pi has no clock
    battery, so right after boot (before it syncs time from the internet) the
    clock, and therefore the timestamp in a filename, can be wrong.
    """
    try:
        photos = sorted(SAVE_DIR.glob("motion_*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in photos[:-MAX_KEPT_PHOTOS]:     # everything except the newest N
            old.unlink(missing_ok=True)
    except OSError as e:
        # Cleanup failing (e.g. SD card turned read-only) must not stop the sensor.
        print(f"  Couldn't trim {SAVE_DIR} ({e.__class__.__name__})")


def handle_motion() -> None:
    """Capture one photo and try to upload it. Never raises for expected problems."""
    path = None
    try:
        path = capture_photo()
        print(f"  Captured {path.name} ({path.stat().st_size // 1024} KB)")
        if upload_photo(path):
            path.unlink(missing_ok=True)   # the Mac has it now
            print("  Uploaded to Mac")
        else:
            keep_for_later(path)
            print("  Kept on SD card for later")
    except subprocess.CalledProcessError as e:
        # errors="replace": never crash on odd bytes in the camera's error output
        print(f"  Capture failed: {e.stderr.decode(errors='replace')[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  Capture failed: camera didn't respond in {CAMERA_TIMEOUT}s")
    except requests.RequestException as e:
        # The Mac is asleep, off, or unreachable. Keep the photo and carry on.
        # Only the error type is shown; str(e) would include the full URL.
        print(f"  Upload failed ({e.__class__.__name__}); kept on SD card for later")
        if path is not None and path.exists():
            keep_for_later(path)


def main() -> None:
    if not MAC_SERVER_URL or not UPLOAD_TOKEN:
        raise SystemExit(f"MAC_SERVER_URL and UPLOAD_TOKEN must be set in {REPO_ROOT / '.env'}")
    TEMP_DIR.mkdir(exist_ok=True)
    SAVE_DIR.mkdir(exist_ok=True)
    trim_saved_photos()   # clean up anything left over from before a crash or power cut
    pir = MotionSensor(PIR_PIN)

    print("Warming up the sensor (30s)...")
    time.sleep(30)
    # Show only the host, not the full URL: logs get copied around.
    print(f"Ready. Uploading to {urlsplit(MAC_SERVER_URL).hostname}. Waiting for motion. Ctrl+C to stop.")

    while True:
        pir.wait_for_motion()
        print(f"{datetime.now():%H:%M:%S}  Motion detected, capturing...")
        try:
            handle_motion()
        except Exception as e:
            # Last line of defence. handle_motion() already handles the
            # problems we expect; this catches the ones we didn't (camera tool
            # missing, SD card full, ...). Log it and keep watching, because
            # a crashed script would mean a silently dead security system.
            print(f"  Unexpected error ({e.__class__.__name__}: {str(e)[:100]}); still watching")
        time.sleep(COOLDOWN_SECONDS)   # cooldown prevents a burst of photos
        pir.wait_for_no_motion()


# Only start the sensor loop when run as a script (python3 pi/motion_capture.py),
# not when imported, so the functions above can be tested on their own.
if __name__ == "__main__":
    main()
