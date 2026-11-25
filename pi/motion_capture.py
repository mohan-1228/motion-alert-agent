"""Take a photo whenever the PIR sensor sees motion, and upload it to the Mac.

Run on the Pi from the repo folder:
    python3 pi/motion_capture.py

Needs a .env file in the repo root with:
    MAC_SERVER_URL=http://<your-mac-name>.local:8000/upload
    UPLOAD_TOKEN=<same value as on the Mac>

Flow for one motion event:
    motion -> rpicam-still saves a JPEG -> POST it to the Mac
      - upload OK (202): delete the local copy (the Mac has it; SD cards
        shouldn't fill up with private photos)
      - upload failed:   keep it in ~/captures, so nothing is lost; the folder
        is capped at MAX_KEPT_PHOTOS so a long Mac outage can't fill the card
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests
from gpiozero import MotionSensor

# --- Settings ---------------------------------------------------------------
PIR_PIN = 17            # GPIO pin the PIR sensor's OUT wire is on (also in pir_test.py)
COOLDOWN_SECONDS = 10   # ignore new motion for this long after a capture
SAVE_DIR = Path.home() / "captures"
CAMERA_TIMEOUT = 30     # seconds; if rpicam-still hangs, give up instead of freezing forever
UPLOAD_TIMEOUT = (3, 15)  # (connect, read) seconds; the Mac replies 202 in ~2 ms when healthy
MAX_KEPT_PHOTOS = 100   # photos kept locally after failed uploads (~2 MB each -> ~200 MB max)

REPO_ROOT = Path(__file__).resolve().parent.parent   # pi/ -> repo root


def load_env(path: Path) -> None:
    """Minimal .env reader: copies KEY=value lines into os.environ.

    Why not python-dotenv like on the Mac? It isn't installed on the Pi, and
    Debian blocks `pip install` into the system Python (PEP 668). Ten lines
    of our own code are simpler than a virtualenv for one small feature.
    Like python-dotenv, it never overrides variables that are already set.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue                          # skip blanks, comments, junk
        key, value = line.split("=", 1)       # split on the FIRST "=" only (values may contain "=")
        os.environ.setdefault(key.strip(), value.strip())


load_env(REPO_ROOT / ".env")
MAC_SERVER_URL = os.environ.get("MAC_SERVER_URL", "").strip()
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN", "").strip()


def capture_photo() -> Path:
    """Take a photo with a timestamped filename and return its path."""
    filename = SAVE_DIR / f"motion_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    subprocess.run(
        ["rpicam-still", "-n", "-t", "500", "-o", str(filename)],
        check=True,             # raise CalledProcessError if the camera command fails
        capture_output=True,    # keep its chatty output off our screen (but available in errors)
        timeout=CAMERA_TIMEOUT, # raise TimeoutExpired if it hangs
    )
    return filename


def upload_photo(path: Path) -> bool:
    """Send one photo to the Mac. Returns True if the Mac accepted it.

    The photo is the raw request body (not a multipart form): the Mac's
    server checks the token *before* reading any of the body, which only
    works with a raw body. `data=f` streams the open file, so the whole
    photo never has to be loaded into the Pi's memory at once.
    """
    with open(path, "rb") as f:
        response = requests.post(
            MAC_SERVER_URL,
            data=f,
            headers={
                "Authorization": f"Bearer {UPLOAD_TOKEN}",
                "Content-Type": "image/jpeg",
                # requests can't always work out the size of a streamed file,
                # and the Mac requires it (411 otherwise), so we send it ourselves.
                "Content-Length": str(path.stat().st_size),
            },
            timeout=UPLOAD_TIMEOUT,
        )
    if response.status_code == 202:
        return True
    # Show the Mac's short error reason (e.g. "bad or missing token"), never our token.
    print(f"  Mac refused the upload: {response.status_code} {response.text[:100]}")
    return False


def trim_saved_photos() -> None:
    """Delete the oldest kept photos so at most MAX_KEPT_PHOTOS remain.

    Filenames start with a timestamp (motion_YYYYMMDD_HHMMSS), so sorting
    them alphabetically also sorts them oldest-first.
    """
    photos = sorted(SAVE_DIR.glob("motion_*.jpg"))
    for old in photos[:-MAX_KEPT_PHOTOS]:     # everything except the newest N
        old.unlink(missing_ok=True)


def handle_motion() -> None:
    """Capture one photo and try to upload it. Never raises for expected problems."""
    try:
        path = capture_photo()
        print(f"  Saved {path.name} ({path.stat().st_size // 1024} KB)")
        if upload_photo(path):
            path.unlink()   # the Mac has it now
            print("  Uploaded to Mac, local copy deleted")
        else:
            trim_saved_photos()
            print("  Kept locally")
    except subprocess.CalledProcessError as e:
        print(f"  Capture failed: {e.stderr.decode()[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  Capture failed: camera didn't respond in {CAMERA_TIMEOUT}s")
    except requests.RequestException as e:
        # The Mac is asleep, off, or unreachable. Keep the photo and carry on:
        # a network problem must never stop the sensor loop.
        # Only the error type is shown; str(e) would include the full URL.
        print(f"  Upload failed ({e.__class__.__name__}); kept locally")
        trim_saved_photos()


def main() -> None:
    if not MAC_SERVER_URL or not UPLOAD_TOKEN:
        raise SystemExit(f"MAC_SERVER_URL and UPLOAD_TOKEN must be set in {REPO_ROOT / '.env'}")
    SAVE_DIR.mkdir(exist_ok=True)
    pir = MotionSensor(PIR_PIN)

    print("Warming up the sensor (30s)...")
    time.sleep(30)
    print(f"Ready. Uploading to {MAC_SERVER_URL}. Waiting for motion. Ctrl+C to stop.")

    while True:
        pir.wait_for_motion()
        print(f"{datetime.now():%H:%M:%S}  Motion detected, capturing...")
        handle_motion()
        time.sleep(COOLDOWN_SECONDS)   # cooldown prevents a burst of photos
        pir.wait_for_no_motion()


# Only start the sensor loop when run as a script (python3 pi/motion_capture.py),
# not when imported, so the functions above can be tested on their own.
if __name__ == "__main__":
    main()
