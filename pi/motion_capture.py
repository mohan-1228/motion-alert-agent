import subprocess
import time
from datetime import datetime
from pathlib import Path
from gpiozero import MotionSensor

PIR_PIN = 17            # set automatically below to match pir_test.py
COOLDOWN_SECONDS = 10   # ignore new motion for this long after a capture
SAVE_DIR = Path.home() / "captures"

SAVE_DIR.mkdir(exist_ok=True)
pir = MotionSensor(PIR_PIN)


def capture_photo() -> Path:
    """Take a photo with a timestamped filename and return its path."""
    filename = SAVE_DIR / f"motion_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    subprocess.run(
        ["rpicam-still", "-n", "-t", "500", "-o", str(filename)],
        check=True,
        capture_output=True,
    )
    return filename


print("Warming up the sensor (30s)...")
time.sleep(30)
print("Ready. Waiting for motion. Ctrl+C to stop.")

while True:
    pir.wait_for_motion()
    print(f"{datetime.now():%H:%M:%S}  Motion detected, capturing...")
    try:
        path = capture_photo()
        print(f"  Saved {path.name}")
    except subprocess.CalledProcessError as e:
        print(f"  Capture failed: {e.stderr.decode()[:200]}")
    time.sleep(COOLDOWN_SECONDS)   # cooldown prevents a burst of photos
    pir.wait_for_no_motion()
