"""Detect people in one photo with YOLO, then describe them with a local vision LLM.

Usage (from the repo root, with the venv active):
    python mac/detect_describe.py path/to/photo.jpg

Why this file exists:
    This is the "brain" of the Mac side, kept separate from the web server
    so it can be tested on any photo without the Pi or the network. Later,
    mac/server.py will import `analyze()` and call it for every upload.

Two-stage design (why YOLO *and* an LLM?):
    1. YOLO is small (5 MB) and fast (milliseconds). It answers only
       "is there a person, and where?" -- a cheap filter.
    2. The vision LLM is big (3.2 GB) and slow (seconds), but it can say
       *what* it sees in words. We only run it when YOLO found a person,
       so cats, shadows and wind never wake up the expensive model.

Golden rule: a failure in stage 2 (the description) must NEVER hide a
stage-1 result (a person was seen). The alert still goes out, just without text.
"""

import io
import os
import sys
from dataclasses import dataclass

# Stop Ultralytics from pip-installing packages by itself at runtime. By default,
# when an image fails to open it tries to install `pi-heif` (iPhone HEIC support);
# other code paths install other extras. That would make .venv drift away from
# mac/requirements.txt and make a server download code unexpectedly. This must
# run BEFORE `import ultralytics`, because Ultralytics reads it once at import.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
from pathlib import Path

import httpx                        # HTTP library the ollama client uses under the hood (for its error types)
import ollama                       # Python client for the local Ollama server
from PIL import Image, UnidentifiedImageError   # Pillow: image loading/resizing (installed with ultralytics)
from ultralytics import YOLO        # YOLO object detector

# --- Settings ---------------------------------------------------------------
# Kept at the top as constants so they're easy to find and tweak while learning.

# Where the YOLO weights live: mac/yolo26n.pt. Ultralytics downloads the file
# here on first use. *.pt files are gitignored (too big / not source code).
YOLO_WEIGHTS = Path(__file__).parent / "yolo26n.pt"   # "n" = nano, the smallest/fastest size

PERSON_CLASS = 0        # YOLO is trained on the COCO dataset, where class 0 is "person"
MIN_CONFIDENCE = 0.4    # ignore detections YOLO is less than 40% sure about (fewer false alarms)

VISION_MODEL = "qwen2.5vl:3b"   # Ollama vision model (must be pulled: `ollama pull qwen2.5vl:3b`)
MAX_IMAGE_SIDE = 1024           # shrink big photos before sending to the LLM (see shrink_for_llm)

PROMPT = (
    "This is a home security camera photo. Describe ONLY the people in at most "
    "2 short sentences: their clothing, where they are, and what they are doing. "
    "Do not describe vehicles, buildings, signs or the background. "
    "Do not guess identities."
)
# Hard cap on answer length, in tokens (~0.75 words each). This is a safety
# net, not the main length control: if the model hits it mid-sentence we trim
# back to the last full sentence in clean_description().
MAX_ANSWER_TOKENS = 150

# How long to wait for Ollama before giving up, in seconds. A cold model load
# takes ~30s on this Mac (keep_alive=0 unloads it after every answer), so 120s
# leaves room. Without a timeout a stuck Ollama would freeze us forever.
OLLAMA_TIMEOUT = 120
_ollama = ollama.Client(timeout=OLLAMA_TIMEOUT)   # talks to http://localhost:11434 by default

# Which device YOLO runs on. Starts as "mps" (the M2 GPU). If MPS ever fails
# with a GPU error we switch to "cpu" *and remember it*, so we don't waste
# time failing on MPS again for every photo.
_device = "mps"


# --- Result container ---------------------------------------------------------
# A dataclass is a tiny class that just holds named fields. It's clearer than
# returning a tuple like (3, 0.87, "...", path) where you must remember the order.
@dataclass
class Analysis:
    person_count: int             # how many people YOLO found
    top_confidence: float         # YOLO's best score (0..1), 0 if nobody found
    description: str | None       # LLM text, or None if skipped or the LLM failed
    annotated_path: Path | None   # copy of the photo with boxes drawn, if people found


# --- Input check ------------------------------------------------------------------
def check_image(image_path: Path) -> None:
    """Fail early, with one clear message, if the file isn't a usable image.

    Without this, a corrupt upload would fail deep inside YOLO with a confusing
    error. Pillow's verify() reads the file's structure without decoding all
    pixels, so it's cheap. Pillow also refuses absurdly large images
    ("decompression bombs": tiny files that expand to gigabytes in RAM).

    ImportError is caught too because of an Ultralytics quirk: importing it
    replaces Image.open with a version that, on any unreadable file, tries to
    load the optional `pi_heif` package (iPhone HEIC support). We don't install
    it (and disabled auto-install above), so an unreadable file raises
    ModuleNotFoundError instead of UnidentifiedImageError.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ImportError) as e:
        raise ValueError(f"not a usable image: {image_path.name} ({e.__class__.__name__})") from e


# --- Stage 1: YOLO ---------------------------------------------------------------
def load_detector() -> YOLO:
    """Load YOLO once. Loading takes ~1s, so a server should do it at startup,
    not on every photo."""
    return YOLO(str(YOLO_WEIGHTS))


def detect_people(model: YOLO, image_path: Path):
    """Run YOLO on one image and return its result object.

    - classes=[PERSON_CLASS]: only look for people, ignore the other 79 COCO classes.
    - conf=MIN_CONFIDENCE: drop weak guesses.
    - device: "mps" = the M2's GPU (Apple's "Metal Performance Shaders"), or "cpu".
    - verbose=False: don't print a log line for every prediction.
    """
    global _device   # we may change the module-level setting, so declare it
    predict = lambda device: model.predict(   # noqa: E731 -- small local helper
        str(image_path), classes=[PERSON_CLASS], conf=MIN_CONFIDENCE,
        device=device, verbose=False,
    )
    try:
        results = predict(_device)
    except (RuntimeError, NotImplementedError) as e:
        # Only GPU-type errors land here (PyTorch raises these when an MPS
        # operation is missing or GPU memory runs out). Bad images were
        # already rejected by check_image(), so they can't be mistaken for this.
        if _device == "cpu":
            raise                      # already on CPU: a real error, don't hide it
        print(f"YOLO on MPS failed ({e.__class__.__name__}); using CPU from now on", file=sys.stderr)
        _device = "cpu"
        results = predict(_device)
    return results[0]   # predict() returns a list (one result per image); we passed one image


# --- Stage 2: vision LLM ------------------------------------------------------------
def shrink_for_llm(image_path: Path) -> bytes:
    """Resize so the longest side is at most MAX_IMAGE_SIDE, return JPEG bytes.

    Why: vision LLMs cut images into small tiles ("vision tokens"). A full-size
    camera photo makes many tiles -> slower answers and more RAM. 1024 px still
    shows clothing and location clearly.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")                          # JPEG can't store transparency
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))   # keeps aspect ratio; never enlarges
        buf = io.BytesIO()                                # an in-memory "file"
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def describe(image_path: Path) -> str | None:
    """Ask the local vision model to describe the photo. Returns None on failure.

    Options explained:
    - keep_alive=0: unload the model from RAM right after answering. Alerts are
      rare and the Mac has only 8 GB, so freeing ~3 GB matters more than the
      few seconds it takes to reload next time.
    - temperature=0.2: low randomness -> factual, consistent descriptions.
    - num_predict: hard cap on answer length (see MAX_ANSWER_TOKENS).
    - num_ctx=2048: small context window -> less memory used.

    Errors we expect and survive (the caller still gets the person count):
    - ConnectionError: Ollama isn't running (start it: `ollama serve` or the app).
    - ollama.ResponseError: Ollama answered with an error, e.g. model not pulled.
    - httpx.HTTPError: network-level problems, including our timeout.
    """
    try:
        response = _ollama.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": PROMPT, "images": [shrink_for_llm(image_path)]}],
            keep_alive=0,
            options={"temperature": 0.2, "num_predict": MAX_ANSWER_TOKENS, "num_ctx": 2048},
        )
    except (ConnectionError, ollama.ResponseError, httpx.HTTPError) as e:
        print(f"Description unavailable: {e.__class__.__name__}: {e}", file=sys.stderr)
        return None
    return clean_description(response.message.content)


def clean_description(text: str | None) -> str | None:
    """Tidy the model's answer. Returns None if there's nothing useful.

    Small models don't always obey "2 sentences". If the answer hit the token
    cap, it ends mid-word ("The background includes a"), which looks broken
    in an alert. If there's at least one full sentence, keep up to its end.
    """
    text = (text or "").strip()        # treat a missing answer like an empty one
    if not text:
        return None
    if text.endswith((".", "!", "?")):
        return text                                   # already ends cleanly
    last_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    return text[: last_end + 1] if last_end > 0 else text   # no full sentence? keep it all


# --- Both stages together -------------------------------------------------------------
def analyze(model: YOLO, image_path: Path, out_dir: Path | None = None,
            want_description: bool = True) -> Analysis:
    """Full pipeline for one photo. This is what the server will call.

    out_dir: where to save the annotated copy. Defaults to the photo's own
    folder. The server should pass its own folder and give uploads unique
    names, so two photos can never overwrite each other's output.

    want_description: set False to skip the LLM (stage 2) even if a person is
    found. The server does this during its alert cooldown: no alert will be
    sent, so spending ~30 s and ~3 GB of RAM on a description would be waste.

    Not safe to run twice at the same time: the YOLO model is shared and two
    LLM loads at once would strain 8 GB of RAM. The server must process one
    photo at a time (a queue or a lock). Raises ValueError for unusable images.
    """
    check_image(image_path)
    result = detect_people(model, image_path)
    count = len(result.boxes)   # one box per detected person

    if count == 0:
        # Nobody there: skip the expensive LLM entirely.
        return Analysis(person_count=0, top_confidence=0.0, description=None, annotated_path=None)

    top_conf = float(result.boxes.conf.max())   # .conf is a tensor of scores; take the best

    # Save a copy with boxes drawn, for the Discord alert. *.jpg is gitignored
    # so photos never reach GitHub.
    out_dir = out_dir or image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = out_dir / f"{image_path.stem}_annotated.jpg"
    result.save(filename=str(annotated))

    return Analysis(
        person_count=count,
        top_confidence=top_conf,
        # Describe the clean photo, not the one with boxes drawn on it.
        description=describe(image_path) if want_description else None,
        annotated_path=annotated,
    )


# --- Command-line entry point --------------------------------------------------------
# This block only runs when the file is executed directly (python mac/detect_describe.py),
# not when another file imports it. That lets the server reuse analyze() later.
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python mac/detect_describe.py PHOTO.jpg")

    photo = Path(sys.argv[1])
    if not photo.is_file():
        sys.exit(f"not a file: {photo}")

    try:
        analysis = analyze(load_detector(), photo)
    except ValueError as e:
        sys.exit(str(e))

    if analysis.person_count == 0:
        print("No person detected -- no alert would be sent.")
    else:
        print(f"Persons: {analysis.person_count} (best confidence {analysis.top_confidence:.2f})")
        print(f"Description: {analysis.description or '(unavailable)'}")
        print(f"Annotated image: {analysis.annotated_path}")
