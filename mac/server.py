"""Upload server: receives photos from the Pi and analyzes them one at a time.

Run from the repo root (with .env containing UPLOAD_TOKEN):
    .venv/bin/uvicorn --app-dir mac server:app --host 0.0.0.0 --port 8000

How a photo flows through this file:

    Pi --POST /upload (raw JPEG body)--> upload()  --saves file, queues it--> replies 202 at once
                                         |
                                         v
                     worker thread: queue -> analyze() -> handle_result() -> delete files

Why a queue + one worker thread?
    analyze() takes ~30 s (the LLM reloads each time) and must not run twice
    at once (shared YOLO model, only 8 GB RAM). If the web request waited for
    it, the Pi would hang for 30 s per photo. Instead the request only *saves*
    the photo and returns; a single background thread does the slow work in
    order. A queue is a thread-safe "to-do list" between the two.
"""

import asyncio
import logging
import os
import queue
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request

# Importing detect_describe also sets YOLO_AUTOINSTALL=false before ultralytics loads.
from detect_describe import Analysis, analyze, load_detector

# --- Settings -------------------------------------------------------------------

# Read .env from the repo root (one folder up from mac/). load_dotenv() copies
# its KEY=value lines into os.environ. It does NOT override variables that are
# already set, so `UPLOAD_TOKEN=x uvicorn ...` still works for testing.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Photos live outside the repo, so they can never be committed by accident.
DATA_DIR = Path.home() / "motion-alerts"
INCOMING_DIR = DATA_DIR / "incoming"     # uploads waiting to be analyzed
ANNOTATED_DIR = DATA_DIR / "annotated"   # copies with YOLO boxes drawn

MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB; a Pi camera JPEG is ~1-5 MB
MAX_QUEUE = 20                        # if 20 photos are already waiting, refuse more
JPEG_MAGIC = b"\xff\xd8\xff"          # every JPEG file starts with these 3 bytes

# logging is like print(), but every line gets a time and level (INFO/WARNING/ERROR),
# and it's safe to use from several threads at once.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server")
# The httpx library (used by the ollama client) logs every HTTP call at INFO.
# That's noise for us, so only show its warnings and errors.
logging.getLogger("httpx").setLevel(logging.WARNING)

# The shared secret the Pi must send. Fail at startup if it's missing:
# a server that silently accepts everyone is worse than one that won't start.
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN", "")
if len(UPLOAD_TOKEN) < 16:
    raise SystemExit(
        "UPLOAD_TOKEN missing or too short (need 16+ chars). Generate one with:\n"
        "  python3 -c 'import secrets; print(secrets.token_urlsafe(32))'\n"
        "and put UPLOAD_TOKEN=<that value> in .env"
    )

# The to-do list between the web handler and the worker thread.
jobs: queue.Queue[Path | None] = queue.Queue(maxsize=MAX_QUEUE)


# --- What to do with a finished analysis ----------------------------------------------
def handle_result(photo: Path, result: Analysis) -> None:
    """Act on one analysis. For now: log it.

    The next feature (Discord alerts) will send result.annotated_path and
    result.description to Discord here, before the files are deleted.
    """
    if result.person_count == 0:
        log.info("%s: no person, ignoring", photo.name)
        return
    # Count and confidence only. The description ("person in a red jacket at
    # the door") is personal data: logs stick around long after the photos are
    # deleted, so it's only shown when debugging (--log-level debug).
    log.info("%s: PERSON x%d (conf %.2f)", photo.name, result.person_count, result.top_confidence)
    log.debug("%s: description: %s", photo.name, result.description)


def annotated_path_for(photo: Path) -> Path:
    """Where analyze() writes the boxed copy of `photo` (same naming rule)."""
    return ANNOTATED_DIR / f"{photo.stem}_annotated.jpg"


def delete_photo_files(photo: Path) -> None:
    """Remove an upload and its annotated copy, if they exist."""
    photo.unlink(missing_ok=True)
    annotated_path_for(photo).unlink(missing_ok=True)


# --- The worker thread -------------------------------------------------------------------
def worker(model) -> None:
    """Runs forever in the background, processing one photo at a time."""
    while True:
        photo = jobs.get()          # blocks (sleeps, using no CPU) until a job arrives
        if photo is None:           # None is our "please stop" signal at shutdown
            break
        try:
            result = analyze(model, photo, out_dir=ANNOTATED_DIR)
            handle_result(photo, result)
        except ValueError as e:
            # An *expected* problem (a corrupt or unreadable photo): one clear
            # line is enough, a 30-line traceback would just bury real bugs.
            log.warning("%s: skipped, %s", photo.name, e)
        except Exception:
            # Catch *everything* here: if one bad photo crashed this thread,
            # every later photo would sit in the queue forever, unprocessed.
            # log.exception also prints the full traceback for debugging.
            log.exception("%s: processing failed", photo.name)
        finally:
            # `finally` runs whether the try succeeded or failed, so photos
            # never pile up on disk (privacy + disk space). We compute the
            # annotated path ourselves instead of reading it from `result`,
            # because if analyze() failed halfway there is no `result`, but
            # the annotated file may already have been written.
            delete_photo_files(photo)
            jobs.task_done()


# --- Startup and shutdown ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI runs the code before `yield` at startup and after it at shutdown.

    Startup is the right time for slow, one-time work: loading YOLO (~1 s)
    once here instead of once per photo.
    """
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    # Leftovers mean the last run crashed or was killed mid-job. Nobody will
    # ever process them, so delete them (privacy: no orphaned photos).
    for leftover in [*INCOMING_DIR.glob("*.jpg"), *ANNOTATED_DIR.glob("*.jpg")]:
        leftover.unlink(missing_ok=True)
    model = load_detector()
    # daemon=True: if the server is killed, don't let this thread keep Python alive.
    thread = threading.Thread(target=worker, args=(model,), daemon=True, name="worker")
    thread.start()
    log.info("Ready. Saving uploads to %s", INCOMING_DIR)
    yield
    # Shutdown. First empty the queue: photos still waiting won't be processed,
    # so delete them now. This also guarantees there's room for the stop
    # signal (put() on a full queue would wait forever).
    while True:
        try:
            waiting = jobs.get_nowait()
        except queue.Empty:
            break
        if waiting is not None:
            delete_photo_files(waiting)
        jobs.task_done()
    jobs.put(None)             # ask the worker to stop after its current photo
    thread.join(timeout=150)   # one photo can take up to ~2 min (Ollama timeout)


app = FastAPI(title="motion-alert receiver", lifespan=lifespan)


# --- HTTP endpoints ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """A cheap "are you alive?" check. No token needed; reveals nothing private.
    Handy from the Pi: curl http://<mac-ip>:8000/health"""
    return {"ok": True, "queued": jobs.qsize()}


@app.post("/upload", status_code=202)
async def upload(
    request: Request,
    authorization: str = Header(default=""),
    content_length: int | None = Header(default=None),
) -> dict:
    """Accept one JPEG from the Pi. Replies 202 Accepted = "got it, will process later".

    The photo is the raw request body (Content-Type: image/jpeg), NOT a
    multipart form. Why it matters: if this function declared an `UploadFile`
    parameter, FastAPI would receive and store the *entire* body before
    calling us, so a stranger with no token could stream gigabytes onto the
    disk and only then get "401". By taking the plain `Request`, nothing of
    the body is read until we decide to read it, so we check, in order:
    token -> declared size -> real size while streaming -> JPEG bytes.

    `Header(...)` parameters read HTTP headers ("content_length" matches the
    "Content-Length" header; FastAPI converts underscores to dashes).
    """
    # 1. Who's asking? compare_digest takes the same time whether the first
    #    or the last character is wrong, so an attacker can't guess the token
    #    one character at a time by measuring response times ("timing attack").
    if not secrets.compare_digest(authorization.encode(), f"Bearer {UPLOAD_TOKEN}".encode()):
        raise HTTPException(status_code=401, detail="bad or missing token")

    # 2. Busy? Cheap early exit before receiving any photo data.
    if jobs.full():
        raise HTTPException(status_code=503, detail="busy, try later")

    # 3. Size, part 1: the client must say how big the body is, and it must be
    #    under the limit. Honest clients (like `requests` on the Pi) always send it.
    if content_length is None:
        raise HTTPException(status_code=411, detail="Content-Length required")
    if content_length > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")

    # 4. Size, part 2: a dishonest client could lie in Content-Length, so we
    #    also count the real bytes as they arrive, in chunks, and stop the
    #    moment the limit is passed. At most ~10 MB is ever held in memory.
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large")

    if not data.startswith(JPEG_MAGIC):
        # Don't trust the filename or Content-Type the client claims; check the bytes.
        raise HTTPException(status_code=415, detail="not a JPEG")

    # 5. Save under a name WE choose: timestamp (readable, sortable) + random
    #    suffix (unique even for two uploads in the same second). Never use a
    #    client-supplied name: "../../.ssh/authorized_keys" is a valid filename too.
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
    path = INCOMING_DIR / name

    # Writing a file is "blocking" work. This function is `async`, meaning it
    # shares one event loop with every other request; if it blocked, the whole
    # server would freeze during the write. asyncio.to_thread runs the write in
    # a helper thread and lets other requests continue meanwhile.
    await asyncio.to_thread(path.write_bytes, bytes(data))

    # 6. Queue it. put_nowait fails immediately if the queue filled up while we
    #    were receiving, so a flood of uploads gets "503 busy", not a full disk.
    try:
        jobs.put_nowait(path)
    except queue.Full:                     # rare: filled up between check and put
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="busy, try later")

    log.info("Queued %s (%d KB, %d waiting)", name, len(data) // 1024, jobs.qsize())
    return {"queued": name}
