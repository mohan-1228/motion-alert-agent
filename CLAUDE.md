# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A local motion-alert agent, split across two machines:

```
Raspberry Pi (PIR sensor + camera) -> Mac Mini (person detection + local LLM agent) -> Discord
```

- `pi/` runs on the Raspberry Pi. It has hardware dependencies (`gpiozero`, the `rpicam-still` CLI) and cannot run on the Mac.
- `mac/` will hold the Mac Mini side (person detection, local LLM agent, Discord alerts). It is empty for now.
- `docs/` holds diagrams and notes.

Status: early development, and a learning project. Only the Pi capture side exists so far. Mac dependencies are pinned in `mac/requirements.txt`. There is no build system, linter, or test suite yet.

## Chosen stack (Mac side, planned)

Everything runs locally on the Mac Mini: **Apple M2, 8 GB unified RAM**. YOLO and the vision LLM share that RAM, so prefer small models and free memory between events. Motion events are rare (a few per hour), so simplicity matters more than speed.

Pipeline for one event:

1. **Pi → Mac upload.** After `capture_photo()`, the Pi POSTs the JPEG as the **raw body** (`data=f`, `Content-Type: image/jpeg`; not multipart) to `MAC_SERVER_URL` with `Authorization: Bearer <UPLOAD_TOKEN>`, using `requests` with `timeout=(3, 15)`. Catch `requests.RequestException` separately from `CalledProcessError` so a Mac outage doesn't kill the sensor loop.
2. **Receiver: FastAPI + uvicorn** (`mac/server.py`, endpoint `POST /upload`). Check the token with `secrets.compare_digest`. Run with `uvicorn server:app --host 0.0.0.0 --port 8000`, which is LAN-only, so never port-forward it.
3. **Person detection: Ultralytics YOLO26 nano** (`yolo26n.pt`). Load the model once at server startup, not per request. Call `model.predict(path, classes=[0], conf=0.4, device="mps")`, where class 0 is person. If MPS raises an op error, fall back to `device="cpu"`. Skip CoreML export because it isn't needed at this event rate. If no person is found, stop here: no LLM call and no alert.
4. **Description: Ollama vision model `qwen2.5vl:3b`** (about 3.2 GB). Fallbacks are `gemma3:4b` (3.4 GB) and `moondream` (1.7 GB). Use `ollama.chat(...)` with `images=[path]`, `keep_alive=0` to unload after each event, `options={"temperature": 0.2, "num_predict": 80, "num_ctx": 2048}`, and downscale the image to about 1024 px on the long side first. The installed `qwen3:4b` is text-only and can't read images.
5. **Alert: Discord Execute Webhook.** POST multipart to `DISCORD_WEBHOOK_URL?wait=true` with `files[0]` (the annotated JPEG) and `payload_json` (an embed whose `image.url` is `attachment://<same filename>`). On a 429 response, sleep `retry_after` seconds and retry once or twice. Keep images well under 8 MB.

Mac dependencies: `ultralytics`, `fastapi`, `uvicorn`, `ollama`, `requests`, `python-dotenv`.

Licensing: Ultralytics is AGPL-3.0. That's fine for a personal, local project, but if the repo is published it must stay AGPL-compatible.

## Running (on the Mac)

```bash
.venv/bin/python mac/detect_describe.py PHOTO.jpg   # YOLO person check, then LLM description if a person is found
```

- `mac/detect_describe.py` holds the detection and description logic. `analyze(model, path)` is the function the server will reuse. Load YOLO once with `load_detector()` and pass the model in.
- It writes `<photo>_annotated.jpg` (with boxes drawn) to `out_dir`, which defaults to the photo's own folder. When testing, use copies outside the repo.
- If the description fails (Ollama is down, the model isn't pulled, or the call times out), `description` is `None` and the person count is still returned. A failed description must never block an alert.
- `YOLO_AUTOINSTALL=false` is set before `import ultralytics`, so Ultralytics can't pip-install extras at runtime (it tried to add `pi-heif`). Keep it in any new entry point that imports ultralytics, and add needed packages to `requirements.txt` instead.
- `analyze()` is not re-entrant. The server must process one photo at a time and give uploads unique names.
- `clean_description()` trims answers that were cut off at the token cap back to the last full sentence, because the small model doesn't reliably obey length limits. Descriptions can also hallucinate small details. Treat them as context; YOLO's person count is what decides whether to alert.
- With `keep_alive=0`, each description takes about 30 s, mostly reloading the model into RAM. This is a deliberate choice for an 8 GB Mac.

### Upload server (`mac/server.py`)

```bash
.venv/bin/uvicorn --app-dir mac server:app --host 0.0.0.0 --port 8000   # needs UPLOAD_TOKEN (16+ chars) in .env
curl http://localhost:8000/health                                       # {"ok": true, "queued": N}
```

- `POST /upload` takes the **raw JPEG as the request body**, plus the header `Authorization: Bearer <UPLOAD_TOKEN>`. It replies **202** straight away. Errors: 401 bad token, 503 queue full (20), 411 no Content-Length, 413 over 10 MB, 415 not a JPEG (magic bytes are checked).
- Don't switch to `UploadFile` or multipart. FastAPI would receive the whole body *before* the token check, so an unauthenticated client could fill the disk. The handler checks the token first, then streams the body with a byte counter.
- Plain HTTP: the token and the photos cross the LAN unencrypted. That's acceptable on a trusted home network. Never port-forward 8000 to the internet.
- Descriptions are logged only at DEBUG level, because logs outlive the deleted photos. On startup, leftover photos from a crash are deleted. On shutdown, queued photos are dropped and deleted.
- Uploads are saved under server-chosen names (`YYYYMMDD_HHMMSS_<rand>.jpg`) in `~/motion-alerts/incoming/`, outside the repo. The client's filename is never used.
- A single worker thread takes jobs from a `queue.Queue` and runs `analyze()` one at a time, because `analyze()` isn't re-entrant. Then it calls `handle_result()` (currently it only logs), and deletes the upload and the annotated copy in a `finally` block.
- The worker must never die. `ValueError` (unusable image) logs one warning line; any other exception logs a traceback, and the loop continues.
- `handle_result()` is where Discord alerting plugs in.

## Mac setup

```bash
python3.13 -m venv .venv && source .venv/bin/activate   # default python3 is 3.15; Homebrew 3.13 avoids wheel surprises
pip install -r mac/requirements.txt   # pinned versions
ollama pull qwen2.5vl:3b
```

Verified on 2026-11-25: torch 2.15.0 sees MPS. `mac/yolo26n.pt` (auto-downloaded there on first use, gitignored) found 4 persons in Ultralytics' sample `bus.jpg`, and `qwen2.5vl:3b` described them correctly.

## Running (on the Pi)

```bash
python3 pi/pir_test.py        # check PIR wiring: prints motion start/stop events
python3 pi/motion_capture.py  # 30s sensor warm-up, then saves a photo on each motion event
```

- The PIR sensor is on GPIO 17 (`PIR_PIN`). This constant is duplicated in both scripts, so change it in both places.
- `motion_capture.py` saves `motion_YYYYMMDD_HHMMSS.jpg` to `~/captures/` by calling `rpicam-still`. After each capture it waits `COOLDOWN_SECONDS`, then waits for the sensor to go idle before re-arming. This keeps one motion event from producing a burst of photos.

## Configuration and privacy

- Secrets go in `.env` (template: `.env.example`): `DISCORD_WEBHOOK_URL`, `MAC_SERVER_URL` (where the Pi will send captures), and `UPLOAD_TOKEN` (auth for those uploads). Nothing reads these yet. They are for the planned Pi→Mac upload and Discord alerting. Load them with `python-dotenv`, and never log their values.
- `.gitignore` excludes every image (`*.jpg`, `*.png`, `captures/`), along with `*.db` and `*.pt` model weights, for privacy and size reasons. Don't commit photos or model files.
- Python virtualenvs are expected at `agent-env/` or `.venv/`.
