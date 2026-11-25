"""Send one "person detected" alert, with the annotated photo, to a Discord webhook.

Why a separate file:
    Same idea as detect_describe.py: one job per file, testable on its own.
    server.py decides *whether* to alert; this file only knows *how*.

What a Discord webhook is:
    A secret URL (https://discord.com/api/webhooks/<id>/<token>) that lets
    anyone who has it post messages into one channel, without a bot account.
    The URL itself is the password, so it lives in .env and is NEVER logged.

Message format (Discord's "Execute Webhook" API):
    A multipart/form-data POST with two parts:
      - "files[0]":     the JPEG file itself
      - "payload_json": a JSON string describing the message. We use an
                        "embed" (the card with a coloured bar, title, fields)
                        whose image is "attachment://<filename>", which tells
                        Discord "show the file I attached in this same request".
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("discord")

# --- Settings -------------------------------------------------------------------------

# requests timeouts as (connect, read) in seconds: give up if we can't even
# connect within 5 s, or if Discord goes 15 s without sending anything back.
# Without a timeout, a hung Discord would freeze our single worker forever.
TIMEOUT = (5, 15)

MAX_ATTEMPTS = 3          # first try + up to 2 retries
MAX_RETRY_WAIT = 10.0     # if Discord says "wait 60 s", don't: it would stall the whole queue
SERVER_ERROR_WAIT = 2.0   # pause before retrying a Discord-side 5xx or network error

EMBED_COLOR = 0xE74C3C    # red bar on the left of the card (hex RGB as an int)
MAX_DESCRIPTION = 4000    # Discord rejects embed descriptions over 4096 characters

VALID_PREFIXES = ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")


@dataclass
class SendResult:
    """What happened, so the server can decide how long to pause alerts.

    ok:          Discord accepted the alert.
    retry_after: if Discord rate-limited us for longer than we were willing to
                 wait, how many seconds it asked for. The server should send
                 nothing to Discord until then, or it'll just be refused again.
    """
    ok: bool
    retry_after: float | None = None


def looks_like_webhook(url: str) -> bool:
    """Cheap sanity check at startup, to catch copy-paste mistakes early."""
    return url.startswith(VALID_PREFIXES)


def build_payload(filename: str, person_count: int, confidence: float,
                  description: str | None) -> dict:
    """The JSON describing the message. Kept separate so it's easy to test/tweak."""
    text = (description or "_No description available._")[:MAX_DESCRIPTION]
    return {
        # Never let message text ping anyone. The description comes from an LLM,
        # so we don't fully control it; if it ever produced "@everyone", this
        # stops Discord from turning that into a real notification for the server.
        "allowed_mentions": {"parse": []},
        "embeds": [{
            "title": f"Person detected ({person_count})" if person_count > 1 else "Person detected",
            "description": text,
            "color": EMBED_COLOR,
            "fields": [{"name": "Confidence", "value": f"{confidence:.0%}", "inline": True}],
            # ISO-8601 UTC time; Discord shows it in each viewer's local timezone.
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image": {"url": f"attachment://{filename}"},   # must match the uploaded filename
        }],
    }


def _retry_after(response: requests.Response) -> float:
    """How long Discord asked us to wait after a 429 ("Too Many Requests").

    Discord sends it in the JSON body ("retry_after", seconds, may be a float)
    and in the Retry-After header. Data from the network can be anything
    (missing, text, an HTTP date, NaN, negative), so any value we can't use
    becomes a safe default of 1 s, and the result is never negative.
    """
    for get in (lambda: response.json()["retry_after"],
                lambda: response.headers["Retry-After"]):
        try:
            wait = float(get())
        except (ValueError, KeyError, TypeError):
            continue
        if math.isfinite(wait):           # rejects NaN and infinity
            return max(0.0, wait)
    return 1.0


def _error_summary(response: requests.Response) -> str:
    """A short, safe description of a Discord error response.

    Only Discord's numeric error code and message, never the raw body: a 400
    response can quote back parts of what we sent, including the description
    (personal data).
    """
    try:
        body = response.json()
        return f"code {body.get('code')}: {str(body.get('message'))[:100]}"
    except (ValueError, AttributeError):
        return "(no details)"


def send_alert(webhook_url: str, image_path: Path | None, person_count: int,
               confidence: float, description: str | None) -> SendResult:
    """Post one alert and report what happened.

    Never raises: the caller is the server's only worker thread, and a Discord
    problem must not stop it from processing the next photo. Worst case it
    takes roughly 3 x 20 s of timeouts plus short pauses.
    """
    if image_path is None or not image_path.is_file():
        log.error("no annotated image to send")
        return SendResult(ok=False)

    filename = image_path.name
    payload = json.dumps(build_payload(filename, person_count, confidence, description))

    for attempt in range(1, MAX_ATTEMPTS + 1):
        last_try = attempt == MAX_ATTEMPTS
        try:
            # The file is reopened each attempt: after one upload its read
            # position is at the end, so a retry would otherwise send 0 bytes.
            with open(image_path, "rb") as f:
                response = requests.post(
                    webhook_url,
                    # params= lets requests add "?wait=true" correctly even if
                    # the URL already has a "?..." part. wait=true makes Discord
                    # reply 200 + the created message instead of an empty 204.
                    params={"wait": "true"},
                    data={"payload_json": payload},
                    files={"files[0]": (filename, f, "image/jpeg")},
                    timeout=TIMEOUT,
                )
        except (requests.RequestException, OSError) as e:
            # Log only the error *type*. str(e) for connection errors usually
            # contains the full URL, i.e. the webhook secret.
            log.warning("attempt %d/%d: network error (%s)", attempt, MAX_ATTEMPTS, e.__class__.__name__)
            if not last_try:
                time.sleep(SERVER_ERROR_WAIT)
            continue

        if response.ok:                      # any 2xx status
            log.info("alert sent (%s)", filename)
            return SendResult(ok=True)

        if response.status_code == 429:      # rate limited: Discord says "slow down"
            wait = _retry_after(response)
            if wait > MAX_RETRY_WAIT or last_try:
                log.warning("rate limited for %.0fs; dropping this alert", wait)
                return SendResult(ok=False, retry_after=wait)
            log.warning("rate limited; waiting %.1fs (attempt %d/%d)", wait, attempt, MAX_ATTEMPTS)
            time.sleep(wait)
            continue

        if response.status_code >= 500:      # Discord's own problem: often temporary
            log.warning("Discord error %d (attempt %d/%d)", response.status_code, attempt, MAX_ATTEMPTS)
            if not last_try:
                time.sleep(SERVER_ERROR_WAIT)
            continue

        # Any other 4xx is OUR problem (deleted webhook = 404, bad payload = 400).
        # Retrying would just fail the same way, so stop.
        log.error("Discord rejected the alert: %d %s", response.status_code, _error_summary(response))
        return SendResult(ok=False)

    log.error("giving up after %d attempts", MAX_ATTEMPTS)
    return SendResult(ok=False)
