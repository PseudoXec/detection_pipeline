# Using the OCR service

Quickstart for the two ways to run this project. For the full reference
(every config option, de-dup/forwarding behaviour, etc.) see `README.md`.

## 1. Install

```bash
cd OCR
pip install -r requirements.txt
```

## 2. Run it

**As a live HTTP service** (the normal day-to-day mode — the edge/Pi
pipeline posts to this):

```bash
python server/server.py
```

Runs directly as a plain script from anywhere — no `-m`, no special
working directory needed. In production, run with gunicorn instead, from
the `OCR/` project root:

```bash
gunicorn -w 2 -b 0.0.0.0:8500 server.server:app
```

**Or, without a server at all** — OCR a folder of saved crops locally:

```bash
python -m cli.ocr_pipeline --folder plate_crops/ --recursive
```

Results print to the console and are written to `output/ocr_results.csv`.

## 3. Check it's alive

```bash
curl http://<server-ip>:8500/health
```

```json
{"status": "ok", "model": "cct-xs-v1-global-model"}
```

## 4. Send it a plate crop

From the edge device, once it has a plate crop:

```bash
curl -X POST http://<server-ip>:8500/ocr/read \
  -H "X-API-Key: <token>" \
  -F "plate_crop=@plate.jpg;type=image/jpeg" \
  -F "track_id=abc123" \
  -F "camera_source=cam1" \
  -F "plate_detected=true" \
  -F "detected_at=2026-09-26T10:15:00Z"
```

- `plate_crop` and `X-API-Key` are the only two fields this service
  actually reads for itself — send whatever other metadata fields you
  want passed straight through to the response/dashboard (`track_id`,
  `camera_source`, box coordinates, timestamps, etc.).
- No plate detected? Send `plate_detected=false` and skip the
  `plate_crop` file entirely (plain form fields, no multipart needed).
- `X-API-Key` is only required if `server.auth_token` is set in
  `config/ocr_config.yaml`.

## 5. Read the response

```json
{
  "track_id": "abc123",
  "camera_source": "cam1",
  "plate_detected": "true",
  "detected_at": "2026-09-26T10:15:00Z",
  "ocr_process": true,
  "ocr_read": "ABC1234",
  "ocr_text": "ABC1234"
}
```

| Field | Meaning |
|---|---|
| `ocr_process` | `true` if an OCR attempt was made at all. |
| `ocr_read` | The final plate string, or `"No Plate Detected"` (never had a plate) / `"Unrecognized"` (had a plate, couldn't get a confident, correctly-formatted read from it). |
| `ocr_text` | The raw text the model actually produced, before cleanup and before the Philippine-plate-format check — kept separate so a bad read is still visible for review even when `ocr_read` says "Unrecognized". |

Every other field you sent comes back unchanged, alongside these three.

- **200** — request handled; if `server.forward_url` is set, this also
  means it was successfully relayed to the dashboard.
- **502** — the OCR read itself succeeded (still in the response body),
  but relaying to `forward_url` failed. Retry the send.
- **400** — malformed request (bad JSON, invalid base64, etc.).
- **401** — missing/wrong `X-API-Key`.

## 6. Tune it (optional)

Edit `config/ocr_config.yaml` — you only need to list the keys you want to
change from the defaults in `config/ocr_config.py`. Common ones:

- `server.auth_token` — require an `X-API-Key` on every request.
- `server.forward_url` — auto-push completed records to the dashboard
  instead of only returning them.
- `ocr.min_confidence` — reads below this become `"Unrecognized"`.
- `ocr.validate_plate_format` — set to `false` to skip the Philippine
  plate-format check (only needed outside the Philippines).
