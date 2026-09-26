# Plate OCR Pipeline (OCR-only, FastPlateOCR)

A standalone pipeline that does **only** OCR — no vehicle detection, no
plate detection, no tracking. It reads plate text with
[FastPlateOCR](https://github.com/ankandrew/fast-plate-ocr), and nothing
else, so it can be split out from the detection pipeline entirely.

## Layout

```
OCR/
├── config/
│   ├── ocr_config.py      # PipelineConfig dataclasses (defaults)
│   └── ocr_config.yaml    # your overrides - edit this one
├── ocr/
│   ├── ocr_reader.py      # FastPlateOCR wrapper: load model, read, score
│   ├── image_enhance.py   # optional CLAHE + sharpen pass before OCR
│   └── plate_format.py    # validates a read against real PH LTO plate formats
├── server/
│   ├── server.py          # HTTP service (Flask) - the mode you'll run day to day
│   └── dedupe.py          # SQLite-backed track_id de-dup / majority-vote cache
├── cli/
│   └── ocr_pipeline.py    # batch/CLI mode - point it at a folder of images
├── requirements.txt
└── README.md
```

Every module here is imported by its package path (`config.ocr_config`,
`ocr.ocr_reader`, `server.dedupe`, etc.). `server/server.py` adds the
project root to `sys.path` itself at import time, so it can be run directly
(`python server/server.py`, from anywhere) as well as with
`python -m server.server` or gunicorn (`server.server:app`, run from the
`OCR/` project root). `cli/ocr_pipeline.py` does not have that bootstrap
yet, so it still needs to be run as a module (`python -m cli.ocr_pipeline`)
or with `OCR/` on `sys.path`.

Two ways to run it:

| Mode | Module | Use it when... |
|---|---|---|
| HTTP server | `server.server` | The detection pipeline runs elsewhere (e.g. on the Pi) and needs to hand a `plate_crop` blob + metadata over the network for this side to OCR. |
| Batch / CLI | `cli.ocr_pipeline` | You have a single image or a folder of plate crops to OCR locally. |

## Install

```bash
cd OCR
pip install -r requirements.txt
```

## Server mode (`server/server.py`)

This is the "server-sided" half of a split architecture: the edge/detection
pipeline stops running OCR itself, and instead POSTs the plate crop it just
produced — plus whatever detection metadata it already has (`track_id`,
`camera_source`, `plate_detected`, box coordinates, `detected_at`, timing
fields, etc.) — to this service. All of those other fields pass straight
through unchanged; this service's only job is to fill in three fields:

- `ocr_process` — `true` if an OCR attempt was actually made (skipped
  entirely when `plate_detected` is false or no `plate_crop` was sent —
  same "don't bother OCRing a non-plate" rule the original pipeline used)
- `ocr_read` — the cleaned, final plate string, or one of two tags: `"No
  Plate Detected"` (`ocr.no_plate_text`) when the vehicle never had a plate
  to begin with, vs `"Unrecognized"` (`ocr.unrecognized_text`) when a plate
  crop existed but nothing readable came off it. Kept distinct so a reviewer
  scanning the dashboard can tell "nothing to read" apart from "tried and
  failed".
- `ocr_text` — the raw/unfiltered text the model produced *before*
  cleanup, kept separate so a bad clean-up step is still visible for review

**Philippine plate format check (`ocr/plate_format.py`)**: a confident read
is also checked against the plate shapes the LTO actually issues (3
letters + 4 digits for current cars, 3 digits + 3 letters for current
motorcycles/tricycles, plus the older 3+3, 3+2, 2+4 and 2+5 formats). A
read that doesn't match any of them — reversed order, wrong grouping, a
stray leftover character — comes back as `ocr_read: "Unrecognized"` the
same as a low-confidence read, while `ocr_text` still shows exactly what
the model produced, for review. Turn this off with
`ocr.validate_plate_format: false` in `ocr_config.yaml` if this service is
ever used outside the Philippines. Government/diplomatic plates aren't
covered by this check (their formats vary too much for a safe regex), so
those pass through as long as they clear `min_confidence`.

The edge pipeline sends every vehicle now, plate or not — a vehicle with no
plate arrives with `plate_detected=false` and no `plate_crop` file at all
(not a multipart request, just plain form fields), which this service
handles the same way as a multipart one.

Run it:

```bash
python server/server.py
# equivalent, from the OCR/ project root:
python -m server.server
# or, in production, from the OCR/ project root:
gunicorn -w 2 -b 0.0.0.0:8500 server.server:app
```

Send it a request — multipart (mirrors what the edge pipeline's
`api_client.py` already POSTs today):

```bash
curl -X POST http://server:8500/ocr/read \
  -H "X-API-Key: <token>" \
  -F "plate_crop=@plate.jpg;type=image/jpeg" \
  -F "track_id=abc123" \
  -F "camera_source=cam1" \
  -F "plate_detected=true" \
  -F "detected_at=2026-09-26T10:15:00Z"
```

or JSON, with `plate_crop` as base64:

```json
{
  "track_id": "abc123",
  "plate_detected": true,
  "plate_crop": "<base64 JPEG bytes>"
}
```

Response — same metadata back, OCR fields filled in:

```json
{
  "track_id": "abc123",
  "camera_source": "cam1",
  "plate_detected": "true",
  "detected_at": "2026-09-26T10:15:00Z",
  "ocr_process": true,
  "ocr_read": "ABC123",
  "ocr_text": "abc123"
}
```

**Auth**: set `server.auth_token` in `config/ocr_config.yaml` and the edge
side must send it back as an `X-API-Key` header, or requests get a 401.
Leave it `null` only on a network you fully control.

**Downstream forwarding**: by default this service only returns the
completed record in the HTTP response — it does not talk to the C#
dashboard itself, to keep it doing OCR and nothing else. If you'd rather it
also push the completed record on to another endpoint automatically, set
`server.forward_url` in `config/ocr_config.yaml`. Forwarding re-attaches
the plate image (under `server.forward_image_field`, default `Image`) and
adds the OCR text under `server.forward_plate_number_field` (default
`PlateNumber`) so the request matches the dashboard's existing endpoint
shape exactly - query params + a multipart file, the same as the edge
pipeline's original direct-to-dashboard calls. Fields listed in
`server.forward_drop_fields` (default: `plate_detected`) are internal to
the edge<->OCR-only handoff and are stripped before forwarding.

Unlike a purely best-effort relay, a failed forward makes `/ocr/read` itself
return `502` (the OCR result is still included in the body). That failure
is what lets the edge pipeline's own retry logic pick the detection back up
later instead of it being silently dropped if the dashboard is briefly
unreachable. With `forward_url` unset, `/ocr/read` always returns `200` and
whoever called it is responsible for relaying the result onward.

**De-duplication / majority voting (`server/dedupe.py`)**: because a failed
forward makes the edge retry, and a retry can succeed on the FAR side even
when the edge never saw the 200 (dropped connection, timeout, this service
restarting mid-request), the same track_id can arrive here more than once
for what is really the same vehicle. With `server.dedupe_enabled: true`
(the default), this service remembers which track_ids it has already
successfully forwarded for `server.dedupe_window_seconds`, and a repeat
within that window is NOT forwarded again — every sighting still casts a
"vote" for whatever text it read, and the MAJORITY text across all votes is
what gets returned/logged.

Two things worth knowing before relying on this for more than "avoid an
obvious duplicate row":

- **Multi-worker deployments are covered.** The cache is backed by a small
  SQLite file (`server.dedupe_db_path`), not in-process memory, so every
  `gunicorn -w N` worker - and this service across restarts - shares the
  same view. A duplicate landing on a different worker than the original
  is still caught.
- The row already forwarded to the dashboard is never corrected. If a later
  duplicate's read differs and even becomes the majority, that's reflected in
  this service's response/logs but is not re-sent to the dashboard (this
  service only creates rows, it has no way to update one by track_id). In
  practice this is rarely an issue: repeat submissions of the same track_id
  are almost always the exact same image (a network retry of the identical
  request), so the "majority" and the "first" answer are usually identical.

See `server/dedupe.py`'s module docstring for the full reasoning.

### Avoiding duplicate rows for good

This cache is a cheap first line of defense, not a guarantee: the check for
"has this track_id already been forwarded" and the actual forward happen as
two separate steps around a network call, so two near-simultaneous
duplicates can both slip past the check before either finishes forwarding.

The durable fix belongs in the dashboard's own database - it's the system
of record, and a unique constraint there is enforced inside a real
transaction regardless of how many times a detection gets retried or how
many workers/processes are involved on this side. Two ways to do it,
whichever fits your dashboard's stack:

**Option A - unique index + let the insert fail on a repeat** (works
anywhere): add a unique constraint on the column the dashboard stores
`track_id` in, and treat the resulting constraint-violation error on insert
as "already have this one, ignore it" rather than a real failure.

```sql
-- SQL Server
ALTER TABLE Detections ADD CONSTRAINT UQ_Detections_TrackId UNIQUE (TrackId);

-- PostgreSQL / SQLite
CREATE UNIQUE INDEX IF NOT EXISTS ux_detections_track_id ON detections(track_id);

-- MySQL
ALTER TABLE Detections ADD UNIQUE KEY ux_detections_track_id (TrackId);
```

**Option B - upsert instead of insert** (update-in-place instead of
erroring, useful if you'd rather the LATEST submission for a track_id win
outright, e.g. so a corrected OCR read on retry actually reaches the row):

```sql
-- SQL Server
MERGE INTO Detections AS target
USING (VALUES (@TrackId, @PlateNumber, ...)) AS src (TrackId, PlateNumber, ...)
ON target.TrackId = src.TrackId
WHEN MATCHED THEN UPDATE SET PlateNumber = src.PlateNumber, ...
WHEN NOT MATCHED THEN INSERT (TrackId, PlateNumber, ...) VALUES (src.TrackId, src.PlateNumber, ...);

-- PostgreSQL / SQLite
INSERT INTO detections (track_id, plate_number, ...) VALUES (?, ?, ...)
ON CONFLICT (track_id) DO UPDATE SET plate_number = excluded.plate_number, ...;
```

Either one makes duplicate rows structurally impossible on the dashboard
side, independent of this service's cache, its worker count, or its uptime.

## Batch / CLI mode (`cli/ocr_pipeline.py`)

Point it at plate-crop images (single file or a folder) and it reads the
text off each one, no server involved. Run from the `OCR/` project root:

```bash
# single image
python -m cli.ocr_pipeline --image plate.jpg

# a folder of crops
python -m cli.ocr_pipeline --folder plate_crops/ --recursive

# custom model, stricter confidence, also write JSON
python -m cli.ocr_pipeline --folder plate_crops/ \
    --model cct-xs-v1-global-model \
    --min-confidence 0.6 \
    --json output/results.json
```

Results print to the console and are written to `output/ocr_results.csv`
by default. Pass `--no-csv` / `--json path` / `--quiet` to change that, or
edit `config/ocr_config.yaml` for anything more permanent.

## Files

| File | Purpose |
|---|---|
| `ocr/ocr_reader.py` | Wraps FastPlateOCR: loads the model lazily, tries multiple image variants, cleans and scores the text. `read_full()` returns cleaned text, raw text, and confidence together. |
| `ocr/image_enhance.py` | Optional CLAHE + sharpen pass applied before OCR (helps low-contrast/small crops). |
| `ocr/plate_format.py` | Validates a cleaned read against real Philippine LTO plate formats; a non-matching read is reported as unrecognized. |
| `config/ocr_config.py` / `config/ocr_config.yaml` | All tunables (model, thresholds, input/output paths, server settings) — same override pattern as the reference project. |
| `server/server.py` | HTTP server mode: receives `plate_crop` + metadata, fills `ocr_process`/`ocr_read`/`ocr_text`, returns the completed record. |
| `server/dedupe.py` | SQLite-backed track_id de-dup + majority-vote cache used by `server.py` to avoid forwarding the same vehicle twice on a retried send - shared across gunicorn workers. |
| `cli/ocr_pipeline.py` | Batch/CLI mode: discovers images, runs the reader over them, writes CSV/JSON/console output. |

## Config highlights (`config/ocr_config.yaml`)

- `ocr.model_name` — any FastPlateOCR model-zoo name, or a path to your own exported ONNX model.
- `ocr.min_confidence` — reads below this become `"Unrecognized"` instead of a low-quality guess.
- `preprocess.try_both_variants` — OCRs both the raw crop and the enhanced one, keeps whichever scores higher (enhancement helps some plates, hurts others).
- `output.save_annotated` — also writes a copy of each image with the read text drawn on it, for quick visual QA.

## Notes

- A missing `fast_plate_ocr` install, a bad image, or a model load failure
  never crashes the run — the affected image just comes back as
  `"Unrecognized"` with `0.0` confidence, and the reason is logged.
- `runtime.workers` defaults to `1`. ONNX Runtime sessions aren't
  guaranteed free-threaded, so only raise it if you've verified your
  build/model handles concurrent `.run()` calls safely.
