# Vehicle & License Plate Detection Pipeline

Edge pipeline for a Raspberry Pi 5 that runs 24/7 next to an RTSP camera:

```
Vehicle Detect -> Crop -> Plate Detect -> Crop -> Store (SQLite buffer)
```

Metadata **and** the crop images themselves are written into a local SQLite
database (`data/pipeline_buffer.db`). That database is the hand-off point to
the separate C# dashboard application - the dashboard reads rows out of it
(and can mark them `synced = 1`); this project's job ends at "detect, crop,
time it, and put it in the buffer."

---

## 1. Project layout

Each stage of the pipeline lives in its own folder by functionality, so a
change to one stage can't accidentally break another:

| Folder / file | Responsibility |
|---|---|
| `config/config.py` | All tunable settings, loaded from `config/config.yaml` |
| `config/config.yaml` | The file you actually edit day-to-day |
| `config/bytetrack_custom.yaml` | ByteTrack tuning, referenced from `config.py` |
| `camera/camera.py` | Background-threaded RTSP reader, reconnect logic |
| `detection/detector.py` | YOLO model loading + inference (the only file that imports `ultralytics`) |
| `detection/tracker.py` | Stable vehicle IDs across frames + duplicate-vehicle suppression |
| `detection/geometry.py` | Box math (IoU, containment, cropping) |
| `detection/image_ops.py` | Image enhancement (contrast/denoise/sharpen) |
| `pipeline/pipeline.py` | Wires everything together for one frame: detect -> crop -> detect -> crop -> store |
| `pipeline/timing.py` | Per-vehicle stage timing (vehicle/plate detect + crop, ms) |
| `storage/storage.py` | SQLite buffer: schema, background batched writer, optional API hand-off |
| `api/api_client.py` | Placeholder POST to the C# dashboard's API (`features.send_via_api`) |
| `run.py` | Entry point: CLI args, main loop, graceful shutdown |
| `export_openvino.py` | One-off utility: convert a `.pt` model to a faster OpenVINO export |

Every folder is a plain Python package (`__init__.py` inside), imported as
`from config.config import PipelineConfig`, `from detection import detector`,
etc. Run everything from the project root (`python3 run.py`) so those
imports resolve.

Every line inside these files has an inline comment explaining what it does
and why - read `pipeline.py` first, it's the one that ties everything
together and matches the flow diagram above almost line for line.

---

## 2. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On a fresh Raspberry Pi 5, if `opencv-python` is slow to install, use the
headless build instead (no GUI bindings needed for a 24/7 service):
```bash
pip install opencv-python-headless
```

---

## 3. Configuration

Edit `config/config.yaml` (see the comments inside it for every option). At minimum,
for live deployment set:

```yaml
camera:
  rtsp_url: "rtsp://user:pass@<camera-ip>:554/stream1"
```

For a quick local test without a camera:
```bash
python run.py --image sample.jpg
python run.py --folder ./sample_images
```

---

## 4. Running

```bash
python run.py
```

Stops cleanly on Ctrl+C, flushing anything still queued in the SQLite writer
thread before exiting.

### Running 24/7 on the Pi

Install it as a systemd service so it starts on boot and restarts itself if
it ever crashes - see `detection-pipeline.service` (copy it to
`/etc/systemd/system/`, edit the two paths inside it, then):
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now detection-pipeline
journalctl -u detection-pipeline -f   # live logs
```

---

## 5. The SQLite buffer (what the C# dashboard reads)

Database file: `config.storage.database_path` (default `data/pipeline_buffer.db`).
Opened in **WAL mode**, so the C# side can read it freely at the same time
this pipeline keeps writing - no file locking conflicts.

### Table: `detections`

> If you're upgrading from an older copy of this pipeline, the schema below
> added new `NOT NULL` box-coordinate columns. `CREATE TABLE IF NOT EXISTS`
> won't retrofit an existing database file, so run `python reset_buffer.py`
> once after upgrading to start with a fresh, matching schema.

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER | primary key |
| `track_id` | TEXT | stable per-vehicle ID assigned by the tracker |
| `camera_source` | TEXT | which RTSP URL / source this came from |
| `vehicle_class` | TEXT | e.g. `car`, `truck`, `motorcycle` |
| `vehicle_confidence` | REAL | vehicle model confidence, 0-1 |
| `vehicle_image` | BLOB | JPEG bytes of the vehicle crop |
| `vehicle_box_x1/y1/x2/y2` | REAL | vehicle box pixel edges **in the full camera frame** - draw this on the original frame |
| `plate_detected` | INTEGER | 0 or 1 |
| `plate_confidence` | REAL (nullable) | plate model confidence, 0-1; NULL if no plate |
| `plate_image` | BLOB (nullable) | JPEG bytes of the plate crop; NULL if no plate |
| `plate_box_x1/y1/x2/y2` | REAL (nullable) | plate box pixel edges **in the vehicle crop** (i.e. relative to `vehicle_image`, not the full frame) - draw this on top of the vehicle image; NULL if no plate |
| `detected_at` | TEXT | ISO-8601 timestamp the vehicle was first seen |
| `vehicle_crop_ms` | REAL | time to crop the vehicle after detection |
| `plate_detect_ms` | REAL | time spent running the plate model |
| `plate_crop_ms` | REAL | time to crop + enhance the plate |
| `total_pipeline_ms` | REAL | **headline number**: vehicle detected -> plate crop finished, end to end |
| `synced` | INTEGER | 0 = not yet picked up by the dashboard; the dashboard should set this to 1 |
| `created_at` | TEXT | row insert time (SQLite default) |

A minimal C# read pattern (System.Data.SQLite / Microsoft.Data.Sqlite):
```csharp
using var connection = new SqliteConnection($"Data Source={dbPath}");
connection.Open();
using var command = connection.CreateCommand();
command.CommandText = "SELECT * FROM detections WHERE synced = 0 ORDER BY id";
using var reader = command.ExecuteReader();
while (reader.Read()) {
    // read columns, save vehicle_image / plate_image blobs to display, etc.
}
// after successfully handling a batch:
// UPDATE detections SET synced = 1 WHERE id IN (...)
```

On-disk JPEG copies of every crop are also written under `output/` (toggle
with `features.save_images_to_disk` in `config/config.yaml`) purely for manual
spot-checking; the database is the source of truth for the dashboard.

---

## 6. Per-vehicle timing

Every finalized row carries a full latency breakdown (see table above):
`vehicle_crop_ms + plate_detect_ms + plate_crop_ms ≈ total_pipeline_ms`. Watch
`total_pipeline_ms` in the log output or query it directly to see how the
pipeline is performing on the Pi's hardware, and which stage (usually plate
detection) is the bottleneck if it's too slow.

---

## 7. Running tests

The original unit tests under `tests/` targeted the old single-file
`main.py`. If you keep them, update their imports to point at the new
module names (`detector`, `geometry`, `tracker`, etc.) - the underlying
functions they were exercising still exist, just split across smaller files.

---

## 8. Speeding things up further / re-exporting models

`export_openvino.py` converts a `.pt` checkpoint into an OpenVINO export,
which runs noticeably faster on the Pi 5's CPU:
```bash
python export_openvino.py --weights models/vehicle.pt --imgsz 480 --benchmark --image sample.jpg
```
Point `config/config.yaml`'s `model.vehicle_weights` / `model.plate_weights` at the
resulting folder.

**Important:** a static-shape OpenVINO export only accepts exactly the size
it was exported at - feed it anything else and inference throws a shape-
mismatch error. If you re-export a model at a different `--imgsz`, update
`model.vehicle_imgsz` / `model.plate_imgsz` in `config/config.yaml` to match, or the
pipeline will crash on the first real frame. The two shipped models in
`models/` were exported at 480 (vehicle) and 640 (plate), which is why those
are the config defaults.

---

## 9. Live view (frames + boxes for the command center)

The dashboard endpoint is built on the `detections` columns: one row per
*finished* vehicle, crops as base64. Video frames don't belong there (they are
continuous and throw-away), so the Pi **serves** the live view and the
dashboard **pulls** it. Turn it on with `features.live_stream: true`.

| URL (default port 8090) | What you get |
|---|---|
| `/live/stream.mjpg` | MJPEG stream, boxes drawn in - opens in a browser or WebView2 `<img>` |
| `/live/frame.jpg` | one JPEG - poll it (e.g. every 100 ms) from WPF |
| `/live/boxes` | latest tracker boxes as JSON (`x1,y1,x2,y2` in full-frame pixels, `in_roi`, `track_id`) |
| `/live/health` | `{"ok": true, ...}` liveness check |

Add `?overlay=0` for the raw frame (draw your own boxes from `/live/boxes`).
If `live.auth_token` is set, send `?token=...` or `Authorization: Bearer ...`.

Frames come straight from the camera thread, so video stays smooth even when
inference is slower than the camera. JPEG encoding only happens while someone
is watching. `features.live_boxes` (push boxes to a server endpoint) still
exists and is independent of this.

