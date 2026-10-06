# Detection Pipeline (vehicle / person) for the Raspberry Pi 5

Edge pipeline that runs 24/7 next to an RTSP camera. The command center chooses **which detection
pipeline is running**, without restarting anything:

```
vehicle mode:  Vehicle Detect -> Crop -> Plate Detect -> Crop -> OCR -> Store (SQLite buffer)
person  mode:  Person  Detect -> Crop -> Face  Detect -> Crop -> Store (SQLite buffer)
idle   mode:   camera + live view only - no models loaded, no detection
```

Every mode writes into the same local SQLite database (`data/pipeline_buffer.db`), each into its own
table, and optionally sends its rows to the C# command center API.

---

## 1. Project layout

**The rule: a file lives in the folder that owns it.** Vehicle-only code is in `modes/vehicle/`,
person-only code in `modes/person/`, everything that talks to the command center is in `api/`, and each
folder that has settings carries its own `config.py` + `config.yaml`. Only code that is genuinely shared sits
in `core/`, and `core/` knows nothing about any mode.

```
run.py                       wiring only: read settings, start shared services, run the camera loop
config/    config.yaml       GLOBAL switches and the mode toggle - nothing else
           config.py         combines the per-folder settings into one PipelineConfig
           loader.py         reads a yaml, resolves relative paths, merges config.local.yaml
core/                        shared engine: detector.py, tracker.py, geometry.py, image_ops.py, finalize_worker.py
camera/    config.py/.yaml   camera + ROI polygon
           camera.py  isapi_client.py  roi_manager.py  resolve.py  frame_loop.py
api/       config.py/.yaml   every command-center endpoint URL
           vehicle_client.py  person_client.py  roi_client.py  camera_source_client.py  result.py
live/      config.py/.yaml   live view settings
           live_server.py  box_publisher.py  box_flow.py  snapshot.py  sinks.py  outputs.py
storage/   config.py/.yaml   shared SQLite buffer settings
           buffer.py  hub.py  reset_buffer.py
control/                     mode_manager.py (switching the active mode)
modes/     base.py  registry.py  idle.py        what every mode looks like, name -> mode, the do-nothing mode
  vehicle/ config.py/.yaml   everything vehicle mode needs
           bytetrack.yaml  models/  mode.py  plate_worker.py  plate_enhance.py  geometry.py
           store.py (table `detections`)  timing.py  ocr/  debug_plate.py
  person/  config.py/.yaml   everything person mode needs
           bytetrack.yaml  models/  mode.py  face.py  quality.py  geometry.py
           store.py (table `person_detections`)
tools/     export_ncnn.py    exports a model for either mode
deploy/    detection-pipeline.service          docs/  CHANGES.md  person-api-contract.md
```

Run everything from the project root (`python3 run.py`) so the imports resolve.

### Where each setting lives

| You want to change | File |
|---|---|
| Which mode runs by default, on/off switches (API, live view, ...), CPU threads | `config/config.yaml` |
| Camera URL and login, ROI polygon | `camera/config.yaml` |
| Command-center endpoint URLs, polling | `api/config.yaml` |
| Live view port, token, quality | `live/config.yaml` |
| Database path, retention | `storage/config.yaml` |
| Vehicle models, thresholds, plate/OCR rules, which columns to fill | `modes/vehicle/config.yaml` |
| Person model, face search, ROI rules for people | `modes/person/config.yaml` |

Relative paths in a mode's yaml (model files, tracker file) are relative to **that mode's folder**, so a mode
can be moved or copied as a unit. Paths for runtime data (`data/`, `output/`) are relative to the project root.

### Where the files went

| Original project | Now |
|---|---|
| `pipeline/pipeline.py`, `plate_worker.py`, `timing.py` | `modes/vehicle/mode.py` (class `VehicleMode`), `plate_worker.py`, `timing.py` |
| `pipeline/finalize_worker.py` | `core/finalize_worker.py` (shared) |
| `storage/storage.py` | `modes/vehicle/store.py` (vehicle table) + `storage/buffer.py` (shared writer) |
| `api/api_client.py` | `api/vehicle_client.py` |
| `ocr/` | `modes/vehicle/ocr/` |
| `detection/detector.py`, `tracker.py` | `core/` |
| `detection/geometry.py` | `core/geometry.py` (generic) + `modes/vehicle/geometry.py` (vehicle ROI rule, plate crop) |
| `detection/image_ops.py` | `core/image_ops.py` (JPEG encode) + `modes/vehicle/plate_enhance.py` (plate sharpening) |
| `models/*.onnx, *.pt` | `modes/vehicle/models/` |
| `config/bytetrack_custom.yaml` | `modes/vehicle/bytetrack.yaml` (person mode has its own copy) |
| `config/config.py` + one big `config.yaml` | split per folder, see the table above |
| `run.py` (camera loops, camera lookup, live server setup) | `camera/frame_loop.py`, `camera/resolve.py`, `live/outputs.py`, `control/mode_manager.py` |
| `reset_buffer.py` | `storage/reset_buffer.py` |
| `debug_plate.py` | `modes/vehicle/debug_plate.py` |
| `export_ncnn.py` | `tools/export_ncnn.py` |
| `detection-pipeline.service`, `CHANGES.md` | `deploy/`, `docs/` |

---

## 2. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the Pi, `opencv-python-headless` (>= 4.8) is the lighter choice for a headless service.

### Person mode needs two model files (not included)

1. **Person detector**: export your person model for the Pi and point `weights` in `modes/person/config.yaml` at it (the folder `modes/person/models/` is where it goes):
   ```bash
   python tools/export_ncnn.py --weights modes/person/models/person.pt --imgsz 416
   ```
   A static-shape export only accepts the size it was exported at, so keep `person.imgsz` equal to it.
2. **Face detector (YuNet)**: download `face_detection_yunet_2023mar.onnx` from the OpenCV Zoo
   (`models/face_detection_yunet` folder of github.com/opencv/opencv_zoo) into `modes/person/models/`. Check the
   license shipped in that folder. If you do not want faces, set `face_enabled: false` in `modes/person/config.yaml`.

Licensing reminder: Ultralytics YOLO is AGPL-3.0 (commercial use needs their Enterprise License or a
different detector). That was already true for vehicle mode; person mode uses the same loader.

---

## 3. Configuration

Each folder's `config.yaml` is commented inside. `config/config.yaml` holds only the global switches
(`mode`, `switches`, `runtime`); everything else is in the folder that owns it (table in section 1).

* **Machine-specific values** (the camera password, IPs): put them in a `config.local.yaml` **next to** the file
  they belong to (for the camera: `camera/config.local.yaml`, same structure). It is merged on top and ignored
  by git, so `camera/config.yaml` can keep a placeholder URL. The shipped `camera/config.yaml` still has your
  current RTSP URL with its password: move it before the project goes to GitHub.
* **A misspelled setting is reported** as a warning at start-up (`unknown setting 'vehicle.model.vehcile_weights'`)
  instead of being silently ignored.
* **`--config file.yaml`** applies one extra file after all the others. Its top-level keys are `mode`, `runtime`,
  `switches`, `camera`, `roi`, `api`, `live`, `storage`, `vehicle`, `person`: handy for a test or a one-off experiment.
* The master switch `switches.api: false` turns off ROI fetch, camera-source lookup and all sending, for every mode.

Quick local tests without a camera:
```bash
python run.py --image sample.jpg --mode person
python run.py --folder ./sample_images --mode vehicle
```

---

## 4. Switching the detection pipeline from the command center

The control endpoint lives on the live-server port (default 8090) and uses the same `live.auth_token`.
It is on while `mode.control_enabled: true`, **even if `switches.live_stream` is false** (then the camera
image routes answer 404 and only `/control/*` and `/live/health` are served).

| Request | Meaning |
|---|---|
| `GET  /control/mode` | Current state (see below) |
| `POST /control/mode` body `{"mode": "person"}` (or `?mode=person`) | Ask for a mode. Answers **202** immediately; the switch happens in the background |
| `GET  /live/health` | Also contains the mode status under `"mode"` |

Status example:
```json
{ "mode": "person", "state": "running", "requested": null, "since": "2026-10-05T14:02:11",
  "allowed": ["vehicle", "person", "idle"],
  "last_switch": { "to": "person", "ok": true, "error": null, "source": "command-center", "seconds": 3.4, "at": "..." } }
```
`state` is `running`, `idle` or `switching`. Errors: 400 unknown mode (the answer lists the allowed ones),
401 bad token, 404 control disabled. After a POST, poll `GET /control/mode` until `mode` is the one you asked
for, or until `last_switch.ok` is `false` (then `last_switch.error` says why and the old mode is still running).

```bash
curl -X POST http://<pi-ip>:8090/control/mode -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" -d '{"mode":"person"}'
```
```csharp
using var http = new HttpClient { BaseAddress = new Uri("http://<pi-ip>:8090") };
http.DefaultRequestHeaders.Authorization = new("Bearer", token);          // only if live.auth_token is set
await http.PostAsJsonAsync("/control/mode", new { mode = "person" });     // 202 Accepted
var status = await http.GetFromJsonAsync<JsonElement>("/control/mode");   // poll until status.mode == "person"
```

### What "seamless" and "CPU friendly" mean here

* **Only one mode ever processes frames.** The other is not paused, it is *stopped*: its worker threads
  are joined and its models are freed, so an unused pipeline costs no CPU and no RAM.
* **Camera, ROI polling, SQLite stores and live view never restart** during a switch, so the stream keeps flowing.
* **No gap in detection** (`mode.preload_on_switch: true`, default): the new mode is loaded and warmed up
  in the background while the old one keeps detecting; the swap is a pointer change between two frames.
  Cost: both sets of models are in RAM for the few seconds the load takes. Set it to `false` on a tight
  memory budget: the old mode is freed first, so there is a short gap while the new one loads.
* **Nothing is lost**: when a mode stops, vehicles/people it was still tracking are stored (a vehicle
  without a finished plate read is stored as "no plate", a person without a face as `face_detected = 0`),
  and rows still waiting to be sent keep being retried because the stores outlive any mode.
* **A broken switch cannot take detection down**: if the new mode fails to load (missing weights...) the
  old mode keeps running and the failure is reported in `last_switch`. If a mode throws on 10 frames in a
  row it falls back to idle. Requests coalesce: A, B, C in quick succession ends on C.
* **The last choice survives reboots** (`mode.remember_last`, file `data/active_mode.json`). Start-up
  order: `--mode` flag, then the saved choice, then `mode.default`.
* Live boxes of the previous mode are cleared on a switch, and track ids are prefixed (`v...` vehicle,
  `p...` person) so they can never be confused.

---

## 5. The SQLite buffer

Opened in WAL mode, so the C# side can read while the Pi writes. Vehicle rows are in `detections`
(unchanged, see below); person rows are in `person_detections`. `synced`: `0` waiting, `1` delivered (or
nothing to deliver), `2` the API said this event is unacceptable (kept for inspection, never retried).

### `detections` (vehicle mode) - unchanged
`id, track_id, camera_source, vehicle_class, vehicle_confidence, vehicle_image, vehicle_box_x1..y2,
plate_detected, plate_confidence, plate_image, plate_box_x1..y2, detected_at, vehicle_detect_ms,
vehicle_crop_ms, plate_detect_ms, plate_crop_ms, total_pipeline_ms, ocr_process, ocr_read, camera_name,
camera_ip, camera_location, synced, created_at`. An existing database file keeps working; the new table
is created next to it automatically, no `storage/reset_buffer.py` needed.

### `person_detections` (person mode) - one row per person track
| Column | Meaning |
|---|---|
| `event_uuid` | unique id of the event (idempotency key for the API) |
| `device_id`, `session_id` | which Pi, which program run (track ids restart every run) |
| `track_id`, `track_first_seen_at`, `track_last_seen_at` | tracker id (`p12`) and how long the person was in view |
| `camera_source`, `camera_name`, `camera_ip`, `camera_location`, `frame_width`, `frame_height` | where it came from; the frame size the boxes refer to |
| `person_confidence`, `person_image`, `person_box_x1..y2` | person crop (JPEG); box in **full-frame** pixels |
| `face_detected` | 0 or 1 |
| `face_confidence`, `face_image`, `face_box_x1..y2`, `face_landmarks` | face crop (JPEG, with margin); box and the 5 landmarks (JSON `[[x,y]x5]`) in pixels **inside `person_image`** |
| `face_sharpness`, `face_quality_score`, `face_attempts` | quality of the stored face; how many face tries the track used |
| `image_format` | `jpeg` |
| `detected_at` | local time `YYYY-MM-DD HH:MM:SS`, same format as `detections` |
| `person_detect_ms`, `person_crop_ms`, `face_detect_ms`, `face_crop_ms`, `total_pipeline_ms` | timings; `total` = frame grabbed -> row built |
| `pipeline_version`, `person_model`, `face_model`, `cpu_temp_c` | what produced it, and the Pi's temperature (watch for throttling) |
| `synced`, `sync_attempts`, `last_sync_attempt_at`, `synced_at`, `sync_error` | delivery bookkeeping |
| `created_at` | row insert time |

```csharp
command.CommandText = "SELECT * FROM person_detections WHERE synced = 0 ORDER BY id";
// after handling a batch:  UPDATE person_detections SET synced = 1 WHERE id IN (...)
```

---

## 6. Person mode in detail

1. YOLO detects and tracks people (ByteTrack, ids `p<n>`).
2. A person counts as inside the ROI when their **feet** (bottom-centre of the box) are in the polygon.
3. For each such person a face is searched in the top part of their crop (`person.face_search_region`) by
   YuNet on a **background thread**, so a slow face model never slows detection or the live overlay.
4. A face is kept only if it is big and sharp enough (`face_min_px`, `face_min_sharpness`); the best one wins.
5. The row is written when a face scores `face_good_enough_quality`, **or** after `face_max_attempts` tries,
   **or** when the person leaves for `stale_finalize_seconds`, **or** when the track expires, **or** when the
   mode is stopped. A person with no usable face is still stored (`face_detected = 0`).
6. The stored person image is the frame the stored face came from, so the face box/landmarks line up with it.

Tuning knobs worth knowing (all in `modes/person/config.yaml`): `face_max_attempts` x `face_retry_interval_seconds` is how long the Pi keeps
trying before storing a person without a face (default about 4 s); raise it if people usually turn toward
the camera late. `roi_crop_detect: true` runs the model only on the ROI window (faster) but then people
outside it are not seen at all.

### Sending person events to the command center
**The Pi never identifies anyone: there is no face matching code in this project**.
It finds and crops faces and sends them; matching happens on the server. `api/person_client.py` sends one
`multipart/form-data` request per person track to `api.person_endpoint_url` (`api/config.yaml`; switch on with
`send_via_api: true` in `modes/person/config.yaml`): the metadata as form fields, `PersonImage` and, when a face
was found, `FaceImage` as files. **The full field list, coordinate conventions, answers and a C# model are in
`docs/person-api-contract.md`: that is the file to give the command-center team.**

* `EventId` is the idempotency key. The endpoint must ignore an `EventId` it already stored; 409 counts as delivered.
* The camera password is removed from `CameraSource` before anything is stored in the person table or sent.
* Only answers that are about *this* event (400, 413, 415, 422) park it (`synced = 2`). Server trouble, a wrong URL
  or token (404, 401, 403, 5xx, 429, no connection) keep it queued and retried with a growing wait (up to an hour
  per event), so an outage of any length loses nothing and fixing the setup releases the backlog.
* Some columns stay local on the Pi and are not sent (track first/last seen, face sharpness, face attempts, model
  names, pipeline version, CPU temperature). Add them to `build_form()` if the server should have them.

---

## 7. Live view

Unchanged: `/live/stream.mjpg`, `/live/view`, `/live/frame.jpg`, `/live/boxes`, `/live/roi`,
`/live/health`, enabled with `switches.live_stream: true` (settings in `live/config.yaml`). `/live/boxes` now includes `"mode"`; the list
keeps its old name `vehicles` for compatibility and holds person boxes in person mode.

---

## 8. Running 24/7 and speed

```bash
sudo cp deploy/detection-pipeline.service /etc/systemd/system/   # edit the two paths inside
sudo systemctl daemon-reload && sudo systemctl enable --now detection-pipeline
journalctl -u detection-pipeline -f
```

Speed: `python tools/export_ncnn.py --weights modes/<vehicle|person>/models/<name>.pt --imgsz <size> --benchmark` (NCNN is
usually the fastest on the Pi 5's CPU). Vehicle model sizes and the ROI window are described in `docs/CHANGES.md`.
