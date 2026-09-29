# Optimization pass - what changed

| Area | Change | Files |
|---|---|---|
| Vehicle model input | Runs on the ROI + margin window (896x512) instead of a 1280x1280 letterboxed frame; boxes mapped back to full-frame px. `vehicle_imgsz: [512, 896]` | `detection/geometry.py`, `detection/detector.py`, `pipeline/pipeline.py`, config |
| CPU threads | Auto ONNX threads = cores-1 (was `cores//2-1`, i.e. 2 of 4 on a Pi 5). `inference_threads: 3`, `decode_threads: 2`, `cv2.setNumThreads(2)` | `detection/detector.py`, `run.py`, config |
| Camera load | `camera.max_fps` (default 10 in yaml): frames over the cap are decoded but skip colour-convert + copy. Also lower the camera's own fps in its web UI | `camera/camera.py`, config |
| Plate retries | New track id -> ONE plate pass -> stored. Same id in later frames -> skipped. `max_plate_attempts: 1`. If raised, each retry re-crops from the current frame (was: same frozen crop) | `pipeline/pipeline.py` |
| Memory leak | Per-track state is one `_TrackState`; crop freed at finalize; whole entry dropped `track_ttl_seconds` after last seen; a vehicle that vanishes before finalizing is stored as "no plate" instead of lost; dedup table pruned | `pipeline/pipeline.py`, `detection/tracker.py` |
| Storage growth | API result is SENT / SKIPPED / FAILED. SENT -> row deleted (or synced=1) + its output/ JPEGs removed. SKIPPED (no plate) -> synced=1 so retention removes it. FAILED -> retried every `send_retry_seconds` (5 tries per row). Old no-plate synced=0 rows are closed out on the first retry pass | `api/api_client.py`, `storage/storage.py` |
| Disk copies | JPEG bytes already encoded for the DB are written by the storage writer thread (no imwrite/second encode in the detection loop); file names carry a per-run tag so restarts no longer overwrite `v1..vN` | `pipeline/pipeline.py`, `storage/storage.py` |
| Housekeeping | Runs at startup and every 6 h (was: first run after 24 h). Also sweeps output/ by age; optional `max_unsynced_days` | `run.py`, `storage/storage.py` |

New settings (all in `config/config.yaml`): `camera.max_fps`, `roi.crop_margin`, `model.inference_threads`,
`tracking.track_ttl_seconds`, `storage.delete_disk_images_after_send`, `storage.max_unsynced_days`,
`storage.send_retry_seconds`, `runtime.opencv_threads`, `features.roi_crop_detect`.

Behaviour changes to be aware of:
* The live overlay only shows vehicles inside the ROI + margin window (the model no longer looks elsewhere).
* Plate detection now runs once per vehicle, on the first frame it is fully inside the ROI.
* `vehicle_imgsz` must match the exported model: `[512, 896]`.


## OCR accuracy pass

| Area | Change | Files |
|---|---|---|
| Reversed reads | PaddleOCR 3.x's document-orientation classifier and unwarping model (on by default) are switched off - they can rotate a plate crop 180 degrees, which flips the order of the text boxes. Fragments are now sorted into reading order from their box positions (rows top-to-bottom, left-to-right within a row) instead of being joined in detector order | `ocr/plate_reader.py` |
| OCR input | The crop is read with an edge-replicated border (detector clips characters that touch the edge) and BOTH the enhanced and the plain crop are tried; the better read wins. Early exit when the first read is already confident | `ocr/plate_reader.py`, `pipeline/pipeline.py` |
| Best-of-N plates | `max_plate_attempts: 4`, spaced `plate_retry_interval_seconds: 0.4` apart. Each attempt is cut, enhanced and OCR'd; the best (OCR score, then plate confidence) is kept and stored, and retries stop as soon as a read scores >= `ocr.accept_score`. A vehicle that leaves the ROI is stored with its best sighting after `plate_stale_finalize_seconds`. The stored vehicle image is the frame the stored plate came from | `pipeline/pipeline.py`, `config/config.py` |
| Plate crop | Taller upscale target (`crop.plate_min_crop_height` 120 -> 160) and padding = max(4 px, 12% of plate height) so first/last characters are not clipped | `detection/geometry.py`, `config/config.py` |
| JPEG quality | New `storage.plate_jpeg_quality: 96` for the plate image (vehicle image stays at `jpeg_quality: 90`) | `pipeline/pipeline.py`, `config/config.py` |

Behaviour changes to be aware of:
* A vehicle can now take up to ~1.2 s (4 attempts x 0.4 s) to be stored when its first plate read is weak. Set `tracking.max_plate_attempts: 1` to get the old one-pass behaviour back.
* "Plate found" no longer means "stored immediately": the DB row appears when the read is good enough, attempts run out, or the vehicle leaves the ROI.


## API metadata pass

| Area | Change | Files |
|---|---|---|
| Camera identity | `CameraName`/`CameraIpAddress`/`Cameralocation` sent to the dashboard API are now fetched from the camera itself over Hikvision ISAPI (`/ISAPI/System/deviceInfo`), once at startup, instead of hard-coded strings. Falls back to just the IP if the camera doesn't answer or ISAPI is off; `Cameralocation` is only sent if the camera has one configured | `camera/isapi_client.py`, `run.py`, `storage/storage.py`, `api/api_client.py` |
| API payload trim | Bounding boxes (`VehicleBoxX1-4`, `PlateBoxX1-4`) and per-stage timings (`vehicle_detect_ms`, `vehicle_crop_ms`, `plate_detect_ms`, `plate_crop_ms`) are no longer sent to the API. `TotalPipelineMs` is now included instead | `api/api_client.py` |

New settings (in `config/config.yaml` under `camera:`): `isapi_enabled`, `isapi_port`, `isapi_https`,
`isapi_username`, `isapi_password`, `isapi_timeout_seconds`.

## RTSP source: static or via command center API

| Area | Change | Files |
|---|---|---|
| Camera source | New `camera.source_mode: "static" \| "api"`. `"static"` (default) keeps today's behaviour - `camera.rtsp_url` from config.yaml. `"api"` fetches `rtsp_url` (+ credentials) from the command center at startup instead, reusing `api.roi_endpoint_url`/`api.camera_id` (the same request already used for the ROI polygon) - falls back to the static `rtsp_url`, if any, on any fetch failure | `api/camera_source_client.py` (new), `run.py`, `config/config.py`, `config/config.yaml` |

The command center endpoint this expects doesn't exist yet - `api/camera_source_client.py`'s docstring spells out the proposed JSON shape (an `rtspUrl` + optional `username`/`password` added to the existing ROI response). Once the real endpoint is confirmed, only `parse_camera_source_response()` in that file should need changing - `run.py` and everything downstream of it already just consume a plain `rtsp_url` string, same as today.

## PaddleOCR removal

| Area | Change | Files |
|---|---|---|
| OCR backend | PaddleOCR removed entirely - deleted as a dependency, a config option, and a code path. `fast_plate_ocr` is now the only OCR backend (it was already the default). `ocr_worker.py` (an unused, unwired PaddleOCR-VL worker) was also deleted | `ocr/__init__.py`, `ocr/plate_reader.py` (deleted), `ocr/ocr_worker.py` (deleted), `config/config.py`, `config/config.yaml`, `requirements.txt` |

Behaviour changes to be aware of:
* `ocr.engine` and `ocr.lang` no longer exist as config options - fast_plate_ocr always runs, they are not just defaults anymore.
* 2-row/stacked plates (e.g. motorcycles), which PaddleOCR's fragment-reordering used to handle, have no OCR fallback now - see `ocr/fast_plate_reader.py`'s module docstring.

## Dead-code cleanup

| Area | Change | Files |
|---|---|---|
| Unused code | Removed `compute_containment()` and its unused config fields (`plate_containment_threshold`, `plate_min_iou`) - leftover from an earlier plate-to-vehicle matching approach that was replaced and never wired in. Also removed the never-called `queue_depth()` getter, the never-called `as_dict()` config method (and its now-unused `asdict` import), and one unused `typing.List` import | `detection/geometry.py`, `config/config.py`, `pipeline/finalize_worker.py` |

No behaviour change - none of the above were reachable from the running pipeline.

## Box lag / oversized box pass

| Area | Change | Files |
|---|---|---|
| Flow seeding | Seeds only on pixels that moved between two frames (fixed camera) and on the inner ~60% of big boxes, so road/background no longer stalls the box | `live/box_flow.py` |
| Flow tracking | Follows the moving point cluster only; 640px work width, 31px window, 4 pyramid levels, up to 2 box-sizes per step | `live/box_flow.py` |
| Extrapolation | Blend 0.4s -> 0.15s, less velocity damping, lead up to 1.2s / 1 box size (was 0.8s / half a box) | `live/live_server.py` |
| Size gate | Boxes may now SHRINK up to 2x per update (grow stays 1.35x) so an oversized first box settles fast | `detection/detector.py` |
| Tracker | Kalman velocity weight 1/50 -> 1/25; new_track_thresh 0.40, track_low_thresh 0.15, match_thresh 0.8 | `detection/detector.py`, `config/bytetrack_custom.yaml` |
| Thresholds | vehicle_conf 0.35 -> 0.40, detector_conf_floor 0.10 -> 0.20, NMS IoU 0.45 -> 0.40 (low-conf boxes are the loose, oversized ones) | `config/config.yaml`, `config/config.py` |

## Receding-vehicle pass (boxes racing ahead of / stalling behind vehicles going away)

| Area | Change | Files |
|---|---|---|
| Extrapolation | Damped projection (speed decays, tau 0.35s) instead of straight v*t; cap 0.6 box; velocity smoothing 0.65/0.35 | `live/live_server.py` |
| Flow speed guard | An abrupt speed jump between steps is blended with the previous velocity | `live/box_flow.py` |
| Flow re-seed | Re-seeds on pixels that moved, not on the stationary box interior (cause of stalls) | `live/box_flow.py` |
| Moving cluster | Stricter selection (80th pct > 2px, keep points >= 50% of it) | `live/box_flow.py` |
