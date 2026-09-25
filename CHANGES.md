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


## No-OCR variant (this build)

This build removes local plate-text OCR entirely. The flow is now strictly
Vehicle Detect -> Crop -> Plate Detect -> Crop -> API, and the plate crop
image is the payload - nothing in this pipeline ever tries to read the
characters on it.

| Area | Change | Files |
|---|---|---|
| OCR module | Removed entirely (FastPlateOCR engine, `read`/`read_scored`, image-variant handling) | `ocr/` (deleted) |
| Background finalize worker | Removed - it existed only to keep OCR's latency off the detection loop; finalizing a vehicle is cheap (JPEG encode + box math) without it, so it now always runs inline | `pipeline/finalize_worker.py` (deleted) |
| Plate ranking | A vehicle's best plate sighting is now picked by plate-detector confidence alone (previously OCR score, then confidence) | `pipeline/pipeline.py` |
| Config | `OcrConfig` and the `ocr:` section removed; `features.ocr_read` / `features.async_ocr` / `features.col_ocr_read` removed | `config/config.py`, `config/config.yaml` |
| Storage schema | `ocr_process` / `ocr_read` columns removed from `DetectionRecord`, the SQLite schema, the migration, and the insert/read paths | `storage/storage.py` |
| API payload | `record_to_query_params` no longer sends a `PlateNumber` field derived from OCR - only the plate crop image is uploaded; reading the plate number is left entirely to whatever is on the other end of the API call | `api/api_client.py` |
| Dependencies | `fast-plate-ocr[onnx]` removed | `requirements.txt` |

Behaviour changes to be aware of:
* Retries now stop as soon as a plate box clears `model.plate_conf_threshold`, not once an OCR read is "good enough" - `tracking.max_plate_attempts` and `plate_retry_interval_seconds` still apply the same way.
* If your C# dashboard/API expects a `PlateNumber` query parameter, it will no longer receive one from this pipeline - update the endpoint to read the plate number from the uploaded image instead (or add the parameter back with a placeholder value if the endpoint requires it to be present).
