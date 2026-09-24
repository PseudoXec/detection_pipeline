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
