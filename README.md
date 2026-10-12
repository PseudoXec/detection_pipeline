# Lean Detection Pipeline for Raspberry Pi 5

Optimized detection pipeline for Raspberry Pi supporting two lean detection flows:

---

### VEHICLE DETECTION FLOW
`vehicle detect → crop → plate detect → crop → send metadata`

### PERSON DETECTION FLOW
`person detect → crop → face detect → crop → send metadata`

---

## 1. Project Structure

```
run.py                       Entry point: reads settings, initializes buffer, runs camera loop
config/    config.yaml       Global switches and mode settings
           config.py         Combines settings into PipelineConfig
           loader.py         Reads yaml, resolves relative paths, merges config.local.yaml
core/                        Shared engine: detector.py, geometry.py, urls.py
camera/    config.py/.yaml   Camera + ROI polygon
           camera.py  isapi_client.py  roi_manager.py  resolve.py  frame_loop.py
api/       config.py/.yaml   Command-center endpoint URLs
           sender.py  roi_client.py  camera_source_client.py  result.py
storage/   config.py/.yaml   Shared SQLite buffer settings
           buffer.py  reset_buffer.py
control/   mode_manager.py   Detection mode manager
modes/     base.py  registry.py  idle.py
  vehicle/ config.py/.yaml   Vehicle mode settings
           mode.py  geometry.py  models/
  person/  config.py/.yaml   Person mode settings
           mode.py  face.py  quality.py  geometry.py  models/
deploy/    detection-pipeline.service
docs/      CHANGES.md  person-api-contract.md
```

## 2. Shared Services

Both flows share:
- **Unified SQLite Buffer (`storage/buffer.py`)**: A single worker thread with thread-safe queue, WAL mode, batch insertions into `detections` / `person_detections`, automatic retry with backoff, and disk retention cleanup.
- **Unified API Sender (`api/sender.py`)**: Sends metadata to the existing endpoints for vehicle detections (`api.vehicle_endpoint_url`) and person events (`api.person_endpoint_url`).
- **ROI Manager (`camera/roi_manager.py`)**: Synchronizes detection polygon with the command center.

## 3. Running the Pipeline

```bash
# Run with camera using default mode (vehicle)
python run.py

# Run specifically in vehicle mode
python run.py --mode vehicle

# Run specifically in person mode
python run.py --mode person

# Run on a single image or folder of images
python run.py --mode vehicle --image test.jpg
python run.py --mode person --folder ./imgs
```