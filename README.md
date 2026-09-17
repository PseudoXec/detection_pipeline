# ANPR Local Pipeline - How to Run

Three files, one command:

- `local_model_infer.py` - runs YOUR local YOLO model (`.pt` weights) on
  photos and crops out every detection.
- `ocr_cropped_plates.py` - runs PaddleOCR on every crop and renames it to
  the recognized text.
- `main.py` - detects vehicles, runs plate detection inside each vehicle crop,
  then runs OCR on each plate crop. This is the only file you run.

All three must live in the **same folder**.

---

## 1. First-time setup (do this once)

Open PowerShell in the project folder (where `main.py` lives).

```powershell
cd C:\Users\User\Documents\Detection_Inference

# Create the venv (only if you don't already have one)
py -m venv .venv

# Activate it - your prompt should now start with (.venv)
.venv\Scripts\Activate.ps1
```

If PowerShell blocks that last command with an execution-policy error, run
this once, then try activating again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install everything from `requirements.txt`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sanity check (should print `ok` with no errors):

```powershell
python -c "import ultralytics, cv2, paddleocr; print('ok')"
```

---

## 2. Point it at your model and your photos

Open `main.py` and edit the `CONFIG` dict near the top:

| Key | What to set it to |
|---|---|
| `vehicle_weights` | Path to your trained `vehicle.pt` file. It must contain both vehicle and plate classes. |
| `folder` | Path to a folder of photos to process (leave `image` as `None`) |
| `image` | Path to a single photo (leave `folder` as `None`) |
| `rtsp_url` | Optional RTSP camera URL (leave `image` and `folder` unset) |
| `vehicle_classes` | Optional comma-separated vehicle class names, e.g. `"car,truck,bus"`. |

Everything else in `CONFIG` has a working default - you don't need to
touch it to get a first run going.

---

## 3. Run it

Every time you come back to a new terminal, activate the venv first:

```powershell
cd C:\Users\User\Documents\Detection_Inference
.venv\Scripts\Activate.ps1
python main.py
```

No flags needed if `CONFIG` is filled in. You'll see one line per file as
it runs, e.g.:

```
photo1.jpg -> cropped (2)
photo1_cropped_1_license_plate_92.jpg -> read (ABC1234)
photo1_cropped_2_license_plate_61.jpg -> check (AB0122)
```

`no_plate` means a vehicle was detected without a matched plate. `orphan` is
an emitted plate result with no vehicle match. `read` = looks good. `check` =
worth a manual look (unreadable crop or low OCR confidence).

### Live RTSP camera

You can put the camera URL directly in the `CONFIG` section of `main.py`:

```python
"image": None,
"folder": None,
"rtsp_url": r"rtsp://user:password@192.168.1.50:554/stream1",
```

Then run:

```powershell
python main.py
```

Or pass the URL without editing the file:

```powershell
python main.py --rtsp-url "rtsp://user:password@192.168.1.50:554/stream1"
```

The live mode reconnects after a failed read, requests 1280x720 at 15 FPS,
keeps the capture buffer small, processes one frame every two captured frames
by default, and opens an
`RTSP detection preview` window with vehicle and plate boxes. Press `q` in
that window or `Ctrl+C` in the terminal to stop. Adjust the stream settings with:

```powershell
python main.py --rtsp-url "rtsp://user:password@camera/stream" `
  --stream-width 1280 --stream-height 720 --stream-fps 15 `
  --stream-frame-skip 2 --stream-buffer-size 1
```

The camera's RTSP profile controls the maximum available quality. If the
camera ignores width, height, or FPS requests, select its high-resolution or
low-resolution stream profile in the URL instead, such as `stream1` or
`stream2`.

Stop a continuous stream with `Ctrl+C`. For a short connection test, use
`--stream-max-frames`, for example `--stream-max-frames 10`.
Use `--no-stream-preview` when running without a desktop display.

---

## 4. Where the results land

Inside `<folder>/vehicle_pipeline_output` (or wherever `--out-dir` points):

- `vehicle_detection/` contains review thumbnails only. These images never
  feed plate detection or OCR.
- `plate_detection/` contains plate crops detected and extracted inside each
  saved vehicle crop, with an 8-pixel default border.
- `ocr/` contains OCR result files for the image/folder workflow.
- `pipeline_log.csv` contains one row per vehicle/plate result, including
  `track_id`, match method, containment, IoU, and orphan rows.

RTSP output is written to the same date-based output folder. RTSP uses
Ultralytics tracking for persistent vehicle IDs, a latest-frame capture queue,
and a separate bounded OCR queue. Plate detection/OCR is attempted at most
three times per track and each track produces one final CSV row when accepted
or when it disappears.

The vehicle detector runs on the source frame, then the dedicated plate
detector runs on each vehicle crop. This multi-level flow reduces the plate
search area and keeps vehicle and plate model responsibilities separate.

---

## 5. Useful command-line overrides

Any `CONFIG` value can be overridden per-run without editing the file:

```powershell
python main.py --skip-detection          # OCR only, reuse existing crops
python main.py --skip-ocr                # detect+crop only, no OCR
python main.py --conf-threshold 0.5      # loosen/tighten detection confidence
python main.py --classes license_plate  # filter to one class for this run
```

---

## 6. Troubleshooting

- **`ModuleNotFoundError`** - the venv isn't activated, or packages didn't
  install into it. Check `where python` points inside `.venv\Scripts\`.
- **"no python" / `python --version` fails** - the `.venv` folder is
  missing or partially broken. Delete it (`Remove-Item -Recurse -Force
  .venv`) and redo step 1.
- **Every file says `check (no detections)`** - your model may be
  detecting things under a different class name than what you set in
  `CONFIG["classes"]`. Set `classes` to `None`, run once, and check the
  class name baked into the resulting crop filenames.
- **Weights file not found** - double check `CONFIG["weights"]` in
  `main.py` is a real path to your `.pt` file.
