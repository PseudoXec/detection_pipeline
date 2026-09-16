# ANPR Local Pipeline - How to Run

Three files, one command:

- `local_model_infer.py` - runs YOUR local YOLO model (`.pt` weights) on
  photos and crops out every detection.
- `ocr_cropped_plates.py` - runs PaddleOCR on every crop and renames it to
  the recognized text.
- `main.py` - runs both stages back to back. This is the only file you run.

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
| `weights` | Path to your trained `.pt` file |
| `folder` | Path to a folder of photos to process (leave `image` as `None`) |
| `image` | Path to a single photo (leave `folder` as `None`) |
| `classes` | Optional. If your model also detects things other than plates (car, person, face, etc.), set this to your plate class's exact name, e.g. `"license_plate"`, so only those get cropped/OCR'd. Leave as `None` to keep every detected class. |

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

`cropped` / `read` = looks good. `check` = worth a manual look (no
detection, unreadable crop, or low OCR confidence).

---

## 4. Where the results land

Inside `<folder>/cropped_detections` (or wherever `--out-dir` points):

- Each crop, named `<original_photo>_cropped_<index>_<class>_<confidence>.jpg`
- After OCR, renamed again to end in `_read_<TEXT>.jpg` or `_check_<TEXT>.jpg`
- `ocr_scan_log.csv` - full detail for every file: exact confidence, which
  preprocessing variant won, and a low-confidence flag

Because the filename always keeps the original photo's name plus the crop
info, you can trace any final file straight back to the source photo just
by reading its name - the CSV is there for the exact numbers.

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
