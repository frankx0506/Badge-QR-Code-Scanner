# Badge Scanner

Offline badge + QR scanner for an Intel RealSense D435i, built to identify conference
attendees as they approach a robot. Finds badges, reads the QR code, tracks people in 3-D
from depth, reads the printed name on the badge, and runs fully offline (no internet required).

## Features

- **Badge detection** with a trained YOLO model, filtered by real-world size using depth
  (rejects posters, screens, and far-away look-alikes).
- **QR reading** via a fast zxing cascade, a model-based deep decoder for small/blurry codes,
  depth-aided burst super-resolution, and a joint multi-frame decoder for still badges at longer range.
- **Name reading**: the attendee's printed name is read off the badge with offline OCR
  (PP-OCR via RapidOCR, in a background thread) and confirmed over several reads. Names stay
  readable further away than the QR code (~1 m).
- **Person tracking** in 3-D (approach states, badge-to-person assignment, QR as identity,
  returning-visitor memory, scan-priority ordering).
- **Auto-exposure** tuned for the badge (not the room), with 90 fps motion-aware blur control.
- **Modes**: Badge → QR, QR-only, and Other objects (read QR codes on boxes, signs, phones).
- **Local web UI** for the live view, results, people, camera controls and auto-tune.

## Run

```bash
python app.py                 # live D435i, opens http://localhost:8000
python web_ui.py --replay DIR # replay saved photos instead of the camera
```

Requires Python 3, OpenCV, pyrealsense2, numpy (scipy + reedsolo for the deep decoder,
rapidocr + onnxruntime for name reading; the OCR models are in `models/ocr/`).
`--no-names` turns name reading off; `--live-codes` shows a code only right after a real read.
The OCR uses an NVIDIA GPU automatically when the driver and CUDA libraries are available.

## Layout

- `app.py`, `web_ui.py`, `web_ui.html` — application and local web UI
- `badge_detector.py`, `badge_filter.py`, `badge_scanner.py`, `badge_tracker.py` — badge pipeline
- `qr_reader.py`, `qr_soft.py`, `qr_depth.py`, `qr_burst.py`, `qr_struct.py`, `free_qr.py`, `object_qr.py` — QR decoding
- `name_reader.py` — badge name OCR
- `person_tracker.py`, `motion90.py`, `auto_tune.py` — tracking, motion, exposure
- `models/` — trained badge model, the person detector and the OCR models (ONNX; OCR models are PP-OCRv6 from PaddleOCR, Apache-2.0)
