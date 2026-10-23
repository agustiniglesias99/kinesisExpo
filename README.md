# Flappy Hand

Flappy Bird controlled with your hand through the webcam.
**Open your hand, then close it into a fist → the bird flaps.**

## Setup (Windows)

1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
   (tick "Add python.exe to PATH" during install). MediaPipe may not have
   wheels for the very newest Python yet.
2. In this folder:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python flappy_hand.py
   ```

On first run the hand-landmark model (`hand_landmarker.task`, ~8 MB) is
downloaded next to the script.

## Controls

| Action | Effect |
|---|---|
| Open hand → fist | Flap / start / restart |
| SPACE | Flap (keyboard fallback) |
| F11 | Toggle fullscreen (starts in fullscreen; set `FULLSCREEN = False` to start windowed) |
| ESC | Quit |

The camera preview in the top-right corner shows the detected hand
(green = open, red = fist) and how many fingers are counted as extended.

## Tuning

Constants at the top of `flappy_hand.py`:

- `CAMERA_INDEX` – use `1`, `2`… if you have several cameras.
- `PIPE_GAP`, `PIPE_SPEED`, `PIPE_INTERVAL`, `GRAVITY`, `FLAP_VELOCITY` – difficulty.
- `CLOSED_MAX_FINGERS` / `OPEN_MIN_FINGERS` – how strict the fist / open-hand
  detection is.
