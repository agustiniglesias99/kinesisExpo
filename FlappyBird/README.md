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
| G | Switch between marker mode and bare-hand (MediaPipe) mode |
| Click on a sticker | Marker mode: learn the sticker colour |
| C | Same, with the sticker inside the white square in the middle |
| S | Save the current camera frame to `capturas/` (for tuning) |
| V | Toggle a double-size camera view (for calibration) |
| F11 | Toggle fullscreen (starts in fullscreen; set `FULLSCREEN = False` to start windowed) |
| ESC | Quit |

The camera preview in the top-right corner shows the detected hand
(green = open, red = fist, yellow = in between) and a bar with how open
the hand is.

## Marker mode (black glove + stickers)

MediaPipe can't see a hand inside a plain black glove, so by default
(`TRACKING_MODE = "markers"`) the game looks for **fluorescent stickers on the
fingertips** instead. No AI model is involved.

Preparing the glove:

- Stick a round fluorescent sticker (~1.5 cm) on the **pad of each fingertip,
  palm side**: that side faces the camera. With the hand open the camera sees
  them; in a fist the fingertips fold into the palm and they hide.
- Colour: **fluorescent pink, magenta or green**. Avoid orange and red (skin
  and wood look similar) and yellow: under warm light the camera sees it
  almost white, just like a lamp. Also avoid any colour that is in the
  background or on the players' clothes.

Calibration (press V for a bigger camera view):

1. **Click on a sticker in the camera view** (or hold it inside the small
   white square in the middle and press `C`). Its hue, saturation and
   brightness are measured and saved in `marcadores.json`, reused on the next
   run. Recalibrate when the lighting changes (e.g. at the venue). A warning
   appears if that colour is also somewhere else in the image.
2. Each sticker found is circled in green (open) / red (fist). Grey circles
   are spots of that colour that are too small or too big to be a sticker.
3. Open hand = 3 or more stickers visible (`MARKER_OPEN_MIN`), fist = 1 or
   fewer (`MARKER_FIST_MAX`).

Note: a fist and "no hand in view" look the same (no stickers), so pulling an
open hand out of view also counts as closing it.

If it misreads, press `S` in the bad pose: raw frames are saved in
`capturas/` and can be used to re-tune offline.

## High score

The best score is kept in `records.csv` next to the script (one row per new
record, `fecha_hora;puntaje`, `;`-separated so it opens in columns in Excel).
Delete the file to reset the record.

## Vertical monitor

The game adapts to the screen's aspect ratio. On a monitor rotated to portrait
(set the orientation to *Portrait* in Windows display settings) it fills the
screen in a phone-like layout. `ORIENTATION = "vertical"` forces that layout,
e.g. to test it in a window on a landscape screen.

## Tuning

Constants at the top of `flappy_hand.py`:

- `CAMERA_INDEX` – use `1`, `2`… if you have several cameras.
- `PIPE_GAP`, `PIPE_SPEED`, `PIPE_INTERVAL`, `GRAVITY`, `FLAP_VELOCITY` – difficulty.
- `CLOSED_MAX_FINGERS` / `OPEN_MIN_FINGERS` – how strict the fist / open-hand
  detection is (bare-hand mode).
- `MARKER_OPEN_MIN`, `MARKER_FIST_MAX`, `MARKER_MIN_AREA`, `MARKER_MAX_AREA`,
  `MARKER_HUE_TOLERANCE` – marker mode, see above.
- `STABLE_FRAMES` – frames a pose must hold before it counts (filters glitches).
