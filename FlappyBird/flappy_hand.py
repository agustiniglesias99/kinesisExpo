"""Flappy Bird controlled by your hand.

Open your hand, then close it into a fist: every open -> closed transition
makes the bird flap. The webcam is read with OpenCV and the game is drawn
with pygame. Two ways to read the hand:

  * marker mode: a glove with a fluorescent sticker on each fingertip; the
    stickers show when the hand is open and hide inside a fist (no AI model).
  * hand mode: hand landmarks come from MediaPipe's HandLandmarker (bare hand).

Keyboard: SPACE also flaps (handy for testing), G switches marker / hand mode,
C learns the sticker colour, V enlarges the camera view, S saves a camera
snapshot, F11 toggles fullscreen, ESC quits.

The best score is stored with its date and time in records.csv.
"""

import csv
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CAMERA_INDEX = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
RECORDS_PATH = os.path.join(BASE_DIR, "records.csv")

FULLSCREEN = True         # start in fullscreen (F11 toggles)
ORIENTATION = "auto"      # "auto" (follow the screen), "vertical" or "horizontal"
SHORT_SIDE = 480          # logical size of the game's short side; the long
LANDSCAPE_HEIGHT = 720    # side is computed at startup from the screen's aspect
WIDTH, HEIGHT = SHORT_SIDE, LANDSCAPE_HEIGHT
FPS = 60
GROUND_H = 80

GRAVITY = 1500.0        # px/s^2
FLAP_VELOCITY = -460.0  # px/s
MAX_FALL_SPEED = 700.0

PIPE_WIDTH = 80
PIPE_GAP = 200          # generous: a hand gesture is slower than a key press
PIPE_SPEED = 170.0      # px/s
PIPE_INTERVAL = 1.8     # seconds between pipes

PREVIEW_W, PREVIEW_H = 192, 144

TRACKING_MODE = "markers"  # "markers" (stickers on the glove) or "hand" (MediaPipe); G toggles
STABLE_FRAMES = 2          # a pose must hold this many frames before it counts

# Hand mode: hysteresis on the number of extended fingers
CLOSED_MAX_FINGERS = 1  # <= this many extended fingers -> fist
OPEN_MIN_FINGERS = 3    # >= this many extended fingers -> open hand

# Marker mode: a fluorescent sticker on the pad of each fingertip (palm side,
# facing the camera). Open hand: the camera sees them; fist: the fingertips
# fold into the palm and the stickers hide. The colour is a hue range in
# OpenCV's HSV (hue 0-179) plus minimum saturation and brightness. Click a
# sticker in the camera view (or hold it in the white square in the middle and
# press C) to learn them from it; they are saved in MARKERS_PATH.
MARKER_HUE = 165             # defaults until calibrated: fluorescent pink
MARKER_HUE_TOLERANCE = 10
MARKER_MIN_SATURATION = 100  # 0-255
MARKER_MIN_VALUE = 80        # 0-255 (brightness)
MARKER_MIN_AREA = 40         # px in the 640x480 camera image; smaller specks are noise
MARKER_MAX_AREA = 4000       # px; a bigger patch of that colour is not a sticker
MARKER_OPEN_MIN = 3          # >= this many stickers visible -> open hand
MARKER_FIST_MAX = 1          # <= this many -> fist (also when the hand leaves the view)
MARKER_SMOOTH_FRAMES = 3     # count = median of the last N frames, filters flicker
MARKER_TARGET = 16           # px; side of the calibration square
MARKER_SAMPLE_RADIUS = 4     # px; calibration samples a (2r+1)^2 patch of the sticker
MARKERS_PATH = os.path.join(BASE_DIR, "marcadores.json")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "capturas")  # S saves camera frames here

# MediaPipe hand landmark indices
WRIST = 0
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, pip) for index..pinky
THUMB_TIP, THUMB_IP, PINKY_MCP = 4, 3, 17
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

OPEN_COLOR, CLOSED_COLOR, PARTIAL_COLOR = (80, 255, 120), (255, 80, 80), (255, 210, 60)
POSE_COLORS = {"open": OPEN_COLOR, "closed": CLOSED_COLOR, "partial": PARTIAL_COLOR}


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Descargando el modelo de detección de manos (~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo guardado en", MODEL_PATH)


# ----------------------------------------------------------------------------
# High score file (records.csv: one row per new record, ';' separated so it
# opens in columns in a Spanish-locale Excel)
# ----------------------------------------------------------------------------
def load_record():
    """Return (best score, 'YYYY-MM-DD HH:MM:SS' or None)."""
    best, when = 0, None
    try:
        with open(RECORDS_PATH, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter=";"):
                try:
                    score = int(row["puntaje"])
                except (KeyError, TypeError, ValueError):
                    continue  # skip damaged rows
                if score > best:
                    best, when = score, row.get("fecha_hora")
    except OSError:
        pass
    return best, when


def save_record(score):
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_new = not os.path.exists(RECORDS_PATH)
    try:
        with open(RECORDS_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            if is_new:
                writer.writerow(["fecha_hora", "puntaje"])
            writer.writerow([when, score])
    except OSError as exc:
        print("No se pudo guardar el récord:", exc)
    return when


def format_when(when):
    try:
        return datetime.strptime(when, "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError):
        return when or ""


# ----------------------------------------------------------------------------
# Sticker colour (marcadores.json, written when calibrating with C)
# ----------------------------------------------------------------------------
def load_marker_color():
    color = {"hue": MARKER_HUE, "tolerance": MARKER_HUE_TOLERANCE,
             "min_saturation": MARKER_MIN_SATURATION, "min_value": MARKER_MIN_VALUE}
    try:
        with open(MARKERS_PATH, encoding="utf-8") as f:
            color.update({k: int(v) for k, v in json.load(f).items() if k in color})
    except (OSError, ValueError, TypeError, AttributeError):
        pass  # no calibration yet (or a damaged file): use the defaults
    return color


def save_marker_color(color):
    try:
        with open(MARKERS_PATH, "w", encoding="utf-8") as f:
            json.dump(color, f, indent=2)
    except OSError as exc:
        print("No se pudo guardar la calibración:", exc)


# ----------------------------------------------------------------------------
# Hand tracking (runs in a background thread so the game stays smooth)
# ----------------------------------------------------------------------------
class HandTracker(threading.Thread):
    def __init__(self, camera_index=CAMERA_INDEX):
        super().__init__(daemon=True)
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(camera_index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.available = self.cap.isOpened()

        self._lock = threading.Lock()
        self._running = True
        self._flaps = 0          # pending flap events not yet consumed by the game
        self.frame = None        # latest RGB frame with landmarks / stickers drawn
        self.mode = TRACKING_MODE
        self.marker_color = load_marker_color()
        self.hand_state = None   # last confirmed pose: "open", "closed" or None (no hand)
        self.pose = None         # current raw pose: "open", "closed", "partial" or None
        self.fingers = 0         # hand mode: extended fingers
        self.markers = 0         # marker mode: stickers visible
        self.openness = 0.0      # 0 = full fist, 1 = fully open
        self.notice = None       # (message, show until) for calibration feedback
        self._counts = deque(maxlen=MARKER_SMOOTH_FRAMES)
        self._calibrate = None   # camera pixel to learn the sticker colour from, or "center"
        self._snapshot = False

    def stop(self):
        self._running = False

    def toggle_mode(self):
        self.mode = "hand" if self.mode == "markers" else "markers"

    def calibrate(self, point=None):
        """Learn the sticker colour at point (x, y in the camera image) on the
        next frame; None = the white square in the middle."""
        self._calibrate = point or "center"

    def snapshot(self):
        self._snapshot = True  # saved on the next frame

    def consume_flaps(self):
        with self._lock:
            n, self._flaps = self._flaps, 0
        return n

    def get_preview(self):
        with self._lock:
            return self.frame

    @staticmethod
    def count_extended_fingers(pts):
        """pts: list of (x, y) pixel coordinates for the 21 landmarks."""
        def dist(a, b):
            return math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])

        # A finger is extended when its tip is farther from the wrist than its
        # middle joint. Using distances makes this work at any hand rotation.
        count = sum(1 for tip, pip in FINGERS if dist(tip, WRIST) > dist(pip, WRIST) * 1.1)
        # Thumb: tip far from the base of the pinky means it is sticking out.
        if dist(THUMB_TIP, PINKY_MCP) > dist(THUMB_IP, PINKY_MCP) * 1.1:
            count += 1
        return count

    def detect_hand(self, result, rgb):
        """Classify a MediaPipe result and draw the landmarks on rgb.
        Returns (pose, extended fingers)."""
        if not result.hand_landmarks:
            return None, 0
        h, w = rgb.shape[:2]
        pts = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
        fingers = self.count_extended_fingers(pts)
        if fingers <= CLOSED_MAX_FINGERS:
            pose = "closed"
        elif fingers >= OPEN_MIN_FINGERS:
            pose = "open"
        else:
            pose = "partial"

        ipts = [(int(x), int(y)) for x, y in pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(rgb, ipts[a], ipts[b], POSE_COLORS[pose], 3)
        for p in ipts:
            cv2.circle(rgb, p, 5, (255, 255, 255), -1)
        return pose, fingers

    def detect_markers(self, frame, rgb):
        """Count the coloured stickers in the BGR frame and circle them on rgb.
        Returns (pose, stickers visible)."""
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        if self._calibrate:
            point, self._calibrate = self._calibrate, None
            self.calibrate_marker(hsv, *((w // 2, h // 2) if point == "center" else point))

        stickers, ignored = self.find_stickers(hsv)
        self._counts.append(len(stickers))
        count = int(np.median(self._counts))
        if count >= MARKER_OPEN_MIN:
            pose = "open"
        elif count <= MARKER_FIST_MAX:
            pose = "closed"
        else:
            pose = "partial"

        # Stickers circled in the pose colour; same-coloured spots that are too
        # small or too big in grey, so stray objects are easy to spot
        for c, color in [(c, POSE_COLORS[pose]) for c in stickers] + [(c, (150, 150, 150)) for c in ignored]:
            if cv2.contourArea(c) < 4:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            cv2.circle(rgb, (int(cx), int(cy)), int(r) + 6, color, 3)
        half = MARKER_TARGET // 2
        cv2.rectangle(rgb, (w // 2 - half, h // 2 - half), (w // 2 + half, h // 2 + half), (255, 255, 255), 2)
        return pose, count

    def find_stickers(self, hsv):
        """Blobs of the sticker colour: (sticker-sized, too small or too big)."""
        col = self.marker_color
        hue_diff = np.abs(hsv[:, :, 0].astype(np.int16) - col["hue"])
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)  # hue is circular: 179 is next to 0
        mask = ((hue_diff <= col["tolerance"]) & (hsv[:, :, 1] >= col["min_saturation"])
                & (hsv[:, :, 2] >= col["min_value"])).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stickers, ignored = [], []
        for c in contours:
            (stickers if MARKER_MIN_AREA <= cv2.contourArea(c) <= MARKER_MAX_AREA else ignored).append(c)
        return stickers, ignored

    def calibrate_marker(self, hsv, x, y):
        """Learn the sticker colour from a small patch around camera pixel (x, y)."""
        h, w = hsv.shape[:2]
        r = MARKER_SAMPLE_RADIUS
        x, y = min(max(int(x), r), w - r - 1), min(max(int(y), r), h - r - 1)
        patch = hsv[y - r:y + r + 1, x - r:x + r + 1].reshape(-1, 3).astype(np.float64)
        sat, val = np.percentile(patch[:, 1], 10), np.percentile(patch[:, 2], 10)
        if sat < 40 or val < 100:  # fluorescent stickers are vivid and bright; the glove isn't
            self.notice = ("Ahí no hay un color vivo", time.monotonic() + 4)
            print("Calibración ignorada: haz clic justo sobre el sticker")
            return
        # Circular mean of the hue, so reds around 179/0 average correctly
        angle = patch[:, 0] * np.pi / 90
        hue = int(round(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()) * 90 / np.pi)) % 180
        hue_diff = np.abs(patch[:, 0] - hue)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        # Ranges just wide enough for the sticker (with margin for it turning
        # or moving away from the light), so the background stays out
        self.marker_color = {"hue": hue,
                             "tolerance": int(np.clip(np.percentile(hue_diff, 90) + 4, 5, 15)),
                             "min_saturation": int(max(30, sat * 0.75)),
                             "min_value": int(max(60, val * 0.8))}
        save_marker_color(self.marker_color)
        print("Color de los marcadores calibrado:", self.marker_color)

        others = [c for c in self.find_stickers(hsv)[0] if cv2.pointPolygonTest(c, (x, y), False) < 0]
        if others:
            self.notice = (f"Ojo: {len(others)} manchas de ese color fuera del sticker", time.monotonic() + 6)
        else:
            self.notice = (f"Color calibrado (tono {hue})", time.monotonic() + 4)

    def save_snapshot(self, frame, rgb, pose, value):
        """Save the raw camera frame and the annotated view, to tune the
        detection offline."""
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{self.mode}_{pose}_{value}"
        cv2.imwrite(os.path.join(SNAPSHOT_DIR, name + "_raw.png"), frame)
        cv2.imwrite(os.path.join(SNAPSHOT_DIR, name + "_vista.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print("Captura guardada:", name)

    def run(self):
        if not self.available:
            return
        landmarker = None
        if os.path.exists(MODEL_PATH):
            options = vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)
        t0 = time.monotonic()
        last_ts = -1
        mode, state, last_pose, stable = self.mode, None, None, 0
        try:
            while self._running:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)  # mirror, feels natural
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.mode != mode:  # switched with G: start over
                    mode, state, last_pose, stable = self.mode, None, None, 0
                    self._counts.clear()

                pose, fingers, markers, openness = None, 0, 0, 0.0
                if mode == "markers":
                    pose, markers = self.detect_markers(frame, rgb)
                    openness = min(1.0, markers / 5)
                elif landmarker:
                    ts = max(int((time.monotonic() - t0) * 1000), last_ts + 1)
                    last_ts = ts
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
                    )
                    pose, fingers = self.detect_hand(result, rgb)
                    openness = fingers / 5

                stable = stable + 1 if pose == last_pose else 1
                last_pose = pose
                # Only a fully open or fully closed hand that holds for a few
                # frames changes the state, so one bad frame can't flap.
                flapped = False
                if pose != "partial" and pose != state and stable >= STABLE_FRAMES:
                    flapped = state == "open" and pose == "closed"
                    state = pose

                if self._snapshot:
                    self._snapshot = False
                    self.save_snapshot(frame, rgb, pose, markers if mode == "markers" else fingers)

                with self._lock:
                    if flapped:
                        self._flaps += 1
                    self.hand_state = state
                    self.pose = pose
                    self.fingers = fingers
                    self.markers = markers
                    self.openness = openness
                    self.frame = rgb
        finally:
            if landmarker:
                landmarker.close()
            self.cap.release()


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------
class Bird:
    RADIUS = 18

    def __init__(self):
        self.x = WIDTH * 0.3
        self.y = HEIGHT * 0.45
        self.vy = 0.0

    def flap(self):
        self.vy = FLAP_VELOCITY

    def update(self, dt):
        self.vy = min(self.vy + GRAVITY * dt, MAX_FALL_SPEED)
        self.y += self.vy * dt

    def draw(self, surf):
        angle = max(-30, min(70, self.vy / 10))
        body = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(body, (255, 214, 0), (30, 30), self.RADIUS)
        pygame.draw.circle(body, (60, 40, 0), (30, 30), self.RADIUS, 2)
        pygame.draw.ellipse(body, (255, 245, 180), (14, 30, 18, 10))       # wing
        pygame.draw.circle(body, (255, 255, 255), (38, 24), 6)              # eye
        pygame.draw.circle(body, (0, 0, 0), (40, 24), 3)
        pygame.draw.polygon(body, (255, 120, 0), [(44, 30), (56, 34), (44, 38)])  # beak
        rotated = pygame.transform.rotate(body, -angle)
        surf.blit(rotated, rotated.get_rect(center=(int(self.x), int(self.y))))


class Pipe:
    def __init__(self):
        self.x = float(WIDTH)
        margin = 60
        self.gap_y = random.randint(margin + PIPE_GAP // 2, HEIGHT - GROUND_H - margin - PIPE_GAP // 2)
        self.passed = False

    def rects(self):
        top = pygame.Rect(int(self.x), 0, PIPE_WIDTH, self.gap_y - PIPE_GAP // 2)
        bottom_y = self.gap_y + PIPE_GAP // 2
        bottom = pygame.Rect(int(self.x), bottom_y, PIPE_WIDTH, HEIGHT - GROUND_H - bottom_y)
        return top, bottom

    def draw(self, surf):
        for r in self.rects():
            pygame.draw.rect(surf, (90, 190, 60), r)
            pygame.draw.rect(surf, (40, 110, 30), r, 3)
            lip_y = r.bottom - 26 if r.top == 0 else r.top
            lip = pygame.Rect(r.x - 6, lip_y, r.width + 12, 26)
            pygame.draw.rect(surf, (100, 210, 70), lip)
            pygame.draw.rect(surf, (40, 110, 30), lip, 3)


def bird_hits(bird, rect):
    cx = max(rect.left, min(bird.x, rect.right))
    cy = max(rect.top, min(bird.y, rect.bottom))
    return (bird.x - cx) ** 2 + (bird.y - cy) ** 2 < (Bird.RADIUS - 2) ** 2


def game_size(desk_w, desk_h):
    """Logical game size with the screen's aspect ratio, so fullscreen has no
    black bars on either a landscape or a portrait (phone-like) monitor."""
    vertical = desk_h > desk_w if ORIENTATION == "auto" else ORIENTATION == "vertical"
    if vertical:
        aspect = max(desk_h, desk_w) / min(desk_h, desk_w)
        return SHORT_SIDE, round(SHORT_SIDE * aspect)
    aspect = max(desk_w, desk_h) / min(desk_w, desk_h)
    return max(SHORT_SIDE, round(LANDSCAPE_HEIGHT * aspect)), LANDSCAPE_HEIGHT


class Game:
    def __init__(self, tracker):
        global WIDTH, HEIGHT
        pygame.init()
        pygame.display.set_caption("Flappy Hand")
        # SCALED stretches the logical game surface to whatever the display is.
        WIDTH, HEIGHT = game_size(*pygame.display.get_desktop_sizes()[0])
        flags = pygame.SCALED | (pygame.FULLSCREEN if FULLSCREEN else 0)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("arial", 52, bold=True)
        self.font = pygame.font.SysFont("arial", 22, bold=True)
        self.font_small = pygame.font.SysFont("arial", 16)
        self.tracker = tracker
        self.best, self.best_when = load_record()
        self.new_record = False
        self.big_preview = False  # V: double-size camera view, handy to calibrate
        self.preview_rect = None  # where the camera view was drawn (for clicks)
        self.ground_offset = 0.0
        self.reset()

    def reset(self):
        self.bird = Bird()
        self.pipes = []
        self.spawn_timer = 0.0
        self.score = 0
        self.state = "ready"  # ready -> playing -> over
        self.over_time = 0.0
        self.new_record = False

    def on_flap(self):
        if self.state == "ready":
            self.state = "playing"
            self.bird.flap()
        elif self.state == "playing":
            self.bird.flap()
        elif self.state == "over" and time.monotonic() - self.over_time > 1.0:
            self.reset()

    def update(self, dt):
        if self.state == "ready":
            self.bird.y = HEIGHT * 0.45 + math.sin(time.monotonic() * 4) * 8
        if self.state != "over":
            self.ground_offset = (self.ground_offset + PIPE_SPEED * dt) % 40
        if self.state != "playing":
            if self.state == "over" and self.bird.y < HEIGHT - GROUND_H - Bird.RADIUS:
                self.bird.update(dt)  # let the bird fall to the ground
            return

        self.bird.update(dt)
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self.pipes.append(Pipe())
            self.spawn_timer = PIPE_INTERVAL
        for p in self.pipes:
            p.x -= PIPE_SPEED * dt
            if not p.passed and p.x + PIPE_WIDTH < self.bird.x:
                p.passed = True
                self.score += 1
        self.pipes = [p for p in self.pipes if p.x + PIPE_WIDTH > -20]

        hit_pipe = any(bird_hits(self.bird, r) for p in self.pipes for r in p.rects())
        hit_ground = self.bird.y + Bird.RADIUS >= HEIGHT - GROUND_H
        if self.bird.y < -Bird.RADIUS:
            self.bird.y, self.bird.vy = -Bird.RADIUS, 0
        if hit_pipe or hit_ground:
            self.state = "over"
            self.over_time = time.monotonic()
            if self.score > self.best:
                self.best = self.score
                self.best_when = save_record(self.score)
                self.new_record = True

    def draw_text(self, text, font, y, color=(255, 255, 255)):
        shadow = font.render(text, True, (0, 0, 0))
        label = font.render(text, True, color)
        x = WIDTH // 2 - label.get_width() // 2
        self.screen.blit(shadow, (x + 2, int(y) + 2))
        self.screen.blit(label, (x, int(y)))

    def on_preview_click(self, pos):
        """A click on a sticker in the camera view learns its colour."""
        frame = self.tracker.get_preview() if self.tracker else None
        r = self.preview_rect
        if frame is None or r is None or not r.collidepoint(pos) or self.tracker.mode != "markers":
            return
        h, w = frame.shape[:2]
        self.tracker.calibrate(((pos[0] - r.x) * w / r.width, (pos[1] - r.y) * h / r.height))

    def draw_preview(self):
        frame = self.tracker.get_preview() if self.tracker else None
        scale = 2 if self.big_preview else 1
        pw, ph = PREVIEW_W * scale, PREVIEW_H * scale
        x, y = WIDTH - pw - 10, 10
        self.preview_rect = pygame.Rect(x, y, pw, ph)
        if frame is not None:
            h, w = frame.shape[:2]
            surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            self.screen.blit(pygame.transform.smoothscale(surf, (pw, ph)), (x, y))
        else:
            pygame.draw.rect(self.screen, (30, 30, 30), (x, y, pw, ph))
            msg = "Sin cámara" if not (self.tracker and self.tracker.available) else "Iniciando..."
            t = self.font_small.render(msg, True, (220, 220, 220))
            self.screen.blit(t, (x + 10, y + ph // 2 - 8))
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, pw, ph), 2)

        if not (self.tracker and self.tracker.available):
            return
        pose = self.tracker.pose
        text, color = {
            "open": ("ABIERTA", OPEN_COLOR),
            "closed": ("PUÑO", CLOSED_COLOR),
            "partial": ("A medias", PARTIAL_COLOR),
            None: ("Sin mano", (220, 220, 220)),
        }[pose]
        # Openness bar: empty = full fist, full = fully open hand
        bar_y = y + ph + 6
        pygame.draw.rect(self.screen, (30, 30, 30), (x, bar_y, pw, 10))
        if pose is not None:
            pygame.draw.rect(self.screen, color, (x, bar_y, int(pw * self.tracker.openness), 10))
        pygame.draw.rect(self.screen, (255, 255, 255), (x, bar_y, pw, 10), 1)

        if self.tracker.mode == "markers":
            notice = self.tracker.notice
            second = (notice[0] if notice and time.monotonic() < notice[1]
                      else "Clic en un sticker: calibrar color")
            lines = [(f"{text}  ({self.tracker.markers} marcas)", color),
                     (second, (220, 220, 220))]
        else:
            lines = [(f"{text}  ({self.tracker.fingers} dedos)", color),
                     ("Mano (MediaPipe)", (220, 220, 220))]
        for i, (line, line_color) in enumerate(lines):
            t = self.font_small.render(line, True, line_color)
            self.screen.blit(t, (x + pw - t.get_width(), bar_y + 14 + i * 18))

    def draw(self):
        self.screen.fill((112, 197, 206))
        for p in self.pipes:
            p.draw(self.screen)

        ground_y = HEIGHT - GROUND_H
        pygame.draw.rect(self.screen, (222, 216, 149), (0, ground_y, WIDTH, GROUND_H))
        pygame.draw.rect(self.screen, (115, 190, 45), (0, ground_y, WIDTH, 14))
        for gx in range(-40, WIDTH + 40, 40):
            sx = gx - int(self.ground_offset)
            pygame.draw.polygon(self.screen, (95, 165, 35),
                                [(sx, ground_y + 14), (sx + 20, ground_y + 14), (sx + 10, ground_y)])

        self.bird.draw(self.screen)

        best_text = f"Récord: {self.best}"
        if self.best_when:
            best_text += f"  ({format_when(self.best_when)})"

        if self.state == "playing":
            self.draw_text(str(self.score), self.font_big, 40)
        elif self.state == "ready":
            self.draw_text("FLAPPY HAND", self.font_big, HEIGHT * 0.30, (255, 230, 0))
            if self.best:
                self.draw_text(best_text, self.font_small, HEIGHT * 0.30 + 64)
            self.draw_text("Abre la mano", self.font, HEIGHT * 0.60)
            self.draw_text("¡y ciérrala para saltar!", self.font, HEIGHT * 0.60 + 30)
            self.draw_text("F11: pantalla completa   ESC: salir", self.font_small, HEIGHT * 0.60 + 80)
            self.draw_text("Clic en sticker: calibrar   G: marcas / mano", self.font_small, HEIGHT * 0.60 + 102)
            self.draw_text("V: vista grande   S: guardar captura", self.font_small, HEIGHT * 0.60 + 124)
        else:
            self.draw_text("FIN DEL JUEGO", self.font_big, HEIGHT * 0.30, (255, 90, 60))
            self.draw_text(f"Puntos: {self.score}", self.font, HEIGHT * 0.30 + 76)
            if self.new_record:
                self.draw_text("¡NUEVO RÉCORD!", self.font, HEIGHT * 0.30 + 108, (255, 230, 0))
            else:
                self.draw_text(best_text, self.font_small, HEIGHT * 0.30 + 110)
            if time.monotonic() - self.over_time > 1.0:
                self.draw_text("Abre y cierra la mano", self.font, HEIGHT * 0.60)
                self.draw_text("para reintentar", self.font, HEIGHT * 0.60 + 30)

        self.draw_preview()
        pygame.display.flip()

    def run(self):
        while True:
            dt = min(self.clock.tick(FPS) / 1000.0, 1 / 20)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.on_preview_click(event.pos)
                    continue
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    self.on_flap()
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif event.key == pygame.K_v:
                    self.big_preview = not self.big_preview
                elif self.tracker and event.key == pygame.K_g:
                    self.tracker.toggle_mode()
                elif self.tracker and event.key == pygame.K_c:
                    self.tracker.calibrate()
                elif self.tracker and event.key == pygame.K_s:
                    self.tracker.snapshot()
            if self.tracker:
                for _ in range(self.tracker.consume_flaps()):
                    self.on_flap()
            self.update(dt)
            self.draw()


def main():
    tracker = None
    try:
        ensure_model()
    except Exception as exc:  # e.g. no internet: marker mode doesn't need the model
        print("Sin modelo de MediaPipe, el modo mano no funcionará:", exc)
    try:
        tracker = HandTracker()
        if tracker.available:
            tracker.start()
        else:
            print("No se pudo abrir la cámara - solo teclado (ESPACIO para saltar).")
    except Exception as exc:  # e.g. no internet to download the model
        print("Detección de manos desactivada:", exc)
        tracker = None

    try:
        Game(tracker).run()
    finally:
        pygame.quit()
        if tracker and tracker.is_alive():
            tracker.stop()
            tracker.join(timeout=2)
            # MediaPipe / the camera driver can block interpreter shutdown;
            # exit hard so ESC always closes the program.
            sys.stdout.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
