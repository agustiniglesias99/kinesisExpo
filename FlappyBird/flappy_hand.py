"""Flappy Bird controlled by your hand.

Open your hand, then close it into a fist: every open -> closed transition
makes the bird flap. The webcam is read with OpenCV, hand landmarks come from
MediaPipe's HandLandmarker and the game is drawn with pygame.

Keyboard: SPACE also flaps (handy for testing), F11 toggles fullscreen,
ESC quits.

The best score is stored with its date and time in records.csv.
"""

import csv
import math
import os
import random
import sys
import threading
import time
import urllib.request
from datetime import datetime

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CAMERA_INDEX = 1
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

# Hysteresis for gesture detection (number of extended fingers)
CLOSED_MAX_FINGERS = 1  # <= this many extended fingers -> fist
OPEN_MIN_FINGERS = 3    # >= this many extended fingers -> open hand

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

OPEN_COLOR, CLOSED_COLOR = (80, 255, 120), (255, 80, 80)


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
        self.frame = None        # latest RGB frame with landmarks drawn
        self.hand_state = None   # "open", "closed" or None (no hand)
        self.fingers = 0

    def stop(self):
        self._running = False

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

    def run(self):
        if not self.available:
            return
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        t0 = time.monotonic()
        last_ts = -1
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while self._running:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)  # mirror, feels natural
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]

                ts = max(int((time.monotonic() - t0) * 1000), last_ts + 1)
                last_ts = ts
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
                )

                state = self.hand_state
                fingers = 0
                if result.hand_landmarks:
                    pts = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
                    fingers = self.count_extended_fingers(pts)
                    if fingers <= CLOSED_MAX_FINGERS:
                        new_state = "closed"
                    elif fingers >= OPEN_MIN_FINGERS:
                        new_state = "open"
                    else:
                        new_state = state  # in between: keep previous state
                    flapped = state == "open" and new_state == "closed"
                    state = new_state

                    color = CLOSED_COLOR if state == "closed" else OPEN_COLOR
                    ipts = [(int(x), int(y)) for x, y in pts]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(rgb, ipts[a], ipts[b], color, 3)
                    for p in ipts:
                        cv2.circle(rgb, p, 5, (255, 255, 255), -1)
                else:
                    flapped = False
                    state = None

                with self._lock:
                    if flapped:
                        self._flaps += 1
                    self.hand_state = state
                    self.fingers = fingers
                    self.frame = rgb
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

    def draw_preview(self):
        frame = self.tracker.get_preview() if self.tracker else None
        x, y = WIDTH - PREVIEW_W - 10, 10
        if frame is not None:
            h, w = frame.shape[:2]
            surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            self.screen.blit(pygame.transform.smoothscale(surf, (PREVIEW_W, PREVIEW_H)), (x, y))
        else:
            pygame.draw.rect(self.screen, (30, 30, 30), (x, y, PREVIEW_W, PREVIEW_H))
            msg = "Sin cámara" if not (self.tracker and self.tracker.available) else "Iniciando..."
            t = self.font_small.render(msg, True, (220, 220, 220))
            self.screen.blit(t, (x + 10, y + PREVIEW_H // 2 - 8))
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, PREVIEW_W, PREVIEW_H), 2)

        if self.tracker and self.tracker.available:
            state = self.tracker.hand_state
            text, color = {
                "open": ("Mano ABIERTA", OPEN_COLOR),
                "closed": ("PUÑO", CLOSED_COLOR),
                None: ("Sin mano", (220, 220, 220)),
            }[state]
            t = self.font_small.render(f"{text}  ({self.tracker.fingers} dedos)", True, color)
            self.screen.blit(t, (x, y + PREVIEW_H + 4))

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
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    self.on_flap()
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
            if self.tracker:
                for _ in range(self.tracker.consume_flaps()):
                    self.on_flap()
            self.update(dt)
            self.draw()


def main():
    tracker = None
    try:
        ensure_model()
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
