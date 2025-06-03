import cv2
import mediapipe as mp
import os
import time
import json
import random
import board
import neopixel
import threading
import subprocess
import pygame

# --- Load config ---
with open("config.json") as f:
    config = json.load(f)

EXIT_DELAY = config["exit_delay_seconds"]
FRAME_RATE = config["frame_rate"]
FADE_STEPS = config["fade_steps"]
VIDEO_DIR = config["video_directory"]
LED_PIN = board.D18  # Typically GPIO18
LED_COUNT = config["led_count"]
LED_COLOR = tuple(config["led_color"])
LED_DELAY = config["led_transition_delay_seconds"]

# --- Initialize LEDs ---
pixels = neopixel.NeoPixel(LED_PIN, LED_COUNT, auto_write=False)

def animate_leds_off():
    for offset in range(LED_COUNT // 2 + 1):
        left = offset
        right = LED_COUNT - offset - 1
        if left < LED_COUNT: pixels[left] = (0, 0, 0)
        if right >= 0: pixels[right] = (0, 0, 0)
        pixels.show()
        time.sleep(LED_DELAY)

# --- VLC control ---
vlc_process = None

def play_random_video_with_vlc():
    global vlc_process

    if vlc_process and vlc_process.poll() is None:
        vlc_process.terminate()

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))]
    if not video_files:
        print("No videos found in", VIDEO_DIR)
        return

    selected = os.path.join(VIDEO_DIR, random.choice(video_files))
    print("Launching VLC for:", selected)

    vlc_process = play_video_as_user(selected)

def play_video_as_user(video_path):
    user = os.getenv("SUDO_USER") or os.getenv("USER")
    return subprocess.Popen([
        "sudo", "-u", user, "vlc",
        "--vout=egl",  # or try --vout=directfb or --vout=mmal for Pi
        "--no-video-title-show",
        "--no-video-deco",
        "--fullscreen",
        "--video-on-top",
        "--loop",
        "--quiet",
        video_path
    ])

def stop_vlc():
    global vlc_process
    if vlc_process and vlc_process.poll() is None:
        print("Stopping VLC")
        vlc_process.terminate()
        vlc_process = None

# --- Setup detection ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
last_seen = 0
playing = False
tracker = None

# --- Initialize fade overlay window ---
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.NOFRAME | pygame.FULLSCREEN)
clock = pygame.time.Clock()
screen_width, screen_height = screen.get_size()

# Black overlay surface
black_overlay = pygame.Surface((screen_width, screen_height))
black_overlay.fill((0, 0, 0))
black_alpha = 255
fade_step = 255 // FADE_STEPS

def fade_to_black():
    global black_alpha
    if black_alpha < 255:
        black_alpha = min(255, black_alpha + fade_step)
    black_overlay.set_alpha(black_alpha)

def fade_from_black():
    global black_alpha
    if black_alpha > 0:
        black_alpha = max(0, black_alpha - fade_step)
    black_overlay.set_alpha(black_alpha)

# --- Detection logic ---
def detect_person(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        h, w = frame.shape[:2]
        xs = [lm.x for lm in results.pose_landmarks.landmark]
        ys = [lm.y for lm in results.pose_landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        box = (
            int(x_min * w),
            int(y_min * h),
            int((x_max - x_min) * w),
            int((y_max - y_min) * h)
        )
        return box
    return None

# --- Main loop ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            stop_vlc()
            exit()

    ret, cam_frame = cap.read()
    if not ret:
        continue

    now = time.time()

    if tracker:
        success, _ = tracker.update(cam_frame)
        if success:
            last_seen = now
        else:
            tracker = None

    if not tracker:
        new_box = detect_person(cam_frame)
        if new_box:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(cam_frame, new_box)
            last_seen = now
            playing = True
            black_alpha = 255
            play_random_video_with_vlc()
            print("Person detected — playing video")

    if playing and (now - last_seen > EXIT_DELAY):
        playing = False
        tracker = None
        stop_vlc()
        print("Person left — fading to black")
        threading.Thread(target=animate_leds_off).start()

    if playing:
        fade_from_black()
    else:
        fade_to_black()

    screen.fill((0, 0, 0))
    screen.blit(black_overlay, (0, 0))
    pygame.display.flip()
    clock.tick(FRAME_RATE)
