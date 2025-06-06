import cv2
import os
import time
import json
import random
import pygame
import subprocess
import threading
import sys

# --- Load config ---
with open("config.json") as f:
    config = json.load(f)

EXIT_DELAY = config["exit_delay_seconds"]
FRAME_RATE = config["frame_rate"]
FADE_STEPS = config["fade_steps"]
VIDEO_DIR = config["video_directory"]
BLACK_IMAGE_PATH = config["black_overlay"]
PERSON_DETECTION_SKIP_INTERVAL = config.get("person_detection_skip_interval", 5)

# --- Global state ---
person_detected = False
last_seen = 0
vlc_process = None
person_detection_frame_count = 0

# --- Initialize pygame fullscreen window ---
pygame.init()
pygame.display.set_caption("Magic Mirror")
info = pygame.display.Info()
screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.NOFRAME | pygame.FULLSCREEN)
clock = pygame.time.Clock()

# Load and scale black overlay
black_overlay = pygame.image.load(BLACK_IMAGE_PATH).convert()
black_overlay = pygame.transform.scale(black_overlay, (info.current_w, info.current_h))
black_overlay.set_alpha(255)
black_alpha = 255
fade_step = 255 // FADE_STEPS

# --- Helper Functions ---
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

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_person(frame):
    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Run HOG person detector
    (regions, _) = hog.detectMultiScale(frame_resized,
                                        winStride=(8, 8),
                                        padding=(8, 8),
                                        scale=1.05)

    return len(regions) > 0

def play_video_as_user(video_path):
    user = os.getenv("SUDO_USER") or os.getenv("USER")
    if not user:
        print("Error: Can't determine non-root user to run VLC.")
        return None

    return subprocess.Popen([
        "sudo", "-u", user, "vlc",
        "--no-video-title-show",
        "--no-video-deco",
        "--fullscreen",
        "--video-on-top",
        "--loop",
        "--quiet",
        video_path
    ])

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

# --- Camera Setup ---
print("Waiting for camera to become available...")
cap = None

try:
    for i in range(10):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)

        if cap.isOpened():
            print(f"Camera found at index {i}")
            break
        time.sleep(1)
    raise RuntimeError("No working camera found.")
except Exception as e:
    pygame.quit()
    sys.exit(1)

if not cap or not cap.isOpened():
    raise RuntimeError("Camera failed to open after multiple attempts.")

# --- Main Loop ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read from camera.")
            time.sleep(1)
            continue

        now = time.time()
        detected = detect_person(frame)

        if detected:
            if not person_detected:
                print("Person detected — playing video")
                play_random_video_with_vlc()
            person_detected = True
            last_seen = now
            fade_from_black()
        else:
            if person_detected and now - last_seen > EXIT_DELAY:
                print("No person detected — fading out")
                if vlc_process:
                    vlc_process.terminate()
                    vlc_process = None
                person_detected = False
            fade_to_black()

        screen.blit(black_overlay, (0, 0))
        pygame.display.flip()
        clock.tick(FRAME_RATE)

except KeyboardInterrupt:
    print("Exiting cleanly...")
    if cap:
        cap.release()
    if vlc_process:
        vlc_process.terminate()
    pygame.quit()
    exit()
