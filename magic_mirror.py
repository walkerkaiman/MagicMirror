import cv2
import os
import time
import json
import random
import pygame
import subprocess
import threading

# --- Load config ---
with open("config.json") as f:
    config = json.load(f)

EXIT_DELAY = config["exit_delay_seconds"]
FRAME_RATE = config["frame_rate"]
FADE_STEPS = config["fade_steps"]
VIDEO_DIR = config["video_directory"]
BLACK_IMAGE_PATH = config["black_overlay"]

# --- Global state ---
person_detected = False
last_seen = 0
vlc_process = None

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

def detect_person(frame):
    # Placeholder logic: check brightness in the center of the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center_region = gray[h//3:2*h//3, w//3:2*w//3]
    avg_brightness = cv2.mean(center_region)[0]
    return avg_brightness > 50  # very basic trigger

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
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open.")

# --- Main Loop ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
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
