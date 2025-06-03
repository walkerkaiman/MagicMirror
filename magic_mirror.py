import cv2
import mediapipe as mp
import pygame
import os
import time
import json
import random
import board
import neopixel
import threading
import numpy as np

# --- Load config ---
with open("config.json") as f:
    config = json.load(f)

EXIT_DELAY = config["exit_delay_seconds"]
FRAME_RATE = config["frame_rate"]
FADE_STEPS = config["fade_steps"]
VIDEO_DIR = config["video_directory"]
BLACK_IMAGE_PATH = config["black_overlay"]
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

# --- Initialize display ---
sample_video = cv2.VideoCapture(os.path.join(VIDEO_DIR, os.listdir(VIDEO_DIR)[0]))
frame_width = int(sample_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(sample_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
sample_video.release()

pygame.init()
screen = pygame.display.set_mode((frame_width, frame_height))
clock = pygame.time.Clock()

black_img = pygame.image.load(BLACK_IMAGE_PATH).convert()
black_img = pygame.transform.scale(black_img, (frame_width, frame_height))
black_img.set_alpha(255)
black_alpha = 255
fade_step = 255 // FADE_STEPS

# --- Setup detection ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
last_seen = 0
playing = False
tracker = None
video_player = None
video_file = None

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

def fade_to_black():
    global black_alpha
    black_alpha = min(255, black_alpha + fade_step)
    black_img.set_alpha(black_alpha)

def fade_from_black():
    global black_alpha
    black_alpha = max(0, black_alpha - fade_step)
    black_img.set_alpha(black_alpha)

def get_random_video_path():
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".mov", ".avi"))]
    return os.path.join(VIDEO_DIR, random.choice(videos)) if videos else None

def open_video(path):
    return cv2.VideoCapture(path)

def is_not_grayscale(color, threshold=30):
    r, g, b = color
    return max(r, g, b) - min(r, g, b) > threshold

def get_gradient_colors_from_frame(frame):
    h, w, _ = frame.shape
    selected_colors = []
    attempts = 0
    while len(selected_colors) < 3 and attempts < 100:
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        color = tuple(map(int, frame[y, x]))
        if is_not_grayscale(color):
            selected_colors.append(color)
        attempts += 1
    if len(selected_colors) < 3:
        selected_colors += [LED_COLOR] * (3 - len(selected_colors))
    return selected_colors

def apply_gradient_to_leds(colors):
    def interpolate(c1, c2, t):
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    segment_len = LED_COUNT // 2
    for i in range(segment_len):
        t = i / segment_len
        pixels[i] = interpolate(colors[0], colors[1], t)
    for i in range(segment_len, LED_COUNT):
        t = (i - segment_len) / segment_len
        pixels[i] = interpolate(colors[1], colors[2], t)
    pixels.show()

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                if video_player: video_player.release()
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
                video_file = get_random_video_path()
                video_player = open_video(video_file)
                ret, first_frame = video_player.read()
                if ret:
                    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                    gradient_colors = get_gradient_colors_from_frame(first_frame)
                    apply_gradient_to_leds(gradient_colors)
                print(f"Person detected — playing: {os.path.basename(video_file)}")

        if playing and (now - last_seen > EXIT_DELAY):
            playing = False
            tracker = None
            if video_player: video_player.release()
            video_player = None
            print("Person left — fading to black.")
            threading.Thread(target=animate_leds_off).start()

        if playing and video_player:
            ret, vid_frame = video_player.read()
            if not ret:
                video_player.release()
                video_player = open_video(video_file)
                ret, vid_frame = video_player.read()

            if ret:
                vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
                surface = pygame.surfarray.make_surface(vid_frame.swapaxes(0, 1))
                surface = pygame.transform.scale(surface, (frame_width, frame_height))
                screen.blit(surface, (0, 0))
                fade_from_black()
        else:
            screen.blit(black_img, (0, 0))
            fade_to_black()

        screen.blit(black_img, (0, 0))
        pygame.display.flip()
        clock.tick(FRAME_RATE)
except KeyboardInterrupt:
    print("Shutting down gracefully...")
    if video_player:
        video_player.release()
    cap.release()
    pygame.quit()
    pixels.fill((0, 0, 0))
    pixels.show()
    exit(0)