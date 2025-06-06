import cv2
import os
import random
import time
import signal
import subprocess
import threading
import mediapipe as mp

# Config
VIDEO_FOLDER = "videos"
FADE_OUT_DURATION = 2  # seconds
CAMERA_INDEX = 0
PERSON_LOST_TIMEOUT = 2  # seconds

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Globals
current_process = None
last_seen_time = 0
person_present = False


def handle_exit(signum, frame):
    print("\n👋 Exiting on user request.")
    stop_video()
    exit(0)


def stop_video():
    global current_process
    if current_process:
        print("🚶 Person left. Fading out and stopping video.")
        time.sleep(FADE_OUT_DURATION)
        current_process.terminate()
        current_process.wait()
        current_process = None


def play_video(path):
    global current_process
    stop_video()
    print(f"🎥 Playing: {path}")
    current_process = subprocess.Popen([
        "mpv", path,
        "--fs",
        "--no-osd-bar",
        "--loop",
        "--quiet"
    ])


def detect_person_mediapipe(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    return results.pose_landmarks is not None


def main():
    global last_seen_time, person_present
    
    signal.signal(signal.SIGINT, handle_exit)
    main()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read from camera.")
                break

            person_detected = detect_person_mediapipe(frame)
            current_time = time.time()

            if person_detected:
                last_seen_time = current_time
                if not person_present:
                    person_present = True
                    random_video = os.path.join(VIDEO_FOLDER, random.choice(os.listdir(VIDEO_FOLDER)))
                    play_video(random_video)
            else:
                if person_present and (current_time - last_seen_time) > PERSON_LOST_TIMEOUT:
                    person_present = False
                    stop_video()

    except KeyboardInterrupt:
        handle_exit(None, None)
    finally:
        cap.release()
        pose.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    main()