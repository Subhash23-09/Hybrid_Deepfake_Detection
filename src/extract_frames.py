import os
import cv2
from tqdm import tqdm

# INPUT PATHS
REAL_DIRS = [
    "data/archive/Celeb-real",
    "data/archive/YouTube-real"
]

FAKE_DIR = "data/archive/Celeb-synthesis"

# OUTPUT
OUTPUT_REAL = "data/processed/real_frames"
OUTPUT_FAKE = "data/processed/fake_frames"

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)


def extract_frames(video_path, output_dir, prefix):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            filename = f"{prefix}_{video_name}_{saved_count}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()


# -------- REAL --------
for real_dir in REAL_DIRS:
    for video in tqdm(os.listdir(real_dir), desc=f"Processing {real_dir}"):
        video_path = os.path.join(real_dir, video)
        extract_frames(video_path, OUTPUT_REAL, "real")

# -------- FAKE --------
for video in tqdm(os.listdir(FAKE_DIR), desc="Processing fake"):
    video_path = os.path.join(FAKE_DIR, video)
    extract_frames(video_path, OUTPUT_FAKE, "fake")

print("Frame extraction done!")