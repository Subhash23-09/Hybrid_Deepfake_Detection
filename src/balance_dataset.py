import os
import random
import shutil

REAL_DIR = "data/processed/real_crops"
FAKE_DIR = "data/processed/fake_crops"

OUTPUT_REAL = "data/final/real"
OUTPUT_FAKE = "data/final/fake"

TARGET_SIZE = 35000  # adjust if needed

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)

# ---- REAL ----
real_images = os.listdir(REAL_DIR)
real_selected = random.sample(real_images, min(TARGET_SIZE, len(real_images)))

for img in real_selected:
    shutil.copy(os.path.join(REAL_DIR, img), os.path.join(OUTPUT_REAL, img))

# ---- FAKE ----
fake_images = os.listdir(FAKE_DIR)
fake_selected = random.sample(fake_images, min(TARGET_SIZE, len(fake_images)))

for img in fake_selected:
    shutil.copy(os.path.join(FAKE_DIR, img), os.path.join(OUTPUT_FAKE, img))

print("Balanced dataset created!")
print(f"Real: {len(real_selected)}")
print(f"Fake: {len(fake_selected)}")