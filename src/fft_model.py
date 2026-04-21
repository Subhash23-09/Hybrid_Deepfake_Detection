import cv2
import numpy as np
import os
import random
from PIL import Image

# ----------------------------
# FFT FEATURE EXTRACTION
# ----------------------------
def extract_fft_features(img):
    # if input is path → read
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    else:
        # PIL → numpy → grayscale
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    if img is None:
        return None

    img = cv2.resize(img, (224, 224))

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)

    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    return magnitude


# ----------------------------
# FFT SCORE (HIGH-FREQ)
# ----------------------------
def fft_score(img):
    fft_img = extract_fft_features(img)

    if fft_img is None:
        return 0

    h, w = fft_img.shape
    center_h, center_w = h // 2, w // 2
    radius = 20

    mask = np.ones_like(fft_img)
    mask[
        center_h - radius:center_h + radius,
        center_w - radius:center_w + radius
    ] = 0

    high_freq = fft_img * mask
    score = np.mean(high_freq)

    return score


# ----------------------------
# TEST (optional)
# ----------------------------
if __name__ == "__main__":
    DATA_DIR = "data/final"

    for _ in range(5):
        class_name = random.choice(["real", "fake"])
        folder = os.path.join(DATA_DIR, class_name)

        img_name = random.choice(os.listdir(folder))
        img_path = os.path.join(folder, img_name)

        score = fft_score(img_path)

        print(f"\nClass: {class_name}")
        print(f"Image: {img_name}")
        print(f"FFT Score: {score:.4f}")