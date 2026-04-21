import os
import cv2
from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(keep_all=False, device=device, post_process=False)

INPUT_REAL = "data/processed/real_frames"
INPUT_FAKE = "data/processed/fake_frames"

OUTPUT_REAL = "data/processed/real_crops"
OUTPUT_FAKE = "data/processed/fake_crops"

os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)


def extract_face(img_path, output_dir, prefix):
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        return

    boxes, _ = detector.detect(img)

    if boxes is not None:
        # Take largest box
        box = boxes[0]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # Crop and resize
        face = img.crop((x1, y1, x2, y2))
        face = face.resize((224, 224), Image.LANCZOS)
        
        # Save
        filename = f"{prefix}_{os.path.basename(img_path)}"
        face.save(os.path.join(output_dir, filename))

# -------- REAL --------
for img_name in tqdm(os.listdir(INPUT_REAL), desc="Real faces"):
    img_path = os.path.join(INPUT_REAL, img_name)
    extract_face(img_path, OUTPUT_REAL, "real")

# -------- FAKE --------
for img_name in tqdm(os.listdir(INPUT_FAKE), desc="Fake faces"):
    img_path = os.path.join(INPUT_FAKE, img_name)
    extract_face(img_path, OUTPUT_FAKE, "fake")

print("Face extraction done!")