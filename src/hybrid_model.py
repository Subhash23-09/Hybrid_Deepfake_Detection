import os
import random
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

# import FFT from your module
from fft_model import fft_score

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "deepfake_model_augmented.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ----------------------------
# TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# CNN SCORE
# ----------------------------
def cnn_score(image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)

    # IMPORTANT:
    # If label mapping is:
    # 0 = fake, 1 = real → use probs[0][0]
    # 0 = real, 1 = fake → use probs[0][1]

    return probs[0][0].item()  # adjust if needed


# ----------------------------
# HYBRID PREDICTION
# ----------------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    # CNN score
    cnn_s = cnn_score(image)

    # FFT score (from external file)
    fft_s = fft_score(image)

    # FUSION (tunable weights)
    final_score = 0.8 * cnn_s + 0.2 * fft_s

    # decision
    label = "FAKE" if final_score > 0.5 else "REAL"

    print("\n--- PREDICTION ---")
    print(f"Image: {image_path}")
    print(f"CNN Score: {cnn_s:.4f}")
    print(f"FFT Score: {fft_s:.4f}")
    print(f"Final Score: {final_score:.4f}")
    print(f"Prediction: {label}")


# ----------------------------
# TEST
# ----------------------------
if __name__ == "__main__":
    DATA_DIR = "data/final"

    class_name = random.choice(["real", "fake"])
    folder = os.path.join(DATA_DIR, class_name)

    img_name = random.choice(os.listdir(folder))
    test_image = os.path.join(folder, img_name)
#    test_image = "data/final/fake/fake_fake_id0_id1_0002_32.jpg"
#    test_image = r"data\final\real\real_real_00000_16.jpg"
    predict(test_image)