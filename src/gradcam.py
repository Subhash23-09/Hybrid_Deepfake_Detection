import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import random
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "deepfake_model_augmented.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ----------------------------
# PICK RANDOM IMAGE
# ----------------------------
DATA_DIR = "data/final"
class_name = random.choice(["real", "fake"])
folder = os.path.join(DATA_DIR, class_name)

img_name = random.choice(os.listdir(folder))
IMAGE_PATH = os.path.join(folder, img_name)

print("Testing:", IMAGE_PATH)

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
# HOOK STORAGE
# ----------------------------
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# hook LAST CONV layer
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ----------------------------
# LOAD IMAGE
# ----------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ----------------------------
# CLEAR HOOK STORAGE
# ----------------------------
features.clear()
gradients.clear()

# ----------------------------
# FORWARD
# ----------------------------
output = model(input_tensor)

probs = torch.softmax(output, dim=1)
pred_class = output.argmax(dim=1).item()

label = "FAKE" if pred_class == 0 else "REAL"

print(f"\nPrediction: {label}")
print(f"Confidence: {probs[0][pred_class].item():.4f}")

# ----------------------------
# BACKWARD
# ----------------------------
model.zero_grad()
output[0, pred_class].backward()

# ----------------------------
# GENERATE CAM
# ----------------------------
grads = gradients[0]
fmap = features[0]

weights = torch.mean(grads, dim=[2, 3], keepdim=True)
cam = torch.sum(weights * fmap, dim=1).squeeze()

cam = torch.relu(cam)
cam = cam.detach().cpu().numpy()

# normalize safely
cam = cam - np.min(cam)
if np.max(cam) != 0:
    cam = cam / np.max(cam)

# resize
cam = cv2.resize(cam, (224, 224))

# ----------------------------
# OVERLAY
# ----------------------------
img_np = np.array(img.resize((224, 224)))

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = heatmap * 0.4 + img_np

# ----------------------------
# SAVE OUTPUT
# ----------------------------
output_path = "gradcam_result.jpg"
cv2.imwrite(output_path, overlay.astype(np.uint8))

print(f"Grad-CAM saved at: {output_path}")