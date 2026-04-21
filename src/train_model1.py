import os
import random
from collections import defaultdict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import roc_auc_score, roc_curve

# Import our new architectural flow
from multi_domain_fusion import MultiDomainFusion

# ----------------------------
# CONFIG (QUICK TRAIN MODE)
# ----------------------------
DATA_DIR = "data/final"
BATCH_SIZE = 8  # Reduced for memory efficiency
EPOCHS = 1      # Quick verification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUICK_TRAIN_SAMPLES = 1000 # 1000 per class (Total 2000)

print(f"🚀 Initializing Quick Train on {DEVICE}...")

# ----------------------------
# TRANSFORMS
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# LOAD DATASET (Subsampled)
# ----------------------------
samples = {"fake": [], "real": []}
for class_name in ["fake", "real"]:
    folder = os.path.join(DATA_DIR, class_name)
    if os.path.exists(folder):
        all_imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Randomly sample for speed
        samples[class_name] = random.sample(all_imgs, min(QUICK_TRAIN_SAMPLES, len(all_imgs)))

print(f"📊 Sampling {len(samples['fake'])} fake and {len(samples['real'])} real images.")

train_samples, val_samples = [], []
for class_name, img_paths in samples.items():
    label = 0 if class_name == "fake" else 1
    # Split the 1000 samples: 800 train, 200 val
    split = int(0.8 * len(img_paths))
    train_samples.extend([(p, label) for p in img_paths[:split]])
    val_samples.extend([(p, label) for p in img_paths[split:]])

random.shuffle(train_samples)

print(f"Train samples: {len(train_samples)}")
print(f"Val samples: {len(val_samples)}")

class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception: # Skip corrupt images
            return torch.zeros(3, 224, 224), label

train_loader = DataLoader(DeepfakeDataset(train_samples, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DeepfakeDataset(val_samples, val_transform), batch_size=BATCH_SIZE)

# ----------------------------
# MODEL & OPTIMIZATION
# ----------------------------
model = MultiDomainFusion().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# TRAIN LOOP
# ----------------------------
print("\n💪 Starting training loop...")
model.train()
total_loss = 0

for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    
    optimizer.zero_grad()
    cnn_logits, final_logits = model(images)
    
    loss_cnn = criterion(cnn_logits, labels)
    loss_fusion = criterion(final_logits, labels)
    loss = loss_fusion + 0.3 * loss_cnn 
    
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    if (i + 1) % 10 == 0:
        print(f"Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

# ----------------------------
# VALIDATION & SAVE
# ----------------------------
print("\n🧪 Running final validation...")
model.eval()
val_probs, val_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        _, final_logits = model(images)
        probs = torch.softmax(final_logits, dim=1)[:, 1]
        val_probs.extend(probs.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

if len(set(val_labels)) > 1:
    auc = roc_auc_score(val_labels, val_probs)
    print(f"Quick Train Finished! Validation AUC: {auc:.4f}")

# ALWAYS save for the app to pick it up
torch.save(model.state_dict(), "deepfake_multidomain.pth")
print("🏆 Model saved as deepfake_multidomain.pth (Ready for Streamlit App)")
