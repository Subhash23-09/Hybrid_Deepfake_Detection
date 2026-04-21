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
# CONFIG
# ----------------------------
DATA_DIR = "data/final"
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ----------------------------
# TRANSFORMS
# ----------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# LOAD DATASET (No Leakage split)
# ----------------------------
samples = []
for label, class_name in enumerate(["fake", "real"]): # 0 = fake, 1 = real
    folder = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(folder):
        continue
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        samples.append((path, label))

groups = defaultdict(list)
for path, label in samples:
    filename = os.path.basename(path)
    parts = filename.split("_")
    video_id = "_".join(parts[1:-1]) if len(parts) >= 3 else filename
    groups[video_id].append((path, label))

group_keys = list(groups.keys())
random.shuffle(group_keys)
split_idx = int(0.8 * len(group_keys))

train_keys = group_keys[:split_idx]
val_keys = group_keys[split_idx:]

train_samples, val_samples = [], []
for key in train_keys: train_samples.extend(groups[key])
for key in val_keys: val_samples.extend(groups[key])

print(f"Train samples: {len(train_samples)}")
print(f"Val samples: {len(val_samples)}")


class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

train_loader = DataLoader(DeepfakeDataset(train_samples, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(DeepfakeDataset(val_samples, val_transform), batch_size=BATCH_SIZE, num_workers=0)


# ----------------------------
# MODEL & OPTIMIZATION
# ----------------------------
model = MultiDomainFusion().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# TRAIN LOOP w/ JOINT SUPERVISION
# ----------------------------
best_auc = 0.0

if len(train_samples) > 0:
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass: Spatial and Multi-Domain Fusion
            cnn_logits, final_logits = model(images)
            
            # Intermediate Supervision: we teach the spatial branch to also be accurate
            loss_cnn = criterion(cnn_logits, labels)
            loss_fusion = criterion(final_logits, labels)
            
            # Combine losses
            loss = loss_fusion + 0.3 * loss_cnn 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"\nEpoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
        # ----------------------------
        # VALIDATION (IEEE Standards: AUC & EER)
        # ----------------------------
        model.eval()
        val_probs, val_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                _, final_logits = model(images)
                
                # Get probability for class 1 (Real)
                probs = torch.softmax(final_logits, dim=1)[:, 1]
                
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        if len(set(val_labels)) > 1:
            auc = roc_auc_score(val_labels, val_probs)
            fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
            
            # Calculate EER (Equal Error Rate)
            fnr = 1 - tpr
            eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
            eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            
            print(f"Validation AUC: {auc:.4f} | EER: {eer:.4f}")
            
            # Save best model
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "deepfake_multidomain.pth")
                print("🏆 New Best Model Saved!")
                
                # Plot ROC
                plt.figure()
                plt.plot(fpr, tpr, label=f'Model ROC curve (area = {auc:0.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.savefig('roc_curve.png')
        else:
            print("Not enough classes in validation set to calculate AUC.")
            torch.save(model.state_dict(), "deepfake_multidomain.pth")

else:
    print("WARNING: No data found in data/final. Skipping training.")