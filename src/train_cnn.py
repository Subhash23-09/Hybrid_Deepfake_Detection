import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import os

# RTX 3050 setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = list(Path(real_dir).glob('*.png'))
        self.fake_images = list(Path(fake_dir).glob('*.png'))
        self.transform = transform
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.images = self.real_images + self.fake_images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = torch.FloatTensor(image.transpose(2,0,1))  # HWC → CHW
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

# EfficientNet-B0 (modern syntax)
model = models.efficientnet_b0(weights='DEFAULT')
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

# Data transforms
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = DeepfakeDataset('data/ffhq', 'data/ff++', transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🚀 Training EfficientNet Deepfake Detector...")
# print(f"📊 Dataset: {len(dataset)} images ({len(Path('data/ffhq').glob('*.png'))} real + {len(Path('data/ff++').glob('*.png'))} fake)")

for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    print(f"Epoch {epoch+1}/5 - Loss: {running_loss/len(dataloader):.4f}, Acc: {100.*correct/total:.1f}%")

torch.save(model.state_dict(), 'deepfake_detector.pth')
print("✅ Model saved: deepfake_detector.pth")
print("🎉 Deepfake detector trained on RTX 3050!")
