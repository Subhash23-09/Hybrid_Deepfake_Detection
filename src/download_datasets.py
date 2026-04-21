import cv2
import numpy as np
from pathlib import Path
import os

def create_realistic_faces():
    os.makedirs('data/ffhq', exist_ok=True)
    
    # REAL faces: Oval skin + eyes + mouth
    for i in range(50):
        h, w = 256, 256
        img = np.ones((h,w,3), dtype=np.uint8) * 180  # Skin tone
        
        # Face oval
        cv2.ellipse(img, (128,120), (60,80), 0, 0, 360, (220,180,160), -1)
        
        # Eyes
        cv2.circle(img, (100,100), 8, (50,50,50), -1)
        cv2.circle(img, (156,100), 8, (50,50,50), -1)
        
        # Mouth
        cv2.ellipse(img, (128,160), (25,10), 0, 0, 180, (0,0,0), 2)
        
        cv2.imwrite(f'data/ffhq/real_{i:03d}.png', img)

    os.makedirs('data/ff++', exist_ok=True)
    
    # FAKE faces: Same + deepfake artifacts
    for i in range(50):
        img = img.copy()
        # Deepfake seams (red borders)
        cv2.rectangle(img, (60,60), (190,190), (0,0,255), 2)
        cv2.line(img, (80,80), (170,170), (0,0,255), 2)
        cv2.imwrite(f'data/ff++/fake_{i:03d}.png', img)
    
    print("✅ 50 realistic real + 50 fake faces ready!")

if __name__ == "__main__":
    create_realistic_faces()
