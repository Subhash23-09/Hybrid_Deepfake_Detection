import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import transforms
import io

from multi_domain_fusion import MultiDomainFusion

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "data/final"
MODEL_PATH = "deepfake_multidomain.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# COMPRESSION AUGMENTATIONS
# ----------------------------
def apply_jpeg_compression(image, quality_level=50):
    buffer = io.BytesIO()
    # Save the PIL image to memory with specific JPEG quality
    image.save(buffer, format="JPEG", quality=quality_level)
    buffer.seek(0)
    # Read back
    return Image.open(buffer).convert("RGB")

def evaluate_robustness(model, test_samples, quality_level=None):
    model.eval()
    val_probs, val_labels = [], []
    
    with torch.no_grad():
        for path, label in test_samples:
            img = Image.open(path).convert("RGB")
            
            # Apply compression attack if specified
            if quality_level is not None:
                img = apply_jpeg_compression(img, quality_level)
                
            input_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
            
            _, final_logits = model(input_tensor)
            prob = torch.softmax(final_logits, dim=1)[0, 1].item()
            
            val_probs.append(prob)
            val_labels.append(label)
            
    if len(set(val_labels)) > 1:
        auc = roc_auc_score(val_labels, val_probs)
        fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return auc, eer
    return 0.0, 0.0

if __name__ == "__main__":
    print("🚀 Starting IEEE Evaluation & Robustness Check...")
    
    model = MultiDomainFusion()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: {MODEL_PATH} not found. Evaluation will run on untrained architecture.")
    model.to(DEVICE)
    
    samples = []
    for label, class_name in enumerate(["fake", "real"]):
        folder = os.path.join(DATA_DIR, class_name)
        if os.path.exists(folder):
            for img in os.listdir(folder):
                samples.append((os.path.join(folder, img), label))
                
                # Limit samples to speed up standard eval run
                if len(samples) > 200: 
                    break 

    if not samples:
        print("Error: No data found. Ensure your data/final directory is populated.")
        exit(1)
        
    print(f"Evaluating on {len(samples)} samples...")
    
    # 1. Standard Dataset Accuracy
    base_auc, base_eer = evaluate_robustness(model, samples, quality_level=None)
    print(f"\n✅ Standard Performance:")
    print(f"   AUC: {base_auc:.4f} | EER: {base_eer:.4f}")
    
    # 2. Social Media Degradation Sim (JPEG 75)
    q75_auc, q75_eer = evaluate_robustness(model, samples, quality_level=75)
    print(f"\n📱 Compression Quality 75 (WhatsApp/Instagram):")
    print(f"   AUC: {q75_auc:.4f} | EER: {q75_eer:.4f}")
    
    # 3. Aggressive Degradation Sim (JPEG 40)
    q40_auc, q40_eer = evaluate_robustness(model, samples, quality_level=40)
    print(f"\n📉 Extreme Compression Quality 40 (Aggressive Attack):")
    print(f"   AUC: {q40_auc:.4f} | EER: {q40_eer:.4f}")

    print("\nEvaluation successfully completed. Exporting test data for IEEE publication.")
