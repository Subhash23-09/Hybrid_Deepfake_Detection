import streamlit as st
import torch
import numpy as np
import cv2
import base64
import tempfile
import os
from PIL import Image
from torchvision import transforms
import ollama
from facenet_pytorch import MTCNN

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "deepfake_multidomain.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_mtcnn_detector():
    # Native PyTorch MTCNN is faster and runs on GPU
    return MTCNN(keep_all=False, device=DEVICE, post_process=False)

face_detector = get_mtcnn_detector()

@st.cache_resource
def load_model():
    # Relative import or ensure path is clear
    from multi_domain_fusion import MultiDomainFusion
    model = MultiDomainFusion()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load weights from {MODEL_PATH}. Exception: {e}")
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# LOGIC
# ----------------------------
def extract_face(image: Image.Image):
    # facenet-pytorch MTCNN expects PIL images for best results
    boxes, probs, landmarks = face_detector.detect(image, landmarks=True)

    if boxes is not None:
        # Take the most confident face
        box = boxes[0]
        landmark = landmarks[0]
        
        # x1, y1, x2, y2
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Boundary checks
        w_img, h_img = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # 1. Clean Crop for inference
        face_img_clean = image.crop((x1, y1, x2, y2))
        
        # 2. Annotated Crop for UI
        # We draw onto a BGR copy for OpenCV visualization
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_bgr_annotated = img_cv[y1:y2, x1:x2].copy()
        
        # Draw landmarks (relative to crop)
        for point in landmark:
            px, py = int(point[0] - x1), int(point[1] - y1)
            cv2.circle(face_bgr_annotated, (px, py), 2, (0, 255, 0), -1)
        
        # Draw border (box around crop)
        h_c, w_c = face_bgr_annotated.shape[:2]
        cv2.rectangle(face_bgr_annotated, (0, 0), (w_c-1, h_c-1), (0, 255, 0), 2)

        face_rgb_annotated = cv2.cvtColor(face_bgr_annotated, cv2.COLOR_BGR2RGB)

        return face_img_clean, Image.fromarray(face_rgb_annotated)
    
    return image, image # Fallback

def get_prediction_and_gradcam(image: Image.Image):
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # We need to hook into the EfficientNet backbone to visualize spatial anomalies
    features, gradients = [], []
    def forward_hook(module, input, output): features.append(output)
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

    target_layer = model.spatial.feature_extractor[-1]
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    model.zero_grad()
    _, final_logits = model(input_tensor)
    
    probs = torch.softmax(final_logits, dim=1)
    
    # Label: 0 = FAKE, 1 = REAL
    pred_class = final_logits.argmax(dim=1).item()
    label = "FAKE" if pred_class == 0 else "REAL"
    confidence = probs[0][pred_class].item()

    # Backward for Grad-CAM
    final_logits[0, pred_class].backward()

    # Generate CAM
    grads = gradients[0]
    fmap = features[0]
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * fmap, dim=1).squeeze()
    cam = torch.relu(cam).detach().cpu().numpy()

    # Normalize and ensure it's a valid float32 numpy array for OpenCV
    cam = np.array(cam, dtype=np.float32)
    
    # Avoid division by zero and handle scalar cases
    if cam.ndim < 2:
        cam = np.expand_dims(cam, axis=0) if cam.ndim == 1 else np.zeros((7, 7), dtype=np.float32)
        
    cam_min = np.min(cam)
    cam_max = np.max(cam)
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # Resize Heatmap
    cam = cv2.resize(cam, (224, 224))

    # Overlay
    img_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img_np
    
    handle_fw.remove()
    handle_bw.remove()

    return label, confidence, overlay.astype(np.uint8)

def encode_image(image_cv_array):
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_cv_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def generate_vlm_explanation(gradcam_overlay_img, classification):
    """
    Uses Moondream vision model strictly grounding explanation in the Grad-CAM representation.
    """
    base64_img = encode_image(gradcam_overlay_img)
    
    # Prompting the multi-modal VLM (Moondream is much smaller and fits in 2GB RAM)
    if classification == "FAKE":
        prompt = "The red areas in this heatmap indicate where the AI found deepfake anomalies. Analyze those specific regions and concisely explain what structural or blending artifacts indicate this image is FAKE."
    else:
        prompt = "The red areas in this heatmap indicate where the AI found the most consistent facial features. Analyze those specific regions and concisely explain why the skin texture and lighting here confirm the image is REAL and authentic."
    
    try:
        response = ollama.chat(
            model='moondream', 
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [base64_img]
            }]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to VLM. Please ensure Ollama is running and 'moondream' model is available (`ollama pull moondream`). Error: {str(e)}"

def process_video(video_path):
    """
    Samples frames from video and runs inference on each.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample 1 frame per 0.5 seconds of video
    sample_interval = max(1, int(fps / 2))
    
    results = []
    processed_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    best_frame_data = None
    max_confidence = -1.0
    all_analyzed_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_interval == 0:
            # Convert BGR to RGB PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Detect Face
            face_clean, face_annotated = extract_face(pil_img)
            
            label, conf, gradcam_img = get_prediction_and_gradcam(face_clean)
            
            # Store result (0=FAKE, 1=REAL)
            prob_fake = (1.0 - conf) if label == "REAL" else conf
            results.append(prob_fake)
            
            # Record frame data for gallery
            frame_data = {
                'annotated': face_annotated,
                'label': label,
                'conf': conf,
                'timestamp': frame_idx / fps,
                'gradcam': gradcam_img
            }
            all_analyzed_frames.append(frame_data)
            
            # Track the most "confident/representative" frame for deep-dive
            if conf > max_confidence:
                max_confidence = conf
                best_frame_data = frame_data
            
            processed_count += 1
            progress_bar.progress(min(1.0, frame_idx / total_frames))
            status_text.text(f"Processed {processed_count} frames...")
            
        frame_idx += 1
        
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        return None
        
    avg_fake_prob = np.mean(results)
    final_label = "FAKE" if avg_fake_prob > 0.5 else "REAL"
    final_conf = avg_fake_prob if final_label == "FAKE" else (1.0 - avg_fake_prob)
    
    return final_label, final_conf, best_frame_data, all_analyzed_frames

# ----------------------------
# UI
# ----------------------------
st.title("🛡️ IEEE Deepfake Detection System")
st.write("Multi-Domain Fusion (Frequency+Spatial) with Visual Grounded Explainability")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4", "avi", "mov"])

if uploaded_file:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'image':
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="Uploaded Original Image", use_container_width=True)

        with st.spinner("Extracting Face (MTCNN)..."):
            face_img_clean, face_img_annotated = extract_face(raw_image)
        
        st.image(face_img_annotated, caption="MTCNN Detected Face (Box & Landmarks)", width=224)

        with st.spinner("Running Multi-Domain Inference..."):
            label, conf, gradcam_img = get_prediction_and_gradcam(face_img_clean)

        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classification", label)
        with col2:
            st.metric("Confidence", f"{conf:.2%}")

        st.subheader("Grad-CAM Spatial Heatmap")
        st.image(gradcam_img, caption="Neural Attention Map", use_container_width=True)

        # AUTOMATIC VLM EXPLANATION
        with st.spinner("Connecting to Multimodal Moondream for Human Explanation..."):
            explanation = generate_vlm_explanation(gradcam_img, label)
        st.subheader("🛡️ AI Grounded Explanation (Moondream)")
        st.info(explanation)

    elif file_type == 'video':
        st.video(uploaded_file)
        
        if st.button("Start Video Analysis"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                result = process_video(tmp_path)
                if result:
                    label, final_conf, best_frame, frame_gallery = result
                    
                    st.divider()
                    st.subheader("Video Analysis Summary")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Aggregated Decision", label)
                    with col2:
                        st.metric("Mean Confidence", f"{final_conf:.2%}")
                    
                    # New Gallery Section
                    with st.expander("🔍 Detailed Frame-by-Frame Results", expanded=True):
                        st.write("MTCNN Face Detection & Confidence for each analyzed frame:")
                        cols = st.columns(4) # Show in grid of 4
                        for idx, fdata in enumerate(frame_gallery):
                            with cols[idx % 4]:
                                st.image(fdata['annotated'], caption=f"T: {fdata['timestamp']:.1f}s")
                                color = "red" if fdata['label'] == "FAKE" else "green"
                                st.markdown(f":{color}[{fdata['label']}: {fdata['conf']:.1%}]")

                    if best_frame:
                        st.divider()
                        st.subheader(f"Deep Dive Analysis: Top Representative Frame ({best_frame['label']})")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(best_frame['annotated'], caption=f"MTCNN Detection (Conf: {best_frame['conf']:.2%})", use_container_width=True)
                        with col2:
                            st.image(best_frame['gradcam'], caption="Spatial Anomaly Map (Grad-CAM)", use_container_width=True)
                        
                        # AUTOMATIC VLM EXPLANATION for the most suspicious frame
                        with st.spinner("Analyzing neural attention map with Moondream VLM..."):
                            explanation = generate_vlm_explanation(best_frame['gradcam'], best_frame['label'])
                        st.subheader("🛡️ AI Grounded Explanation (Moondream)")
                        st.info(explanation)
                else:
                    st.error("Could not process video or no faces detected.")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)