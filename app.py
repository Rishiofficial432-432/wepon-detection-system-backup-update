import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time

# Set page config
st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-camera.png", width=100)
    st.title("Settings")
    
    confidence = st.slider("Model Confidence", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    source_type = st.radio("Source", ("Image Upload", "Video Upload"))

# Main Content
st.title("🔍 Automated Weapon Detection System")
st.markdown("### Detect weapons in images and videos using Deep Learning")

# Load Model
@st.cache_resource
def load_model():
    # Load the best model after training, for now use yolov8n.pt or 
    # the path to 'runs/detect/weapon_detection_model/weights/best.pt' if it exists
    # Fallback to yolov8n.pt if not trained yet
    import os
    # Check locations in order of preference
    possible_paths = [
        "best.pt",  # Portable location (root)
        "best_model.pt", # Portable location (root) - alternative name
        "runs/detect/weapon_detection_model/weights/best.pt",  # Training output
        "yolov8n.pt"  # Fallback
    ]
    
    model_path = "yolov8n.pt"
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    if model_path == "yolov8n.pt":
         st.warning("Using pre-trained YOLOv8n (not fine-tuned). If you have a trained model, name it 'best.pt' and place it in this folder.")
    
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

if source_type == "Image Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
        if st.button("Detect Weapons"):
            with st.spinner('Analysing...'):
                res = model.predict(image, conf=confidence)
                res_plotted = res[0].plot()
                
                with col2:
                    st.image(res_plotted, caption='Detection Result', use_container_width=True)
                    
                # Show detections
                with st.expander("Detection Details"):
                    for box in res[0].boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = res[0].names[cls]
                        st.write(f"Detected **{name}** with confidence **{conf:.2f}**")

elif source_type == "Video Upload":
    uploaded_file = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
                
            # Process frame
            results = model.predict(frame, conf=confidence)
            res_plotted = results[0].plot()
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            stframe.image(frame_rgb, caption='Processing Video...', use_container_width=True)
            
        vf.release()

st.markdown("---")
st.markdown("Developed with ❤️ using YOLOv8 and Streamlit")
