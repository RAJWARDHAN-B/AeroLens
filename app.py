import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# Page configuration
st.set_page_config(
    page_title="AeroLens | Wind Turbine Defect Detection",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Header Styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 1.2rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }

    /* Card Styling for Results */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
    }

    /* File Uploader Customization */
    .stFileUploader section {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 2px dashed rgba(0, 210, 255, 0.3) !important;
        border-radius: 10px !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d2ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/ffffff/wind-turbine.png", width=100)
    st.title("Settings")
    
    st.markdown("---")
    
    model_path = st.selectbox(
        "Select Model Weight",
        ["best.pt", "last.pt", "best.onnx"],
        index=0
    )
    
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.1, 1.0, 0.45, 0.05)
    
    st.markdown("---")
    st.markdown("### Model Info")
    st.info("Upload the `best.pt` file to the project root directory to enable detection.")

# Main Page Design
st.markdown('<h1 class="main-header">AeroLens AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Wind Turbine Defect Detection System</p>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        return None
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_yolo_model(model_path)

if model is None:
    st.warning(f"⚠️ **{model_path}** not found. Please ensure the model file is in the project directory.")
    st.markdown("""
        ### Setup Instructions:
        1. Ensure you have the `best.pt` or `best.onnx` file.
        2. Place it in the directory: `d:/PROGRAMMING/Internships_assignments/WindRover/AeroLens/`
        3. Refresh this page.
    """)
else:
    # File Uploader
    uploaded_file = st.file_uploader("Choose an image of a wind turbine...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("### Detection Result")
            
            # Predict
            with st.spinner('Analyzing for defects...'):
                results = model.predict(
                    source=image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    save=False
                )
                
                # Check for defects
                result = results[0]
                annotated_img = result.plot() # returns numpy array (BGR)
                
                # Convert BGR to RGB
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                # Show detection image
                st.image(annotated_img_rgb, use_container_width=True)
                
                # Download Button
                result_img = Image.fromarray(annotated_img_rgb)
                import io
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=byte_im,
                    file_name="detected_defects.png",
                    mime="image/png",
                )
        
        # Detection Details
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Detection Summary")
        
        num_defects = len(result.boxes)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Defects Detected", num_defects)
        
        if num_defects > 0:
            st.success(f"Detections complete: {num_defects} defects identified.")
            
            # Show classes found
            classes = result.names
            detected_classes = [classes[int(box.cls)] for box in result.boxes]
            class_counts = {name: detected_classes.count(name) for name in set(detected_classes)}
            
            st.write("#### Classification Breakdown:")
            for cls_name, count in class_counts.items():
                st.write(f"- **{cls_name}**: {count}")
            
            # Detailed Box Info (Optional)
            with st.expander("Show Raw Detection Data"):
                st.write(result.boxes.data)
        else:
            st.info("No defects detected in this image at the current confidence threshold.")
            
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #718096; font-size: 0.8rem;">
        Developed for Wind Turbine Maintenance & Safety | AeroLens AI 2024
    </div>
    """, 
    unsafe_allow_html=True
)
