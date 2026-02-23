import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# Fix for Streamlit Cloud non-writable directory
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

# Logo path
LOGO_PATH = "aerolenslogobg.png"

# Page configuration
st.set_page_config(
    page_title="AeroLens.ai | Wind Turbine Defect Detection",
    page_icon=Image.open(LOGO_PATH) if os.path.exists(LOGO_PATH) else "🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 52, 96, 0.4) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Sidebar Widgets */
    .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.03);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    /* Header Styling */
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 4rem;
        background: linear-gradient(120deg, #00d2ff 0%, #9face6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        letter-spacing: -1px;
    }

    .sub-title {
        color: #00d2ff;
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }

    .sub-header {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2.5rem;
        max-width: 600px;
    }

    /* Card Styling for Results */
    .prediction-card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    /* File Uploader Customization */
    .stFileUploader section {
        background-color: rgba(255, 255, 255, 0.01) !important;
        border: 2px dashed rgba(0, 210, 255, 0.5) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2.5rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        width: 100%;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.5);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d2ff;
        font-size: 2.5rem !important;
    }

    /* Center Logo */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }

    /* Hide Streamlit Menu and Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.image("https://img.icons8.com/isometric-line/100/ffffff/wind-turbine.png", width=100)
    
    st.markdown("<h2 style='text-align: center; color: white;'>Navigation</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### ⚙️ Engine Settings")
    model_path = st.selectbox(
        "Model Weights",
        ["best.pt", "last.pt", "best.onnx"],
        index=0
    )
    
    conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IOU Filter", 0.1, 1.0, 0.45, 0.05)
    
    st.markdown("---")
    st.markdown("### ℹ️ App Information")
    st.info("AeroLens.ai uses YOLOv8 for sub-millimeter precision in defect identification.")

# Main Page Design
st.markdown('<h1 class="main-header">AeroLens.ai</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Intelligent Wind Turbine Inspection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Harnessing advanced computer vision to ensure the structural integrity and efficiency of renewable energy infrastructure.</p>', unsafe_allow_html=True)

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
