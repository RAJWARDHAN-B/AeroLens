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
        background-color: #0f172a !important; /* Deeper, more solid background */
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Sidebar Text & Labels */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #f8fafc !important; /* Ultra-bright white for labels */
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #00d2ff !important;
    }

    /* Sidebar Widgets Styling */
    .stSelectbox > div > div, .stSlider > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }

    /* Sidebar Info Box */
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(0, 210, 255, 0.1);
        color: #e2e8f0;
        border: 1px solid rgba(0, 210, 255, 0.2);
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

    /* GLOBAL TEXT COLOR - FORCE ALL TO READABLE WHITE/SLATE */
    html, body, [data-testid="stAppViewContainer"], [data-testid="sidebar-content"] {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Force all headings and standard text */
    h1, h2, h3, h4, h5, h6, p, li, span, small {
        color: #ffffff !important;
    }

    /* Label & Widget Text */
    label, [data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important; /* Slate 100 */
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 8px !important;
    }

    /* Enhanced File Uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(0, 210, 255, 0.3);
        border-radius: 15px;
        padding: 2.5rem;
        transition: all 0.3s ease;
        display: flex;
        justify-content: center;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #00d2ff;
        background-color: rgba(0, 210, 255, 0.05);
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.1);
    }

    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }
    
    /* Browse Files Button */
    [data-testid="stFileUploader"] button {
        background-color: #00d2ff !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        transition: transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    [data-testid="stFileUploader"] button:hover {
        transform: scale(1.05);
        background-color: #ffffff !important;
    }

    /* File Uploader Text Legibility */
    [data-testid="stFileUploader"] section div {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] section small {
        color: #94a3b8 !important;
        font-weight: 400 !important;
    }

    /* Selectbox & Input Text */
    .stSelectbox div[data-baseweb="select"] > div {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Slider value text */
    [data-testid="stThumbValue"] {
        color: #00d2ff !important;
        font-weight: 700 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #00d2ff !important;
    }

    /* Override Sidebar specific dimming */
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #ffffff !important;
        font-size: 1rem !important;
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
        color: #00d2ff !important; 
        font-weight: 700;
        font-size: 1.6rem;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-weight: 400;
        font-size: 1.15rem;
        color: #cbd5e1 !important; 
        margin-bottom: 2.5rem;
        max-width: 800px;
        line-height: 1.6;
    }

    /* Success/Info boxes text */
    .stAlert div {
        color: #ffffff !important;
    }

    /* Download Button Styling */
    .stDownloadButton button {
        width: 100% !important;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.4) !important;
        background: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%) !important;
    }

    /* Cards and Containers */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        backdrop-filter: blur(10px);
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
        Developed for Wind Turbine Maintenance & Safety | AeroLens AI 2026
    </div>
    """, 
    unsafe_allow_html=True
)
