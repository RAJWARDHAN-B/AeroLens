import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import io
import zipfile

# Fix for Streamlit Cloud
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

LOGO_PATH = "aerolenslogobg.png"

# Page Config
st.set_page_config(
    page_title="AeroLens.ai | Wind Turbine Defect Detection",
    page_icon="🌬️",
    layout="wide",
)

# -----------------------------
# Premium UI Styling
# -----------------------------

st.markdown("""
<style>

.stApp{
background: radial-gradient(circle at top,#0f172a,#020617);
color:white;
font-family: 'Inter', sans-serif;
}

/* header */

.main-title{
font-size:4rem;
font-weight:800;
background:linear-gradient(120deg,#00d2ff,#3a7bd5);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.subtitle{
color:#94a3b8;
font-size:1.2rem;
margin-bottom:30px;
}

/* sidebar */

[data-testid="stSidebar"]{
background:#020617;
border-right:1px solid rgba(255,255,255,0.1);
}

/* cards */

.result-card{
background:rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.1);
padding:25px;
border-radius:15px;
margin-top:20px;
}

/* uploader */

[data-testid="stFileUploader"]{
border:2px dashed rgba(0,210,255,0.3);
border-radius:15px;
padding:2rem;
background:rgba(255,255,255,0.03);
}

/* buttons */

.stDownloadButton button{
background:linear-gradient(90deg,#00d2ff,#3a7bd5);
color:white;
border:none;
border-radius:8px;
font-weight:700;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH)

    st.markdown("## ⚙️ Engine Settings")

    model_path = st.selectbox(
        "Model Weights",
        ["best.pt", "last.pt", "best.onnx"],
        index=0
    )

    conf_threshold = st.slider("Confidence Threshold",0.1,1.0,0.25,0.05)

    iou_threshold = st.slider("IOU Threshold",0.1,1.0,0.45,0.05)

    st.markdown("---")

    st.info(
        "AeroLens.ai uses **YOLOv8 deep learning models** to detect "
        "structural defects in wind turbines."
    )

# -----------------------------
# Title
# -----------------------------

st.markdown('<div class="main-title">AeroLens.ai</div>', unsafe_allow_html=True)

st.markdown(
'<div class="subtitle">AI-powered wind turbine defect detection platform</div>',
unsafe_allow_html=True
)

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model(path):

    if not os.path.exists(path):
        return None

    try:
        return YOLO(path)
    except:
        return None


model = load_model(model_path)

if model is None:

    st.error("⚠️ Model file not found. Place your weights in the project folder.")

    st.stop()

# -----------------------------
# Multi Image Upload
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload wind turbine images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

# -----------------------------
# Prediction Section
# -----------------------------

if uploaded_files:

    st.success(f"{len(uploaded_files)} image(s) uploaded")

    progress_bar = st.progress(0)

    zip_buffer = io.BytesIO()

    zip_file = zipfile.ZipFile(zip_buffer,"w")

    total_defects = 0

    for i,uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file)

        col1,col2 = st.columns(2)

        with col1:

            st.markdown("### Original Image")

            st.image(image,use_container_width=True)

        with col2:

            st.markdown("### Detection Result")

            with st.spinner("Running AI inspection..."):

                results = model.predict(
                    source=image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    save=False
                )

                result = results[0]

                annotated = result.plot()

                annotated_rgb = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                st.image(annotated_rgb,use_container_width=True)

                num_defects = len(result.boxes)

                total_defects += num_defects

                # convert to PIL for download
                output_img = Image.fromarray(annotated_rgb)

                buf = io.BytesIO()

                output_img.save(buf,format="PNG")

                img_bytes = buf.getvalue()

                st.download_button(
                    "Download Result",
                    img_bytes,
                    file_name=f"result_{i+1}.png",
                    mime="image/png",
                    key=f"download{i}"
                )

                zip_file.writestr(
                    f"result_{i+1}.png",
                    img_bytes
                )

        # summary card

        st.markdown('<div class="result-card">',unsafe_allow_html=True)

        st.markdown("### Detection Summary")

        st.metric("Defects Detected",num_defects)

        if num_defects > 0:

            classes = result.names

            detected_classes = [
                classes[int(box.cls)] for box in result.boxes
            ]

            class_counts = {
                name: detected_classes.count(name)
                for name in set(detected_classes)
            }

            st.write("#### Defect Types")

            for cls,count in class_counts.items():

                st.write(f"**{cls}** : {count}")

        else:

            st.info("No defects detected")

        st.markdown('</div>',unsafe_allow_html=True)

        progress_bar.progress((i+1)/len(uploaded_files))

        st.markdown("---")

    zip_file.close()

    # -----------------------------
    # Global Summary
    # -----------------------------

    st.markdown("## 📊 Batch Summary")

    colA,colB = st.columns(2)

    colA.metric("Images Processed",len(uploaded_files))

    colB.metric("Total Defects Detected",total_defects)

    st.download_button(
        "📦 Download All Results (ZIP)",
        zip_buffer.getvalue(),
        file_name="aerolens_results.zip",
        mime="application/zip"
    )

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.markdown(
"""
<center>
AeroLens.ai • Wind Turbine Inspection Platform • 2026
</center>
""",
unsafe_allow_html=True
)