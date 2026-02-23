# AeroLens

A computer vision project to detect the defects in wind turbines.

## 🚀 Getting Started

### 1. Model Weights
The app requires YOLOv8 weights to work. Please place your `best.pt`, `best.onnx`, or `last.pt` files in the root of the `AeroLens` directory.

### 2. Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

## ✨ Features
- **Real-time Detection**: Upload any image and get instant defect detection.
- **Adjustable Parameters**: Tweak confidence and IOU thresholds on the fly.
- **Premium UI**: Modern dark theme with detailed metrics and summaries.
