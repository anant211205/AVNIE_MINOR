# Set environment variables before importing YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["STREAMLIT_THREAD_CHECK"] = "false"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import torch
from torch import serialization
from pyzbar.pyzbar import decode

# Import ML models after environment setup
import easyocr
from ultralytics import YOLO

# Set YOLO to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Page config
st.set_page_config(page_title="FASTag Fraud Detection System", layout="wide")

# Initialize models
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        with st.spinner("Loading ML models..."):
            # Force models to use CPU
            torch.set_grad_enabled(False)
            
            # Load YOLO models with error handling
            try:
                vehicle_model = YOLO('models/vehicle_type_detector.pt')
                vehicle_model.to('cpu')
            except Exception as e:
                st.error(f"Error loading vehicle detection model: {e}")
                return None, None, None

            try:
                number_model = YOLO('models/number_plate_detector.pt')
                number_model.to('cpu')
            except Exception as e:
                st.error(f"Error loading number plate detection model: {e}")
                return None, None, None

            # Initialize EasyOCR reader
            try:
                reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                st.error(f"Error initializing EasyOCR: {e}")
                return None, None, None

            st.success("All models loaded successfully!")
            return vehicle_model, number_model, reader
    except Exception as e:
        st.error(f"Unexpected error during model loading: {e}")
        return None, None, None

# Load models
vehicle_model, number_model, reader = load_models()
if None in (vehicle_model, number_model, reader):
    st.error("Failed to initialize one or more models. Please check the logs.")
    st.stop()


# Load metadata
@st.cache_data
def load_metadata():
    try:
        # Correct format for Google Sheets CSV export
        sheet_id = "15n-TEr0VY7Mw0x9NbiG2K-Nnt5hPr36gW0pil6rbaWo"
        gid = "1854474132"
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        
        # Read CSV with error handling
        df = pd.read_csv(url, on_bad_lines='skip')
        
        # Print column names for debugging
        if df is not None:
            st.write("Available columns in metadata:", df.columns.tolist())
        
        return df
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

# Load metadata with better error handling
try:
    metadata = load_metadata()
    if metadata is None:
        st.error("Failed to load metadata. The system will continue with limited functionality.")
except Exception as e:
    st.error(f"Unexpected error while loading metadata: {e}")
    metadata = None

def detect_vehicle_type(img, model):
    try:
        results = model(img)
        if len(results[0].boxes) == 0:
            return None, 0.0
        
        cls_id = int(results[0].boxes.cls[0])
        confidence = float(results[0].boxes.conf[0])
        label = results[0].names[cls_id]
        return label, confidence
    except Exception as e:
        st.error(f"Error detecting vehicle type: {e}")
        return None, 0.0

# Image processing functions
def detect_number_plate(img, model):
    try:
        results = model(img)
        if len(results[0].boxes) == 0:
            return None, None
            
        xyxy = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        plate_img = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        return plate_img, xyxy
    except Exception as e:
        st.error(f"Error detecting number plate: {e}")
        return None, None

def preprocess_plate_image(plate_img):
    height, width = plate_img.shape[:2]
    scale = 2
    resized = cv2.resize(plate_img, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    return sharpened

def extract_number_text(plate_img):
    try:
        result = reader.readtext(plate_img, decoder='beamsearch')
        if not result:
            return None, 0.0
        text = result[0][1]
        confidence = float(result[0][2])  # EasyOCR confidence score
        return text, confidence
    except Exception as e:
        st.error(f"Error reading number plate text: {e}")
        return None, 0.0

def decode_barcode(img):
    if not isinstance(img, Image.Image):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    result = decode(img)
    if not result:
        return None
    
    return result[0].data.decode('utf-8')

def get_vehicle_type_by_number_plate(number):
    if metadata is None or number is None:
        return None
    
    row = metadata[metadata['car number plate label'] == number]
    return row['class'].values[0] if not row.empty else None

def get_vehicle_type_by_fastag_id(fastag_id):
    if metadata is None or fastag_id is None:
        return None
    
    # Try different possible column names for FASTag number
    fastag_columns = ['Fastag Number', 'fastag_number', 'FASTag_Number', 'FASTag Number', 'fastagno']
    vehicle_type_columns = ['Fastag class', 'fastag_class', 'vehicle_type', 'Vehicle Type', 'class']
    
    # Find the correct column names
    fastag_col = next((col for col in fastag_columns if col in metadata.columns), None)
    vehicle_type_col = next((col for col in vehicle_type_columns if col in metadata.columns), None)
    
    if fastag_col is None or vehicle_type_col is None:
        st.error(f"Required columns not found in metadata. Available columns: {metadata.columns.tolist()}")
        return None
    
    row = metadata[metadata[fastag_col] == fastag_id]
    return row[vehicle_type_col].values[0] if not row.empty else None

def process_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


# UI Layout
st.title("FASTag Fraud Detection System")
st.write("Upload images to detect potential fraud in vehicle classification")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Vehicle Image")
    vehicle_img_file = st.file_uploader("Choose a vehicle image", type=['jpg', 'jpeg', 'png'])
    
with col2:
    st.subheader("Upload FASTag Barcode Image")
    barcode_img_file = st.file_uploader("Choose a FASTag barcode image", type=['jpg', 'jpeg', 'png'])

st.subheader("Or use an example image")
use_example = st.checkbox("Use example images")

if use_example:
    st.info("Note: Example functionality would need to be implemented with actual dataset paths")

if st.button("Check for Fraud"):
    if (vehicle_img_file is not None and barcode_img_file is not None) or use_example:
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Processing vehicle image...")
        
        if use_example:
            st.warning("Example functionality not implemented yet. Please upload images.")
            progress_bar.empty()
            status_text.empty()
        else:
            # Initialize variables
            vehicle_type_yolo = None
            vehicle_type_ft = None
            vehicle_type_np = None
            
            vehicle_img = process_uploaded_image(vehicle_img_file)
            vehicle_img_file.seek(0)
            
            vehicle_type_yolo, vt_confidence = detect_vehicle_type(vehicle_img, vehicle_model)
            
            st.image(vehicle_img_file, caption="Vehicle Image", use_column_width=True)
            st.info(f"Detected Vehicle Type (YOLO): {vehicle_type_yolo} (Confidence: {vt_confidence:.2f})")
            
            progress_bar.progress(0.3)
            status_text.text("Processing FASTag barcode...")
            
            barcode_img = process_uploaded_image(barcode_img_file)
            barcode_img_file.seek(0)
            
            st.image(barcode_img_file, caption="FASTag Barcode", use_column_width=True)
            
            fastag_id = decode_barcode(barcode_img)
            if fastag_id:
                st.info(f"Decoded FASTag ID: {fastag_id}")
                
                vehicle_type_ft = get_vehicle_type_by_fastag_id(fastag_id)
                if vehicle_type_ft:
                    st.info(f"Vehicle Type from FASTag Metadata: {vehicle_type_ft}")
                else:
                    st.warning("FASTag ID not found in metadata")
                    vehicle_type_ft = None
            else:
                st.error("Could not decode barcode. Please upload a clearer image.")
                vehicle_type_ft = None
            
            progress_bar.progress(0.6)
            
            if vehicle_type_yolo and vehicle_type_ft and vehicle_type_yolo == vehicle_type_ft:
                st.success("✅ No Fraud Detected: Vehicle type from image matches FASTag record")
                progress_bar.progress(1.0)
                status_text.empty()
            else:
                status_text.text("Mismatch detected! Checking number plate...")
                
                plate_img, plate_coords = detect_number_plate(vehicle_img, number_model)
                
                if plate_img is not None and plate_img.size > 0:
                    vehicle_with_box = vehicle_img.copy()
                    cv2.rectangle(vehicle_with_box, 
                                 (plate_coords[0], plate_coords[1]), 
                                 (plate_coords[2], plate_coords[3]), 
                                 (0, 255, 0), 2)
                    
                    st.image(vehicle_with_box, caption="Vehicle with Detected Number Plate", use_column_width=True)
                    
                    plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                    st.image(plate_rgb, caption="Extracted Number Plate", use_column_width=True)
                    
                    plate_text, ocr_confidence = extract_number_text(plate_img)
                    
                    if plate_text:
                        st.info(f"Detected Number Plate: {plate_text} (Confidence: {ocr_confidence:.2f})")
                        
                        vehicle_type_np = get_vehicle_type_by_number_plate(plate_text)
                        
                        if vehicle_type_np:
                            st.info(f"Vehicle Type from Number Plate Metadata: {vehicle_type_np}")
                            
                            if (vehicle_type_np == vehicle_type_ft) or (vehicle_type_np == vehicle_type_yolo):
                                st.warning("⚠️ Possible mismatch but number plate supports one of the sources")
                            else:
                                st.error("🚨 Fraud detected: All three sources disagree on vehicle type")
                        else:
                            st.warning("Number plate not found in metadata")
                            st.error("🚨 Cannot verify - Number plate not in database")
                    else:
                        st.error("Could not read text from number plate")
                        if vehicle_type_yolo != vehicle_type_ft:
                            st.error("🚨 Potential fraud: Vehicle type mismatch between image and FASTag")
                else:
                    st.error("Could not detect number plate")
                    if vehicle_type_yolo != vehicle_type_ft:
                        st.error("🚨 Potential fraud: Vehicle type mismatch between image and FASTag")
            
            # Final fraud detection logic
            if vehicle_type_yolo and vehicle_type_ft:  # Only check if both main identifiers are present
                if vehicle_type_yolo != vehicle_type_ft:  # If main comparison fails
                    if vehicle_type_np:  # If we have number plate data
                        if vehicle_type_np != vehicle_type_yolo and vehicle_type_np != vehicle_type_ft:
                            st.error("🚨 Fraud detected: All three sources disagree on vehicle type")
                    else:  # No number plate data
                        st.error("🚨 Fraud detected: Vehicle type mismatch between image and FASTag")
    else:
        st.error("Please upload both images for vehicle and FASTag barcode")

# Add system information
st.sidebar.title("System Information")
st.sidebar.info("""
**FASTag Fraud Detection System**

This system uses:
- YOLO for vehicle type detection
- YOLO for license plate detection
- EasyOCR for license plate reading
- PyZBar for barcode scanning
- Metadata lookup for verification

Upload images of a vehicle and its FASTag barcode to check for potential fraud.
""")