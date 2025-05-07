import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import easyocr
from pyzbar.pyzbar import decode
import pandas as pd
import io

# Page config
st.set_page_config(page_title="FASTag Fraud Detection System", layout="wide")

# Initialize models
@st.cache_resource
def load_models():
    vehicle_model = YOLO('models/vehicle_type_detector.pt')
    number_model = YOLO('models/number_plate_detector.pt')
    reader = easyocr.Reader(['en'], gpu=True)
    return vehicle_model, number_model, reader

vehicle_model, number_model, reader = load_models()

# Load metadata
@st.cache_data
def load_metadata():
    sheet_id = "1C3AVLoTkFq-Y8XzP1wYpelZXtmTV3bcv"
    sheet_name = "metadata1"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

metadata = load_metadata()

# Image processing functions
def detect_vehicle_type(img, model):
    results = model(img)
    if len(results[0].boxes) == 0:
        return "Unknown", 0.0
    
    # Get class with highest confidence
    cls_id = int(results[0].boxes.cls[0])
    confidence = float(results[0].boxes.conf[0])
    label = results[0].names[cls_id]
    
    return label, confidence

def detect_number_plate(img, model):
    results = model(img)
    if len(results[0].boxes) == 0:
        return None, None
    
    # Convert from tensor to numpy array if needed
    if hasattr(results[0].boxes.xyxy, 'cpu'):
        xyxy = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    else:
        xyxy = results[0].boxes.xyxy[0].numpy().astype(int)
    
    # Get the cropped plate image
    if isinstance(img, str):
        img_cv = cv2.imread(img)
    else:
        img_cv = img
        
    h, w = img_cv.shape[:2]
    # Ensure coordinates are within image bounds
    x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
    plate_img = img_cv[y1:y2, x1:x2]
    
    return plate_img, xyxy

def preprocess_plate_image(plate_img):
    # Resize plate image (for better OCR)
    height, width = plate_img.shape[:2]
    scale = 2
    resized = cv2.resize(plate_img, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)
    
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    return sharpened

def extract_number_text(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None, 0.0
    
    # Try with different preprocessing methods
    preprocessed = preprocess_plate_image(plate_img)
    
    # OCR with beamsearch for better accuracy
    result = reader.readtext(preprocessed, decoder='beamsearch')
    
    if not result:
        return None, 0.0
    
    # Return the text with highest confidence
    text = result[0][1]
    confidence = result[0][2]
    return text, confidence

def decode_barcode(img):
    # Convert to PIL Image if needed
    if not isinstance(img, Image.Image):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Decode barcode
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
    
    row = metadata[metadata['Fastag Number'] == fastag_id]
    return row['Fastag class'].values[0] if not row.empty else None

def process_uploaded_image(uploaded_file):
    # Read the file into an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

# UI Layout
st.title("FASTag Fraud Detection System")
st.write("Upload images to detect potential fraud in vehicle classification")

# Create columns for file uploaders
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Vehicle Image")
    vehicle_img_file = st.file_uploader("Choose a vehicle image", type=['jpg', 'jpeg', 'png'])
    
with col2:
    st.subheader("Upload FASTag Barcode Image")
    barcode_img_file = st.file_uploader("Choose a FASTag barcode image", type=['jpg', 'jpeg', 'png'])

# Display example selection
st.subheader("Or use an example image")
use_example = st.checkbox("Use example images")

if use_example:
    # Here you would provide some example images from your dataset
    # You would need to adapt this to your actual dataset structure
    st.info("Note: Example functionality would need to be implemented with actual dataset paths")

# Process button
if st.button("Check for Fraud"):
    if (vehicle_img_file is not None and barcode_img_file is not None) or use_example:
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process vehicle image
        status_text.text("Processing vehicle image...")
        
        if use_example:
            # Here you would load example images
            # This is placeholder code
            # vehicle_img = cv2.imread('/content/vehicle_dataset_processed/images/val/some_image.jpg')
            # barcode_img = cv2.imread('/path/to/example/barcode.jpg')
            st.warning("Example functionality not implemented yet. Please upload images.")
            progress_bar.empty()
            status_text.empty()
        else:
            # Process uploaded images
            vehicle_img = process_uploaded_image(vehicle_img_file)
            vehicle_img_file.seek(0)  # Reset file pointer for display
            
            # 1. Detect vehicle type from image using YOLO
            vehicle_type_yolo, vt_confidence = detect_vehicle_type(vehicle_img, vehicle_model)
            
            # Display vehicle image and detection
            st.image(vehicle_img_file, caption="Vehicle Image", use_column_width=True)
            st.info(f"Detected Vehicle Type (YOLO): {vehicle_type_yolo} (Confidence: {vt_confidence:.2f})")
            
            progress_bar.progress(0.3)
            status_text.text("Processing FASTag barcode...")
            
            # 2. Process barcode image
            barcode_img = process_uploaded_image(barcode_img_file)
            barcode_img_file.seek(0)  # Reset file pointer for display
            
            # Display barcode image
            st.image(barcode_img_file, caption="FASTag Barcode", use_column_width=True)
            
            # 3. Decode FASTag ID
            fastag_id = decode_barcode(barcode_img)
            if fastag_id:
                st.info(f"Decoded FASTag ID: {fastag_id}")
                
                # 4. Get vehicle type from FASTag metadata
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
            
            # 5. Compare vehicle types from YOLO and FASTag
            if vehicle_type_yolo and vehicle_type_ft and vehicle_type_yolo == vehicle_type_ft:
                st.success("✅ No Fraud Detected: Vehicle type from image matches FASTag record")
                progress_bar.progress(1.0)
                status_text.empty()
            else:
                status_text.text("Mismatch detected! Checking number plate...")
                
                # 6. If types don't match, extract and check number plate
                plate_img, plate_coords = detect_number_plate(vehicle_img, number_model)
                
                if plate_img is not None and plate_img.size > 0:
                    # Draw bounding box on original image
                    vehicle_with_box = vehicle_img.copy()
                    cv2.rectangle(vehicle_with_box, 
                                 (plate_coords[0], plate_coords[1]), 
                                 (plate_coords[2], plate_coords[3]), 
                                 (0, 255, 0), 2)
                    
                    st.image(vehicle_with_box, caption="Vehicle with Detected Number Plate", use_column_width=True)
                    
                    # Display the cropped plate
                    plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                    st.image(plate_rgb, caption="Extracted Number Plate", use_column_width=True)
                    
                    # Extract text from plate
                    plate_text, ocr_confidence = extract_number_text(plate_img)
                    
                    if plate_text:
                        st.info(f"Detected Number Plate: {plate_text} (Confidence: {ocr_confidence:.2f})")
                        
                        # Get vehicle type from number plate metadata
                        vehicle_type_np = get_vehicle_type_by_number_plate(plate_text)
                        
                        if vehicle_type_np:
                            st.info(f"Vehicle Type from Number Plate Metadata: {vehicle_type_np}")
                            
                            # Check for fraud based on all three sources
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
                
                progress_bar.progress(1.0)
                status_text.empty()
    else:
        st.warning("Please upload both vehicle and FASTag barcode images or select the example option")

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