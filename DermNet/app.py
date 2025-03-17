import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import base64
import torch
torch._C._jit_clear_class_registry()



# Function to get base64 string of a binary file.
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your icon file (adjust as needed).
icon_path = "icon.png"  # Ensure this file exists in your working directory.
icon_base64 = get_base64_of_bin_file(icon_path)

# ----- Custom CSS Styling with Rotated Background -----
st.markdown(
    f"""
    <style>
    /* Create a pseudo-element to display a rotated background image with a lighter gradient */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80");
        background-attachment: fixed;
        background-size: cover;
        transform: rotate(180deg);
        z-index: -1;
        opacity: 0.8;
    }}
    /* Center the title and subtitle */
    h1, p {{
        text-align: center;
    }}
    /* Style the main title */
    h1 {{
        color: #ffffff;
        font-family: 'Montserrat', sans-serif;
        font-size: 3em;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 20px;
    }}
    /* Style the uploader and camera input sections */
    .stFileUploader, .stCameraInput {{
        background-color: rgba(180, 180, 180, 0.5);
        padding: 15px;
        border-radius: 10px;
        text-color: #000000;
    }}

    /* Style buttons */
    .stButton button {{
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----- Display Title with Icon -----
st.markdown(
    f"""
    <h1 style='text-align: center;'>
      <img src='data:image/png;base64,{icon_base64}' style='height: 50px; margin-right: 10px; vertical-align: middle;'>
      Skin Condition Prediction
      <img src='data:image/png;base64,{icon_base64}' style='height: 50px; margin-left: 10px; vertical-align: middle;'>
    </h1>
    <p style='text-align: center; font-size: 1.2em;'>
      Upload an image or capture one using your camera to analyze the skin condition.
    </p>
    """,
    unsafe_allow_html=True
)

# ----- ONNX Model Loading and Preprocessing -----
# Hardcoded file path to the ONNX model.
onnx_model_path = "model.onnx"  # Adjust this path if needed.

@st.cache_resource
def load_onnx_model(path):
    session = ort.InferenceSession(path)
    return session

session = load_onnx_model(onnx_model_path)

# Get input and output names from the model.
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess_image(image, target_size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        # Normalization 
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    image = image.convert("RGB")
    tensor = preprocess(image)
    return tensor.unsqueeze(0).numpy()

# Define class labels.
class_labels = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]

# ----- Streamlit App Interface for Inference -----
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
st.info("\n\n\n OR \n\n\n")
captured_image = st.camera_input("Take a picture")

# Choose which image to process.
image = None
if uploaded_image:
    image = Image.open(uploaded_image)
elif captured_image:
    image = Image.open(captured_image)

if image:
    st.image(image, caption="Input Image", use_container_width=True)
    input_tensor = preprocess_image(image)
    
    # Run inference.
    outputs = session.run([output_name], {input_name: input_tensor})[0]
    
    # Convert logits to probabilities using softmax.
    probabilities = np.exp(outputs) / np.exp(outputs).sum(axis=1, keepdims=True)
    predicted_index = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(np.max(probabilities, axis=1)[0]) * 100
    
    # Check if confidence is less than 40%
    if confidence < 40:
        st.error("Image not suitable, please retake image")
    else:
        st.success(f"Predicted Condition: {class_labels[predicted_index]}")
        st.info(f"Confidence: {confidence:.2f}%")
