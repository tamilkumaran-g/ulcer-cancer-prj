import streamlit as st
from ultralytics import YOLO
from PIL import Image
import base64

# Inject CSS for background image and custom styles
def add_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main-title {{
            color: white;
            text-align: center;
            padding-top: 20px;
            text-shadow: 1px 1px 2px black;
        }}
        .markdown-text {{
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            padding: 1rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add your background image URL (public image or host yourself)
add_background("stomach.png")  # Replace with your image URL

# Load your trained YOLO model
@st.cache_resource
def load_model():
    return YOLO("gastric_ulcer_best.pt")  # your model file

model = load_model()

st.markdown("<h1 class='main-title'>🧪 Gastro Vision</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='markdown-text'>Upload an <strong>endoscopy image</strong>, and the model will predict whether it's a <strong>cancer</strong> or an <strong>ulcer</strong>.</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("📤 Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Run YOLO prediction
    results = model(image)
    top_class = results[0].probs.top1
    confidence = results[0].probs.top1conf
    label = model.names[top_class]

    # Display result
    st.success(f"🧠 Prediction: **{label}**  \n🎯 Confidence: **{confidence:.2%}**")
