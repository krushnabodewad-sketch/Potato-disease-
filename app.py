import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Config
st.set_page_config(
    page_title="Krushi-AI",
    page_icon="🌿",
    layout="centered"
)

# 2. Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700&display=swap');
    * { font-family: 'Mukta', sans-serif; }
    .stApp { background-color: #f4f8f4; }
    .hero-banner {
        background: linear-gradient(135deg, #073b22, #1e874b);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        color: white;
        margin-bottom: 16px;
    }
    .main-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px;
        border: 1px solid #d9edd9;
        margin-bottom: 14px;
    }
    .res-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px;
        border: 1px solid #c8e6c9;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Existing .h5 Models Load Karne
@st.cache_resource
def load_all_models():
    p_m = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False)
    c_m = tf.keras.models.load_model('cotton_model.h5', compile=False)
    s_m = tf.keras.models.load_model('soybean_model.h5', compile=False)
    return p_m, c_m, s_m

try:
    potato_model, cotton_model, soybean_model = load_all_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"Model load zale nahi: {e}")

# Class Labels
POTATO_CLASSES = ['Potato Early Blight (करपा)', 'Potato Late Blight (उशिरा करपा)', 'Potato Healthy (निरोगी बटाटा)']
COTTON_CLASSES = ['Diseased Cotton Leaf (करपा/रोग)', 'Diseased Cotton Plant (रोगग्रस्त झाड)', 'Fresh Cotton Leaf (निरोगी पान)', 'Fresh Cotton Plant (निरोगी झाड)']
SOYBEAN_CLASSES = ['Soybean Caterpillar (अळी)', 'Soybean Leaf Beetle (भुंगा)', 'Soybean Healthy (निरोगी सोयाबीन)']

# 4. Header
st.markdown("""
<div class="hero-banner">
    <h2>🌱 कृषी-AI : पीक व रोग निदान प्रणाली</h2>
    <p>आविष्कार संशोधन प्रकल्प • Deep Learning (.h5) Models</p>
</div>
""", unsafe_allow_html=True)

# 5. Crop Selection (Kontech galat prediction honar nahi)
st.markdown('<div class="main-card">', unsafe_allow_html=True)
crop_choice = st.selectbox(
    "🌾 पिकाचा प्रकार निवडा:",
    ("🥔 बटाटा (Potato)", "🌱 सोयाबीन (Soybean)", "☁️ कापूस (Cotton)")
)

uploaded_file = st.file_uploader("पिकाच्या पानाचा फोटो अपलोड करा (JPG/PNG)", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# 6. Prediction Logic
if uploaded_file is not None and models_ready:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="निवडलेले छायाचित्र", use_container_width=True)

    with st.spinner("AI मॉडेलद्वारे निदान सुरू आहे..."):
        resized_img = img.resize((224, 224))
        arr = np.array(resized_img, dtype=np.float32)

        if "बटाटा" in crop_choice:
            input_tensor = np.expand_dims(arr, axis=0)
            preds = potato_model(input_tensor, training=False).numpy()[0]
            if np.sum(preds) > 1.05 or np.sum(preds) < 0.95:
                preds = tf.nn.softmax(preds).numpy()
            idx = int(np.argmax(preds))
            label = POTATO_CLASSES[idx]
            conf = float(preds[idx]) * 100

        elif "सोयाबीन" in crop_choice:
            input_tensor = np.expand_dims(arr / 255.0, axis=0)
            preds = soybean_model(input_tensor, training=False).numpy()[0]
            if np.sum(preds) > 1.05 or np.sum(preds) < 0.95:
                preds = tf.nn.softmax(preds).numpy()
            idx = int(np.argmax(preds))
            label = SOYBEAN_CLASSES[idx]
            conf = float(preds[idx]) * 100

        else:
            input_tensor = np.expand_dims(arr / 255.0, axis=0)
            preds = cotton_model(input_tensor, training=False).numpy()[0]
            if np.sum(preds) > 1.05 or np.sum(preds) < 0.95:
                preds = tf.nn.softmax(preds).numpy()
            idx = int(np.argmax(preds))
            label = COTTON_CLASSES[idx]
            conf = float(preds[idx]) * 100

        st.markdown(f"""
        <div class="res-card">
            <h3 style="color: #1b5e20; margin: 0 0 8px 0;">✅ निदान: {label}</h3>
            <p style="font-weight: bold; margin: 0; color: #2e7d32; font-size: 16px;">अचूकता (Confidence): {conf:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
