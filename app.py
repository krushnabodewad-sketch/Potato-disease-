import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="कृषी-AI : पीक संरक्षण",
    page_icon="🌿",
    layout="centered"
)

# मॉडेल लोड करणे
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('best_crop_model.h5', compile=False)

model = load_trained_model()

# Colab मधील क्लासची नावे (ज्या क्रमाने प्रिंट झाली होती तीच नावे ठेवा)
CLASS_NAMES = [
    'Cotton Bacterial Blight',
    'Potato Early Blight',
    'Potato Healthy',
    'Potato Late Blight',
    'Soybean Healthy'
]

st.markdown("""
<div style="background: linear-gradient(135deg, #073b22, #1e874b); padding: 18px; border-radius: 14px; text-align: center; color: white;">
    <h2 style="margin:0;">🌱 कृषी-AI : स्मार्ट पीक व रोग निदान</h2>
    <p style="margin:5px 0 0 0;">Custom Deep Learning Model (MobileNetV2)</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("पिकाचा फोटो निवडा (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="निवडलेला फोटो", use_container_width=True)

    with st.spinner("AI विश्लेषण करत आहे..."):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # मॉडेलमध्ये आधीच Rescaling लेयर असल्याने थेट इनपुट देणे
        predictions = model.predict(img_array)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx]) * 100
        result_label = CLASS_NAMES[predicted_idx]

        st.markdown(f"""
        <div style="background: #ffffff; padding: 16px; border-radius: 12px; border: 1px solid #c8e6c9; margin-top: 15px;">
            <h3 style="color: #1b5e20; margin: 0 0 8px 0;">निदान: {result_label}</h3>
            <p style="font-weight: bold; margin: 0; color: #333;">अचूकता (Confidence): {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
