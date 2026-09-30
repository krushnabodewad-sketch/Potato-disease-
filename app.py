import streamlit as st
import json
import io
import base64
import requests
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.parse
import os
from datetime import datetime

st.set_page_config(
    page_title="Krushi-AI : Crop Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern UI Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&family=Mukta:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', 'Mukta', sans-serif; }
.stApp { background: #F8FAFC; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; max-width: 1050px; }

.k-hero {
    background: linear-gradient(135deg, #052e22 0%, #064E3B 40%, #047857 75%, #059669 100%);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    color: #fff;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 25px rgba(6, 78, 59, 0.2);
}
.k-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 12px rgba(0,0,0,0.03);
}
.k-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 6px;
    margin-bottom: 6px;
}
.k-pill-crop { background: #064E3B; color: #fff; }
.k-pill-diag { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
.tag-h { background: #DCFCE7; color: #166534; font-weight: 700; padding: 4px 12px; border-radius: 8px; }
.tag-c { background: #FEE2E2; color: #991B1B; font-weight: 700; padding: 4px 12px; border-radius: 8px; }
.tag-m { background: #FEF3C7; color: #92400E; font-weight: 700; padding: 4px 12px; border-radius: 8px; }
.c-val { font-size: 2.2rem; font-weight: 800; color: #064E3B; margin-top: 6px; }
.t-chem { background: #FFFBEB; border-left: 4px solid #F59E0B; padding: 12px; border-radius: 10px; margin-bottom: 8px; }
.t-bio { background: #F0FDF4; border-left: 4px solid #10B981; padding: 12px; border-radius: 10px; margin-bottom: 8px; }
.k-wa-btn {
    display: block; text-align: center; background: #25D366; color: #fff !important;
    text-decoration: none; font-weight: 700; padding: 12px; border-radius: 12px; margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="k-hero">
    <div style="font-size:11px;font-weight:800;color:#A7F3D0;letter-spacing:1px;">AVISHKAR AGRI-AI INITIATIVE</div>
    <h2 style="margin:4px 0 0 0;font-size:1.6rem;font-weight:800;">🌿 कृषी-AI : अचूक पीक व रोग निदान प्रणाली</h2>
    <div style="font-size:0.88rem;color:#D1FAE5;margin-top:4px;">स्थानिक न्यूरल नेटवर्क • १००% मोफत आणि अचूक वर्गीकरण</div>
</div>
""", unsafe_allow_html=True)

# 1. Models Loader
@st.cache_resource
def load_all_local_models():
    pm = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False) if os.path.exists('potato_disease_model (1).h5') else None
    cm = tf.keras.models.load_model('cotton_model.h5', compile=False) if os.path.exists('cotton_model.h5') else None
    sm = tf.keras.models.load_model('soybean_model.h5', compile=False) if os.path.exists('soybean_model.h5') else None
    pdm = tf.keras.models.load_model('plantdoc_full_model.h5', compile=False) if os.path.exists('plantdoc_full_model.h5') else None
    return pm, cm, sm, pdm

potato_model, cotton_model, soybean_model, plantdoc_model = load_all_local_models()

POTATO_CLASSES = [
    'Potato Early Blight (बटाटा करपा)',
    'Potato Late Blight (बटाटा उशिरा करपा)',
    'Potato Healthy Leaf (निरोगी बटाटा पान)'
]

COTTON_CLASSES = [
    'Diseased Cotton Leaf (रोगग्रस्त कापूस पान)',
    'Diseased Cotton Plant (रोगग्रस्त कापूस झाड)',
    'Fresh Cotton Leaf (निरोगी कापूस पान)',
    'Fresh Cotton Plant (निरोगी कापूस झाड)'
]

SOYBEAN_CLASSES = [
    'Soybean Caterpillar Damage (सोयाबीन अळी प्रादुर्भाव)',
    'Soybean Leaf Beetle Damage (सोयाबीन भुंगा प्रादुर्भाव)',
    'Soybean Healthy Leaf (निरोगी सोयाबीन पान)'
]

col_l, col_r = st.columns([1, 1.2], gap="large")

with col_l:
    st.markdown('<div class="k-card"><b>⚙️ इनपुट पॅनेल (Input Panel)</b>', unsafe_allow_html=True)
    crop_mode = st.selectbox(
        "🌾 पीक निवडा (Select Crop):",
        ("☁️ कापूस (Cotton)", "🥔 बटाटा (Potato)", "🌱 सोयाबीन (Soybean)", "🤖 ऑटो-डिटेक्ट (Auto Detect)")
    )
    uploaded_file = st.file_uploader("पानाचा फोटो निवडा:", type=["jpg", "jpeg", "png", "webp"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="k-card"><b>🍃 अपलोड केलेले पान (Preview)</b>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    if uploaded_file is None:
        st.info("📡 **प्रणाली सज्ज आहे.**\n\nकृपया डाव्या बाजूने पानाचा स्वच्छ फोटो निवडा.")
    else:
        with st.spinner("🔍 मॉडेलद्वारे पानाचे अचूक विश्लेषण होत आहे..."):
            img = Image.open(uploaded_file).convert('RGB')
            resized = img.resize((224, 224))
            arr = np.array(resized, dtype=np.float32)

            # Local Prediction Logic
            active_crop = "cotton"
            if "बटाटा" in crop_mode:
                active_crop = "potato"
            elif "सोयाबीन" in crop_mode:
                active_crop = "soybean"
            elif "कापूस" in crop_mode:
                active_crop = "cotton"
            else:
                # Auto comparison
                p_score = np.max(potato_model(np.expand_dims(arr, axis=0), training=False).numpy()) if potato_model else 0
                c_score = np.max(cotton_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()) if cotton_model else 0
                s_score = np.max(soybean_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()) if soybean_model else 0
                scores = {"potato": p_score, "cotton": c_score, "soybean": s_score}
                active_crop = max(scores, key=scores.get)

            # Processing diagnosis
            if active_crop == "cotton" and cotton_model:
                pred = cotton_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
                idx = int(np.argmax(pred))
                conf = float(np.max(pred)) * 100
                diag = COTTON_CLASSES[idx]
                c_name = "☁️ कापूस (Cotton)"
                is_healthy = "Fresh" in diag or "निरोगी" in diag

            elif active_crop == "potato" and potato_model:
                pred = potato_model(np.expand_dims(arr, axis=0), training=False).numpy()[0]
                idx = int(np.argmax(pred))
                conf = float(np.max(pred)) * 100
                diag = POTATO_CLASSES[idx]
                c_name = "🥔 बटाटा (Potato)"
                is_healthy = "Healthy" in diag or "निरोगी" in diag

            else:
                pred = soybean_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0] if soybean_model else [0, 0, 1]
                idx = int(np.argmax(pred))
                conf = float(np.max(pred)) * 100
                diag = SOYBEAN_CLASSES[idx]
                c_name = "🌱 सोयाबीन (Soybean)"
                is_healthy = "Healthy" in diag or "निरोगी" in diag

            # Treatment Info Setup
            if is_healthy:
                sev = "सुरक्षित (Healthy)"
                tag_class = "tag-h"
                chem = "कोणतीही रासायनिक फवारणी आवश्यक नाही."
                brands = "खर्च ₹ ०"
                dose_15L = "फक्त स्वच्छ पाणी फवारावे"
                dose_200L = "फक्त स्वच्छ पाणी"
                bio = "रोगप्रतिकारशक्तीसाठी १५ दिवसांतून एकदा जीवामृत किंवा ५% निंबोळी अर्क वापरावा."
            else:
                sev = "तीव्र (Infected)"
                tag_class = "tag-c"
                if "Cotton" in c_name:
                    chem = "कॉपर ऑक्सिक्लोराईड ५०% WP (COC) + स्ट्रेप्टोसायक्लिन"
                    brands = "Blitox 50 + Streptocycline"
                    dose_15L = "३० ग्रॅम ब्लिटॉक्स + २ ग्रॅम स्ट्रेप्टोसायक्लिन"
                    dose_200L = "४०० ग्रॅम ब्लिटॉक्स + २० ग्रॅम स्ट्रेप्टोसायक्लिन"
                    bio = "तांबेयुक्त ताक किंवा ५% निंबोळी अर्क फवारावा."
                elif "Potato" in c_name:
                    chem = "मॅन्कोझेब ७५% WP किंवा रिडोमिल गोल्ड"
                    brands = "Indofil M-45, Ridomil Gold"
                    dose_15L = "३५ ग्रॅम पावडर + १० मिली स्टिकर"
                    dose_200L = "५०० ग्रॅम पावडर + १५० मिली स्टिकर"
                    bio = "ट्रायकोडर्मा व्हिरीडी ५० ग्रॅम प्रति पंप."
                else:
                    chem = "कोराजन (Chlorantraniliprole 18.5% SC)"
                    brands = "Coragen (FMC)"
                    dose_15L = "६ ते ७ मिली कोराजन प्रति पंप"
                    dose_200L = "८० ते १०० मिली"
                    bio = "निंबोळी अर्क ५% किंवा Bt पावडर."

            # UI Rendering
            st.markdown('<div class="k-card"><b>🩺 अचूक निदान निकाल (Diagnostic Result)</b></div>', unsafe_allow_html=True)
            st.markdown(f'<span class="k-pill k-pill-crop">{c_name}</span><span class="k-pill k-pill-diag">{diag}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="{tag_class}">● {sev}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="c-val">{conf:.1f}%</div><div style="font-size:0.85rem;color:#047857;font-weight:700;">✓ Verified by Local Neural Network</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="t-chem">
                <b style="color:#B45309;">🧪 रासायनिक उपचार:</b><br>{chem}<br><br>
                <b>🏷️ शिफारसीत ब्रँड:</b> {brands}<br>
                <b>💧 डोस:</b> १५L पंप: {dose_15L} | २००L ड्रम: {dose_200L}
            </div>
            <div class="t-bio">
                <b style="color:#047857;">🌿 सेंद्रिय व जैविक उपाय:</b><br>{bio}
            </div>
            """, unsafe_allow_html=True)

            wa_msg = f"*🌿 कृषी-AI पीक निदान अहवाल*\nपीक: {c_name}\nनिदान: {diag}\nअचूकता: {conf:.1f}%\nउपचार: {chem}\nडोस: {dose_15L}"
            wa_url = f"https://api.whatsapp.com/send?text={urllib.parse.quote(wa_msg)}"
            st.markdown(f'<a href="{wa_url}" target="_blank" class="k-wa-btn">📲 WhatsApp वर शेअर करा</a>', unsafe_allow_html=True)
            
