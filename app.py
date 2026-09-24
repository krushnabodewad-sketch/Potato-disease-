import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# १. पेज संरचना
st.set_page_config(
    page_title="कृषी-AI : स्मार्ट पीक संरक्षण",
    page_icon="🌿",
    layout="centered"
)

# २. स्टायलिंग
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700;800&family=Poppins:wght@500;600;700&display=swap');
    * { font-family: 'Mukta', 'Poppins', sans-serif; }
    .stApp { background-color: #f4f8f4; }
    .hero-banner {
        background: linear-gradient(135deg, #073b22, #1e874b);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        color: white;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .main-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #d9edd9;
        margin-bottom: 14px;
    }
    .res-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #c8e6c9;
        margin-top: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .badge-crop { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
</style>
""", unsafe_allow_html=True)

# ३. मॉडेल्स लोड करणे
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
    st.error(f"मॉडेल लोड करताना त्रुटी आली: {e}")

# लेबल्स
POTATO_CLASSES = ['Potato Early Blight (बटाटा करपा)', 'Potato Late Blight (बटाटा उशिरा करपा)', 'Potato Healthy Leaf (निरोगी बटाटा पान)']
COTTON_CLASSES = ['Diseased Cotton Leaf (रोगग्रस्त कापूस पान)', 'Diseased Cotton Plant (रोगग्रस्त कापूस झाड)', 'Fresh Cotton Leaf (निरोगी कापूस पान)', 'Fresh Cotton Plant (निरोगी कापूस झाड)']
SOYBEAN_CLASSES = ['Soybean Caterpillar Damage (सोयाबीन पान - अळी प्रादुर्भाव)', 'Soybean Leaf Beetle Damage (सोयाबीन पान - भुंगा प्रादुर्भाव)', 'Soybean Healthy Leaf (निरोगी सोयाबीन पान)']

# ४. हेडर
st.markdown("""
<div class="hero-banner">
    <h2 style="margin:0; font-size: 22px;">🌱 कृषी-AI : ऑटो पीक व रोग निदान प्रणाली</h2>
    <p style="margin:5px 0 0 0; font-size: 13px; color: #d1f2e2;">आविष्कार संशोधन प्रकल्प • ऑटो-डिटेक्ट डीप लर्निंग व्हिजन</p>
</div>
""", unsafe_allow_html=True)

# ५. इनपुट सेक्शन
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("##### 📷 पिकाच्या पानाचा फोटो अपलोड करा (ऑटोमॅटिक डिटेक्शन):")
uploaded_file = st.file_uploader("छायाचित्र निवडा (JPG / PNG)", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# ६. ऑटो-डिटेक्शन आणि क्लासिफिकेशन लॉजिक
if uploaded_file is not None and models_ready:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

    with st.spinner("🤖 AI आपोआप पिकाचा प्रकार व रोग ओळखत आहे..."):
        resized_img = img.resize((224, 224))
        arr = np.array(resized_img, dtype=np.float32)

        # १. बटाटा मॉडेल प्रेडिक्शन
        preds_p = potato_model(np.expand_dims(arr, axis=0), training=False).numpy()[0]
        if np.sum(preds_p) > 1.05 or np.sum(preds_p) < 0.95:
            preds_p = tf.nn.softmax(preds_p).numpy()
        idx_p = int(np.argmax(preds_p))
        conf_p = float(preds_p[idx_p])

        # २. सोयाबीन मॉडेल प्रेडिक्शन (०-१ नॉर्मलायझेशन)
        preds_s = soybean_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
        if np.sum(preds_s) > 1.05 or np.sum(preds_s) < 0.95:
            preds_s = tf.nn.softmax(preds_s).numpy()
        idx_s = int(np.argmax(preds_s))
        conf_s = float(preds_s[idx_s])

        # ३. कापूस मॉडेल प्रेडिक्शन (०-१ नॉर्मलायझेशन)
        preds_c = cotton_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
        if np.sum(preds_c) > 1.05 or np.sum(preds_c) < 0.95:
            preds_c = tf.nn.softmax(preds_c).numpy()
        idx_c = int(np.argmax(preds_c))
        conf_c = float(preds_c[idx_c])

        # --- स्मार्ट ऑटो-डिटेक्ट कॅलिब्रेशन ---
        # कापूस मॉडेल इतर साध्या हिरव्या पानांना ५०-७०% फॉल्स स्कोअर देते, म्हणून त्याला संतुलित केले आहे:
        score_potato = conf_p * 1.35
        score_soybean = conf_s * 1.30
        score_cotton = conf_c * 0.70  # कापूस फॉल्स-पॉझिटिव्ह पेनल्टी

        if score_soybean >= score_potato and score_soybean >= score_cotton:
            crop_name = "🌱 सोयाबीन (Soybean Leaf)"
            diagnosed_label = SOYBEAN_CLASSES[idx_s]
            final_conf = conf_s * 100
        elif score_potato >= score_soybean and score_potato >= score_cotton:
            crop_name = "🥔 बटाटा (Potato Leaf)"
            diagnosed_label = POTATO_CLASSES[idx_p]
            final_conf = conf_p * 100
        else:
            crop_name = "☁️ कापूस (Cotton Leaf)"
            diagnosed_label = COTTON_CLASSES[idx_c]
            final_conf = conf_c * 100

        # निकाल दाखवणे
        st.markdown(f"""
        <div class="res-card">
            <span class="badge badge-crop">🌾 ओळखलेले पीक: {crop_name}</span>
            <h3 style="color: #1b5e20; margin: 8px 0 10px 0; font-size: 19px;">📋 स्थिती व निदान: {diagnosed_label}</h3>
            <div style="background: #eef7ee; padding: 8px 12px; border-radius: 8px; display: inline-block;">
                <span style="font-weight: 700; color: #2e7d32;">अचूकता (Confidence): {final_conf:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
