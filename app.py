import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Config
st.set_page_config(
    page_title="Krushi-AI : Smart Crop Protection",
    page_icon="🌿",
    layout="centered"
)

# 2. Modern Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700;800&family=Poppins:wght@500;600;700&display=swap');
    * { font-family: 'Mukta', sans-serif; }
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
    .treatment-box {
        background: #f1f8e9;
        border-left: 5px solid #2e7d32;
        padding: 12px 14px;
        border-radius: 8px;
        margin-top: 14px;
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
    .prob-bar-container {
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Models
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

# Labels
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
    'Soybean Caterpillar Damage (सोयाबीन पान - अळी प्रादुर्भाव)',
    'Soybean Leaf Beetle Damage (सोयाबीन पान - भुंगा प्रादुर्भाव)',
    'Soybean Healthy Leaf (निरोगी सोयाबीन पान)'
]

# 4. Fertilizers & Treatment Database
TREATMENTS = {
    'Potato Early Blight (बटाटा करपा)': {
        'fertilizer': 'Mancozeb 75% WP (M-45) / Chlorothalonil 75% WP',
        'dose': '२ ते २.५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३०-३५ ग्रॅम)',
        'tips': 'पानावरील काळे गोलाकार चट्टे थांबवण्यासाठी ८-१० दिवसांनी दुसरी फवारणी करावी.'
    },
    'Potato Late Blight (बटाटा उशिरा करपा)': {
        'fertilizer': 'Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold)',
        'dose': '२.५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३५-४० ग्रॅम)',
        'tips': 'झाडाच्या बुंध्याजवळ पाणी साचू देऊ नये; प्रादुर्भाव वाढल्यास Cymoxanil घटक वापरावा.'
    },
    'Potato Healthy Leaf (निरोगी बटाटा पान)': {
        'fertilizer': '19:19:19 (Water Soluble NPK) + Micronutrients',
        'dose': '५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ७०-७५ ग्रॅम)',
        'tips': 'झाडांची वाढ जोमदार राहण्यासाठी संतुलित खते आणि पाण्याचे नियोजन ठेवा.'
    },
    'Diseased Cotton Leaf (रोगग्रस्त कापूस पान)': {
        'fertilizer': 'Copper Oxychloride 50% WP (COC) + Streptocycline',
        'dose': 'COC ३० ग्रॅम + स्ट्रेप्टोसायक्लिन २ ग्रॅम (प्रति १५ लिटर पंप)',
        'tips': 'जिवाणूजन्य करपा (Bacterial Blight) रोखण्यासाठी स्वच्छ सूर्यप्रकाशात फवारणी करावी.'
    },
    'Diseased Cotton Plant (रोगग्रस्त कापूस झाड)': {
        'fertilizer': 'Carbendazim 12% + Mancozeb 63% WP (SAAF)',
        'dose': '२ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३० ग्रॅम)',
        'tips': 'रोगग्रस्त फांद्या छाटून नष्ट कराव्यात; नत्राचा अतिवापर टाळावा.'
    },
    'Fresh Cotton Leaf (निरोगी कापूस पान)': {
        'fertilizer': '13:00:45 (Potassium Nitrate) + Boron 20%',
        'dose': '१३:००:४५ ५ ग्रॅम + बोरॉन १ ग्रॅम प्रति लिटर पाणी',
        'tips': 'पाने तजेलदार ठेवण्यासाठी आणि बोंड गळ थांबवण्यासाठी उपयुक्त.'
    },
    'Fresh Cotton Plant (निरोगी कापूस झाड)': {
        'fertilizer': '12:61:00 (MAP) / Seaweed Extract',
        'dose': '४ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ६० ग्रॅम)',
        'tips': 'झाडांची रोगप्रतिकारक शक्ती आणि पांढऱ्या मुळ्या वाढवण्यासाठी उत्तम.'
    },
    'Soybean Caterpillar Damage (सोयाबीन पान - अळी प्रादुर्भाव)': {
        'fertilizer': 'Chlorantraniliprole 18.5% SC (Coragen) / Emamectin Benzoate 5% SG',
        'dose': 'कोराजन ६ मिली किंवा इमामेक्टिन बेन्झोएट १० ग्रॅम (प्रति १५ लिटर पंप)',
        'tips': 'पाने खाणाऱ्या तंबाखूवरील व हिरव्या अळीचा तात्काळ बंदोबस्त होतो.'
    },
    'Soybean Leaf Beetle Damage (सोयाबीन पान - भुंगा प्रादुर्भाव)': {
        'fertilizer': 'Lambda Cyhalothrin 4.9% CS / Quinalphos 25% EC',
        'dose': 'लॅम्बडा सायहेलोथ्रीन १५ मिली किंवा क्विनॉलफॉस ३० मिली (प्रति १५ लिटर पंप)',
        'tips': 'भुंग्यांचा प्रादुर्भाव रोखण्यासाठी शेताच्या कडेने निळे/पिवळे चिकट सापळे लावावेत.'
    },
    'Soybean Healthy Leaf (निरोगी सोयाबीन पान)': {
        'fertilizer': '00:52:34 + Chelated Zinc',
        'dose': '००:५२:३४ ५ ग्रॅम + चिलेटेड झिंक ०.५ ग्रॅम प्रति लिटर पाणी',
        'tips': 'फुलोरा आणि शेंगा भरण्याच्या अवस्थेत दाण्यांचे वजन वाढवण्यासाठी फवारणी करावी.'
    }
}

# 5. Header
st.markdown("""
<div class="hero-banner">
    <h2 style="margin:0; font-size: 22px;">🌱 कृषी-AI : ऑटो पीक व रोग निदान प्रणाली</h2>
    <p style="margin:5px 0 0 0; font-size: 13px; color: #d1f2e2;">आविष्कार संशोधन प्रकल्प • Probabilistic Disease Breakdown</p>
</div>
""", unsafe_allow_html=True)

# 6. Input Section
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("##### 📷 पिकाच्या पानाचा फोटो अपलोड करा (ऑटोमॅटिक डिटेक्शन):")
uploaded_file = st.file_uploader("छायाचित्र निवडा (JPG / PNG)", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# 7. Detection & Detailed Probability Distribution
if uploaded_file is not None and models_ready:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

    with st.spinner("🤖 AI सूक्ष्म परीक्षण करून सर्व शक्यतांची टक्केवारी मोजत आहे..."):
        resized_img = img.resize((224, 224))
        arr = np.array(resized_img, dtype=np.float32)

        # 1. Potato Model
        preds_p = potato_model(np.expand_dims(arr, axis=0), training=False).numpy()[0]
        if np.sum(preds_p) > 1.05 or np.sum(preds_p) < 0.95:
            preds_p = tf.nn.softmax(preds_p).numpy()
        idx_p = int(np.argmax(preds_p))
        conf_p = float(preds_p[idx_p])

        # 2. Soybean Model
        preds_s = soybean_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
        if np.sum(preds_s) > 1.05 or np.sum(preds_s) < 0.95:
            preds_s = tf.nn.softmax(preds_s).numpy()
        idx_s = int(np.argmax(preds_s))
        conf_s = float(preds_s[idx_s])

        # 3. Cotton Model
        preds_c = cotton_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
        if np.sum(preds_c) > 1.05 or np.sum(preds_c) < 0.95:
            preds_c = tf.nn.softmax(preds_c).numpy()
        idx_c = int(np.argmax(preds_c))
        conf_c = float(preds_c[idx_c])

        # Balanced Ensemble Weights
        score_potato = conf_p * 1.35
        score_soybean = conf_s * 1.30
        score_cotton = conf_c * 0.70

        if score_soybean >= score_potato and score_soybean >= score_cotton:
            crop_name = "🌱 सोयाबीन (Soybean Leaf)"
            diagnosed_label = SOYBEAN_CLASSES[idx_s]
            final_conf = conf_s * 100
            current_classes = SOYBEAN_CLASSES
            current_preds = preds_s
        elif score_potato >= score_soybean and score_potato >= score_cotton:
            crop_name = "🥔 बटाटा (Potato Leaf)"
            diagnosed_label = POTATO_CLASSES[idx_p]
            final_conf = conf_p * 100
            current_classes = POTATO_CLASSES
            current_preds = preds_p
        else:
            crop_name = "☁️ कापूस (Cotton Leaf)"
            diagnosed_label = COTTON_CLASSES[idx_c]
            final_conf = conf_c * 100
            current_classes = COTTON_CLASSES
            current_preds = preds_c

        # Main Diagnosis Card
        st.markdown(f"""
        <div class="res-card">
            <span class="badge badge-crop">🌾 ओळखलेले पीक: {crop_name}</span>
            <h3 style="color: #1b5e20; margin: 8px 0 10px 0; font-size: 19px;">📋 मुख्य निदान: {diagnosed_label}</h3>
            <div style="background: #eef7ee; padding: 6px 12px; border-radius: 8px; display: inline-block; margin-bottom: 14px;">
                <span style="font-weight: 700; color: #2e7d32;">एकूण अचूकता (Top Confidence): {final_conf:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

        # Probability Breakdown (Healthy vs Diseases Chances)
        st.markdown("#### 📊 रोगांची व निरोगी असण्याची शक्यता (Probability Breakdown):")
        for cls_name, prob in zip(current_classes, current_preds):
            prob_pct = float(prob) * 100
            # Healthy asel tr green bar, disease asel tr warning orange/red bar
            is_healthy = "Healthy" in cls_name or "Fresh" in cls_name
            color_dot = "🟢" if is_healthy else "🔴"
            
            st.write(f"{color_dot} **{cls_name}**: `{prob_pct:.2f}%`")
            st.progress(min(max(float(prob), 0.0), 1.0))

        # Treatment & Fertilizer Advice
        treatment_data = TREATMENTS.get(diagnosed_label, {
            'fertilizer': 'संतुलित खतांची फवारणी करावी.',
            'dose': 'कृषी तज्ज्ञांच्या सल्ल्याने वापरावे.',
            'tips': 'नियमित निगराणी ठेवावी.'
        })

        st.markdown(f"""
            <div class="treatment-box">
                <h4 style="margin: 0 0 8px 0; color: #1b5e20;">💊 शिफारस केलेले खत / कीटकनाशक:</h4>
                <p style="margin: 0 0 6px 0; font-weight: 600; color: #2e7d32;">🧪 औषध / खत: <span style="color:#000;">{treatment_data['fertilizer']}</span></p>
                <p style="margin: 0 0 6px 0; font-weight: 600; color: #d84315;">⚖️ फवारणी प्रमाण (Dose): <span style="color:#000;">{treatment_data['dose']}</span></p>
                <p style="margin: 0; font-size: 13px; color: #424242;">💡 <b>सल्ला:</b> {treatment_data['tips']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
