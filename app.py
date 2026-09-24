import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Krushi-AI : Smart Agro Diagnostics",
    page_icon="🌿",
    layout="centered"
)

# 2. Modern UI & Print Styling
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
        padding: 20px;
        border: 1px solid #c8e6c9;
        margin-top: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .treatment-box {
        background: #f1f8e9;
        border-left: 5px solid #2e7d32;
        padding: 12px 14px;
        border-radius: 8px;
        margin-top: 12px;
    }
    .bio-box {
        background: #e8f5e9;
        border-left: 5px solid #43a047;
        padding: 12px 14px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 8px;
    }
    .badge-crop { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
    .badge-high { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
    .badge-low { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
</style>
""", unsafe_allow_html=True)

# 3. Model Loading
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

# 4. Comprehensive Treatment & Organic Database
TREATMENTS = {
    'Potato Early Blight (बटाटा करपा)': {
        'severity': 'मध्यम ते तीव्र (Moderate to High)',
        'fertilizer': 'Mancozeb 75% WP (M-45) / Chlorothalonil 75% WP',
        'dose': '२ ते २.५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३०-३५ ग्रॅम)',
        'bio': 'ट्रायकोडर्मा व्हिरीडी (Trichoderma viride) ५० ग्रॅम प्रति पंप किंवा ताक आणि हिंग फवारणी.',
        'tips': 'पानावरील काळे गोलाकार चट्टे आढळल्यास ८-१० दिवसांनी दुसरी फवारणी करावी.'
    },
    'Potato Late Blight (बटाटा उशिरा करपा)': {
        'severity': 'अति-तीव्र (Critical Risk)',
        'fertilizer': 'Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold)',
        'dose': '२.५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३५-४० ग्रॅम)',
        'bio': 'स्यूडोमोनास फ्लुओरेसेन्स (Pseudomonas fluorescens) ५ मिली प्रति लिटर पाणी.',
        'tips': 'झाडाच्या मुळाशी पाणी साचू देऊ नये; धुक्याच्या वातावरणात तातडीने उपाययोजना करा.'
    },
    'Potato Healthy Leaf (निरोगी बटाटा पान)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '19:19:19 (Water Soluble NPK) + Micronutrients',
        'dose': '५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ७०-७५ ग्रॅम)',
        'bio': 'दशपर्णी अर्क किंवा जीवामृत दर १५ दिवसांनी द्यावे.',
        'tips': 'झाडांची वाढ जोमदार राहण्यासाठी संतुलित खते आणि नियमित पाणी व्यवस्थापन ठेवा.'
    },
    'Diseased Cotton Leaf (रोगग्रस्त कापूस पान)': {
        'severity': 'मध्यम (Moderate)',
        'fertilizer': 'Copper Oxychloride 50% WP (COC) + Streptocycline',
        'dose': 'COC ३० ग्रॅम + स्ट्रेप्टोसायक्लिन २ ग्रॅम (प्रति १५ लिटर पंप)',
        'bio': 'तांबेयुक्त ताक फवारणी किंवा निंबोळी अर्क ५%.',
        'tips': 'जिवाणूजन्य करपा रोखण्यासाठी स्वच्छ सूर्यप्रकाशात फवारणी करावी.'
    },
    'Diseased Cotton Plant (रोगग्रस्त कापूस झाड)': {
        'severity': 'तीव्र (High Risk)',
        'fertilizer': 'Carbendazim 12% + Mancozeb 63% WP (SAAF)',
        'dose': '२ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३० ग्रॅम)',
        'bio': 'ट्रायकोडर्मा हरझियानम जमिनीतून ड्रेचिंग करावे.',
        'tips': 'रोगग्रस्त फांद्या छाटून नष्ट कराव्यात; नत्राचा अतिवापर टाळावा.'
    },
    'Fresh Cotton Leaf (निरोगी कापूस पान)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '13:00:45 (Potassium Nitrate) + Boron 20%',
        'dose': '१३:००:४५ ५ ग्रॅम + बोरॉन १ ग्रॅम प्रति लिटर पाणी',
        'bio': 'पंचगव्य ३० मिली प्रति लिटर पाणी फवारणी.',
        'tips': 'पाने तजेलदार ठेवण्यासाठी आणि बोंड गळ थांबवण्यासाठी उपयुक्त.'
    },
    'Fresh Cotton Plant (निरोगी कापूस झाड)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '12:61:00 (MAP) / Seaweed Liquid Extract',
        'dose': '४ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ६० ग्रॅम)',
        'bio': 'ह्युमिक ॲसिड (Humic Acid) १२% मुळाशी सोडावे.',
        'tips': 'झाडांची रोगप्रतिकारक शक्ती आणि पांढऱ्या मुळ्या वाढवण्यासाठी उत्तम.'
    },
    'Soybean Caterpillar Damage (सोयाबीन पान - अळी प्रादुर्भाव)': {
        'severity': 'तीव्र नुकसान (High)',
        'fertilizer': 'Chlorantraniliprole 18.5% SC (Coragen) / Emamectin Benzoate 5% SG',
        'dose': 'कोराजन ६ मिली किंवा इमामेक्टिन बेन्झोएट १० ग्रॅम (प्रति १५ लिटर पंप)',
        'bio': 'निंबोळी अर्क ५% (५० मिली प्रति पंप) किंवा बॅसिलस थुरिनजिएन्सिस (Bt).',
        'tips': 'पाने खाणाऱ्या तंबाखूवरील व हिरव्या अळीचा तात्काळ बंदोबस्त होतो.'
    },
    'Soybean Leaf Beetle Damage (सोयाबीन पान - भुंगा प्रादुर्भाव)': {
        'severity': 'मध्यम (Moderate)',
        'fertilizer': 'Lambda Cyhalothrin 4.9% CS / Quinalphos 25% EC',
        'dose': 'लॅम्बडा सायहेलोथ्रीन १५ मिली किंवा क्विनॉलफॉस ३० मिली (प्रति १५ लिटर पंप)',
        'bio': 'ब Beauveria bassiana ५ ग्रॅम प्रति लिटर फवारणी.',
        'tips': 'भुंग्यांचा प्रादुर्भाव रोखण्यासाठी शेताच्या कडेने पिवळे चिकट सापळे लावावेत.'
    },
    'Soybean Healthy Leaf (निरोगी सोयाबीन पान)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '00:52:34 + Chelated Zinc',
        'dose': '००:५२:३४ ५ ग्रॅम + चिलेटेड झिंक ०.५ ग्रॅम प्रति लिटर पाणी',
        'bio': 'जीवामृत आणि वेस्ट डीकंपोजरचा वापर.',
        'tips': 'फुलोरा आणि शेंगा भरण्याच्या अवस्थेत दाण्यांचे वजन वाढवण्यासाठी फवारणी करावी.'
    }
}

# 5. Header
st.markdown("""
<div class="hero-banner">
    <h2 style="margin:0; font-size: 22px;">🌱 कृषी-AI : स्मार्ट पीक व रोग निदान प्रणाली</h2>
    <p style="margin:5px 0 0 0; font-size: 13px; color: #d1f2e2;">आविष्कार संशोधन प्रकल्प • Automated Multimodal Diagnostics & Advisory</p>
</div>
""", unsafe_allow_html=True)

# 6. Input Section (Gallery + Realtime Camera)
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("##### 📷 पिकाच्या पानाचे छायाचित्र द्या:")
mode = st.radio("इनपुट माध्यम निवडा:", ("गॅलरीतून निवडा (Upload)", "थेट कॅमेरा वापरा (Camera)"), horizontal=True)

uploaded_file = None
if mode == "गॅलरीतून निवडा (Upload)":
    uploaded_file = st.file_uploader("छायाचित्र निवडा (JPG / PNG)", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("कॅमेरा पानासमोर धरून फोटो क्लिक करा")
st.markdown('</div>', unsafe_allow_html=True)

# 7. Diagnostics & Advisory
if uploaded_file is not None and models_ready:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

    with st.spinner("🤖 AI सखोल परीक्षण करत आहे..."):
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

        treatment_data = TREATMENTS.get(diagnosed_label, {
            'severity': 'सामान्य',
            'fertilizer': 'संतुलित खतांची फवारणी करावी.',
            'dose': 'कृषी तज्ज्ञांच्या सल्ल्याने वापरावे.',
            'bio': 'दशपर्णी अर्क किंवा सेंद्रिय खते वापरावीत.',
            'tips': 'नियमित निगराणी ठेवावी.'
        })

        is_healthy = "Healthy" in diagnosed_label or "Fresh" in diagnosed_label
        badge_style = "badge-low" if is_healthy else "badge-high"

        # Main Diagnosis Card
        st.markdown(f"""
        <div class="res-card">
            <span class="badge badge-crop">🌾 ओळखलेले पीक: {crop_name}</span>
            <span class="badge {badge_style}">धोका पातळी: {treatment_data['severity']}</span>
            <h3 style="color: #1b5e20; margin: 8px 0 10px 0; font-size: 19px;">📋 मुख्य निदान: {diagnosed_label}</h3>
            <div style="background: #eef7ee; padding: 6px 12px; border-radius: 8px; display: inline-block; margin-bottom: 14px;">
                <span style="font-weight: 700; color: #2e7d32;">एकूण अचूकता (Top Confidence): {final_conf:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

        # Probability Breakdown
        st.markdown("#### 📊 संभाव्यता विवरण (Chances Breakdown):")
        for cls_name, prob in zip(current_classes, current_preds):
            prob_pct = float(prob) * 100
            icon = "🟢" if ("Healthy" in cls_name or "Fresh" in cls_name) else "🔴"
            st.write(f"{icon} **{cls_name}**: `{prob_pct:.2f}%`")
            st.progress(min(max(float(prob), 0.0), 1.0))

        # Treatment Cards (Chemical + Bio)
        st.markdown(f"""
            <div class="treatment-box">
                <h4 style="margin: 0 0 8px 0; color: #1b5e20;">🧪 रासायनिक उपाय व खते (Chemical Control):</h4>
                <p style="margin: 0 0 6px 0; font-weight: 600; color: #2e7d32;">औषध / खत: <span style="color:#000;">{treatment_data['fertilizer']}</span></p>
                <p style="margin: 0 0 6px 0; font-weight: 600; color: #d84315;">फवारणी प्रमाण (Dose): <span style="color:#000;">{treatment_data['dose']}</span></p>
            </div>
            
            <div class="bio-box">
                <h4 style="margin: 0 0 8px 0; color: #2e7d32;">🌿 जैविक / सेंद्रिय पर्याय (Organic Solution):</h4>
                <p style="margin: 0 0 6px 0; font-weight: 600; color: #2e7d32;">सेंद्रिय घटक: <span style="color:#000;">{treatment_data['bio']}</span></p>
                <p style="margin: 0; font-size: 13px; color: #424242;">💡 <b>व्यवस्थापन सल्ला:</b> {treatment_data['tips']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary Report Download Button
        report_text = f"""--- कृषी-AI पीक आरोग्य अहवाल ---
ओळखलेले पीक: {crop_name}
मुख्य निदान: {diagnosed_label}
अचूकता: {final_conf:.2f}%
धोका पातळी: {treatment_data['severity']}

[रासायनिक उपाय]
औषध/खत: {treatment_data['fertilizer']}
प्रमाण: {treatment_data['dose']}

[जैविक उपाय]
सेंद्रिय घटक: {treatment_data['bio']}
सल्ला: {treatment_data['tips']}
---------------------------------------"""

        st.download_button(
            label="📥 शेतकरी सल्ला अहवाल डाउनलोड करा (Download Report)",
            data=report_text,
            file_name="crop_diagnosis_report.txt",
            mime="text/plain"
        )
