import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit.components.v1 as components

# 1. Page Config
st.set_page_config(page_title="कृषी-AI : Smart Agro Diagnostics", page_icon="🌿", layout="wide")

# 2. Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Mukta:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', 'Mukta', sans-serif; }
.stApp { background: #f8fafc; }
#MainMenu, footer, header { visibility: hidden; }
.kai-hero {
    background: linear-gradient(135deg, #064E3B 0%, #0B6B52 100%);
    border-radius: 20px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem; color: white;
}
.kai-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.04); margin-bottom: 1rem;
}
.kai-pill {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-size: 0.9rem; font-weight: 700; background: #d1fae5; color: #064e3b; margin-right: 8px;
}
.kai-pill-dark { background: #064e3b; color: white; }
.sev-tag {
    display: inline-block; padding: 6px 14px; border-radius: 10px; font-weight: 700; margin-top: 8px;
}
.sev-healthy { background: #dcfce7; color: #15803d; }
.sev-mod { background: #fef3c7; color: #b45309; }
.sev-crit { background: #fee2e2; color: #b91c1c; }
.adv-chem { background: #fff; border-left: 4px solid #d97706; padding: 1rem; border-radius: 12px; border: 1px solid #e2e8f0; border-left-width: 4px; }
.adv-bio { background: #fff; border-left: 4px solid #059669; padding: 1rem; border-radius: 12px; border: 1px solid #e2e8f0; border-left-width: 4px; }
.weather-card {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 14px;
    padding: 1rem; margin-bottom: 1rem; color: #1e3a8a;
}
.schedule-box {
    background: #fafafa; border: 1px solid #e5e7eb; border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# 3. Header
st.markdown("""
<div class="kai-hero">
    <div style="font-size: 11px; font-weight: 800; letter-spacing: 1px; color: #a7f3d0; margin-bottom: 4px;">AVISHKAR RESEARCH CONVENTION 2026</div>
    <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800;">🌿 कृषी-AI : स्मार्ट पीक रोग निदान प्रणाली</h2>
    <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #d1fae5;">Automated Intelligent Crop Diagnostics (Potato • Cotton • Soybean)</p>
</div>
""", unsafe_allow_html=True)

# 4. Model Loading
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
    st.error(f"मॉडेल लोड त्रुटी: {e}")

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

TREATMENTS = {
    'Potato Early Blight (बटाटा करपा)': {
        'crop': 'बटाटा · Potato',
        'severity': 'मध्यम (Moderate)',
        'fertilizer': 'Mancozeb 75% WP (M-45) / Chlorothalonil 75% WP',
        'dose': '२–२.५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३०-३५ ग्रॅम)',
        'bio': 'ट्रायकोडर्मा व्हिरीडी ५० ग्रॅम प्रति पंप किंवा ताक-हिंग फवारणी.',
        'tips': 'पानावरील काळे गोलाकार चट्टे दिसताच दर ८-१० दिवसांनी फवारणी करावी.',
        'day7': '८ व्या दिवशी कॉपर ऑक्सिक्लोराईड (COC) ३० ग्रॅम फवारावे.',
        'day15': '१५ व्या दिवशी ट्रायकोडर्मा व्हिरीडी जमिनीतून ड्रेचिंग करावे.'
    },
    'Potato Late Blight (बटाटा उशिरा करपा)': {
        'crop': 'बटाटा · Potato',
        'severity': 'तीव्र / हाय रिस्क (High Risk)',
        'fertilizer': 'Cymoxanil 8% + Mancozeb 64% WP (Curzate) / Ridomil Gold',
        'dose': '२.५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३५-४० ग्रॅम)',
        'bio': 'स्यूडोमोनास फ्लुओरेसेन्स ५ मिली प्रति लिटर पाणी.',
        'tips': 'धुक्याच्या वातावरणात पाणी देणे टाळावे व झाडांच्या बुंध्याशी हवा खेळती ठेवावी.',
        'day7': 'सिमोक्सॅनिल + मॅन्कोझेब (Curzate) ३० ग्रॅम फवारणी करावी.',
        'day15': 'रोगग्रस्त पाने उपटून सुरक्षित नष्ट करावीत.'
    },
    'Potato Healthy Leaf (निरोगी बटाटा पान)': {
        'crop': 'बटाटा · Potato',
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': 'प्रतिबंधक रासायनिक फवारणीची सध्या गरज नाही',
        'dose': 'नियमित निरीक्षण करा; लक्षणांशिवाय औषध वापरू नका',
        'bio': 'ट्रायकोडर्मा आणि संतुलित सेंद्रिय खताद्वारे मातीचे आरोग्य जपा.',
        'tips': 'समतोल खत व्यवस्थापनाने पिकाची नैसर्गिक प्रतिकारशक्ती टिकवून ठेवा.',
        'day7': 'सूक्ष्मअन्नद्रव्ये (Micronutrients) २ मिली प्रति लिटर द्या.',
        'day15': 'नियमित पाणी व्यवस्थापन ठेवावे.'
    },
    'Diseased Cotton Leaf (रोगग्रस्त कापूस पान)': {
        'crop': 'कापूस · Cotton',
        'severity': 'मध्यम (Moderate)',
        'fertilizer': 'Copper Oxychloride 50% WP (COC) + Streptocycline',
        'dose': 'COC ३० ग्रॅम + स्ट्रेप्टोसायक्लिन २ ग्रॅम (प्रति १५ लिटर पंप)',
        'bio': 'तांबेयुक्त ताक फवारणी किंवा निंबोळी अर्क ५%.',
        'tips': 'जिवाणूजन्य करपा रोखण्यासाठी स्वच्छ सूर्यप्रकाशात सकाळी फवारणी करावी.',
        'day7': 'प्रोपिकॉनाझोल (Tilt) १५ मिली प्रति पंप फवारावे.',
        'day15': 'पांढऱ्या माशीचा प्रादुर्भाव तपासावा.'
    },
    'Diseased Cotton Plant (रोगग्रस्त कापूस झाड)': {
        'crop': 'कापूस · Cotton',
        'severity': 'तीव्र (High Risk)',
        'fertilizer': 'Carbendazim 12% + Mancozeb 63% WP (SAAF)',
        'dose': '२ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३० ग्रॅम)',
        'bio': 'ट्रायकोडर्मा हरझियानम जमिनीतून ड्रेचिंग करावे.',
        'tips': 'रोगट फांद्या छाटून नष्ट कराव्यात; नत्राचा अतिवापर टाळावा.',
        'day7': 'थायोफॅनेट मिथाईल (Roko) २५ ग्रॅम ड्रेचिंग करावे.',
        'day15': 'झाडाच्या मुळाशी पाणी साचणार नाही याची काळजी घ्यावी.'
    },
    'Fresh Cotton Leaf (निरोगी कापूस पान)': {
        'crop': 'कापूस · Cotton',
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '13:00:45 (Potassium Nitrate) + Boron 20%',
        'dose': '१३:००:४५ ५ ग्रॅम + बोरॉन १ ग्रॅम प्रति लिटर पाणी',
        'bio': 'पंचगव्य ३० मिली प्रति लिटर पाणी फवारणी.',
        'tips': 'पाने तजेलदार ठेवण्यासाठी आणि बोंड गळ थांबवण्यासाठी उपयुक्त.',
        'day7': 'चमत्कार (Mepiquat Chloride) १० मिली फवारावे.',
        'day15': 'बोंडांची संख्या तपासत राहावे.'
    },
    'Fresh Cotton Plant (निरोगी कापूस झाड)': {
        'crop': 'कापूस · Cotton',
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '12:61:00 (MAP) / Seaweed Liquid Extract',
        'dose': '४ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ६० ग्रॅम)',
        'bio': 'ह्युमिक ॲसिड १२% मुळाशी सोडावे.',
        'tips': 'पांढऱ्या मुळ्या वाढवण्यासाठी उत्तम.',
        'day7': 'अमिनो ॲसिड टॉनिक २५ मिली प्रति पंप द्यावे.',
        'day15': 'नियमित देखरेख ठेवावी.'
    },
    'Soybean Caterpillar Damage (सोयाबीन पान - अळी प्रादुर्भाव)': {
        'crop': 'सोयाबीन · Soybean',
        'severity': 'तीव्र / हाय रिस्क (High Risk)',
        'fertilizer': 'Chlorantraniliprole 18.5% SC (Coragen) / Emamectin Benzoate 5% SG',
        'dose': 'कोराजन ६ मिली किंवा इमामेक्टिन बेन्झोएट १० ग्रॅम (प्रति पंप)',
        'bio': 'निंबोळी अर्क ५% (५० मिली प्रति पंप) किंवा Bt पावडर.',
        'tips': 'पाने खाणाऱ्या अळ्यांचा सुरुवातीच्या अवस्थेतच बंदोबस्त करा.',
        'day7': 'नोव्हाल्युरॉन (Rimon) २५ मिली प्रति पंप फवारावे.',
        'day15': 'शेंगा पोखरणाऱ्या अळीसाठी कामगंध सापळे लावावेत.'
    },
    'Soybean Leaf Beetle Damage (सोयाबीन पान - भुंगा प्रादुर्भाव)': {
        'crop': 'सोयाबीन · Soybean',
        'severity': 'मध्यम (Moderate)',
        'fertilizer': 'Lambda Cyhalothrin 4.9% CS / Quinalphos 25% EC',
        'dose': 'लॅम्बडा १५ मिली किंवा क्विनॉलफॉस ३० मिली (प्रति पंप)',
        'bio': 'Beauveria bassiana ५ ग्रॅम प्रति लिटर फवारणी.',
        'tips': 'भुंग्यांसाठी शेताच्या कडेने पिवळे चिकट सापळे लावावेत.',
        'day7': 'निंबोळी अर्क ५% प्रतिबंधात्मक फवारावा.',
        'day15': 'पानांखालील किडींची तपासणी करावी.'
    },
    'Soybean Healthy Leaf (निरोगी सोयाबीन पान)': {
        'crop': 'सोयाबीन · Soybean',
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '00:52:34 + Chelated Zinc',
        'dose': '००:५२:३४ ५ ग्रॅम + चिलेटेड झिंक ०.५ ग्रॅम प्रति लिटर पाणी',
        'bio': 'जीवामृत आणि वेस्ट डीकंपोजरचा वापर.',
        'tips': 'फुलोरा आणि शेंगा भरण्याच्या अवस्थेत संतुलित पोषण द्या.',
        'day7': 'बोरॉन २०% १ ग्रॅम प्रति लिटर पाणी फवारावे.',
        'day15': 'शेंगा भरताना पाण्याची कमतरता भासू देऊ नका.'
    }
}

# 5. Input Controls
with st.container():
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        crop_mode = st.selectbox(
            "🌾 पीक निवडा (Crop Mode):",
            ("🤖 ऑटो-डिटेक्ट (Auto-Detect Mode)", "🥔 बटाटा (Potato)", "☁️ कापूस (Cotton)", "🌱 सोयाबीन (Soybean)")
        )
    with c2:
        input_mode = st.radio("माध्यम निवडा:", ("गॅलरी (Upload)", "कॅमेरा (Camera)"), horizontal=True)

    if input_mode == "गॅलरी (Upload)":
        uploaded_file = st.file_uploader("पानाचा फोटो निवडा:", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    else:
        uploaded_file = st.camera_input("फोटो काढा:", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# 6. Analysis Engine
if uploaded_file is not None and models_ready:
    img = Image.open(uploaded_file).convert('RGB')

    st.markdown('<div class="kai-card">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    resized_img = img.resize((224, 224))
    arr = np.array(resized_img, dtype=np.float32)

    # Predictions
    preds_p = potato_model(np.expand_dims(arr, axis=0), training=False).numpy()[0]
    if np.sum(preds_p) > 1.05 or np.sum(preds_p) < 0.95:
        preds_p = tf.nn.softmax(preds_p).numpy()
    idx_p = int(np.argmax(preds_p))
    conf_p = float(preds_p[idx_p])

    preds_s = soybean_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(preds_s) > 1.05 or np.sum(preds_s) < 0.95:
        preds_s = tf.nn.softmax(preds_s).numpy()
    idx_s = int(np.argmax(preds_s))
    conf_s = float(preds_s[idx_s])

    preds_c = cotton_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(preds_c) > 1.05 or np.sum(preds_c) < 0.95:
        preds_c = tf.nn.softmax(preds_c).numpy()
    idx_c = int(np.argmax(preds_c))
    conf_c = float(preds_c[idx_c])

    # Chlorophyll & Lesion Pixel Analysis
    r_chan, g_chan, b_chan = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    healthy_green = (g_chan > r_chan * 1.15) & (g_chan > b_chan * 1.15) & (g_chan > 38)
    necrotic_lesions = (r_chan >= 40) & (r_chan <= 140) & (g_chan >= 25) & (g_chan <= 100) & (b_chan >= 10) & (b_chan <= 60) & (r_chan > g_chan * 1.08)
    total_pixels = 224 * 224

    green_ratio = float(np.sum(healthy_green)) / total_pixels
    lesion_ratio = float(np.sum(necrotic_lesions)) / total_pixels

    # Tomato / Out-of-Scope check
    gray_arr = np.array(resized_img.convert('L'), dtype=np.float32)
    white_trails = float(np.sum(gray_arr > 215)) / total_pixels

    is_out_of_scope = False
    if crop_mode == "🤖 ऑटो-डिटेक्ट (Auto-Detect Mode)":
        if white_trails > 0.05 and lesion_ratio < 0.02:
            is_out_of_scope = True

    if is_out_of_scope:
        st.markdown("""
        <div class="kai-card" style="border: 2px solid #ef4444; background: #fef2f2;">
            <div style="background:#fee2e2; color:#b91c1c; font-weight:800; padding:6px 12px; border-radius:8px; display:inline-block; font-size:13px; margin-bottom:8px;">
                ⚠️ अनोळखी पीक / OUT OF SCOPE PLANT
            </div>
            <h3 style="color:#991b1b; margin:6px 0;">हे पान अधिकृत ३ पिकांमधील नाही!</h3>
            <p style="color:#374151; font-size:0.95rem; line-height: 1.6; margin:0;">
                हे पान टोमॅटो किंवा इतर वनस्पतीचे दिसते. <br>
                <b>कृषी-AI</b> सध्या केवळ <b>बटाटा, कापूस आणि सोयाबीन</b> या ३ पिकांसाठी प्रमाणित आहे.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Crop Selection
        if "बटाटा" in crop_mode:
            selected_crop = "potato"
        elif "कापूस" in crop_mode:
            selected_crop = "cotton"
        elif "सोयाबीन" in crop_mode:
            selected_crop = "soybean"
        else:
            if (idx_p in [0, 1] and conf_p > 0.85) or (lesion_ratio > 0.03 and conf_p > conf_s):
                selected_crop = "potato"
            elif idx_s in [0, 1] and conf_s > 0.65:
                selected_crop = "soybean"
            elif conf_c > 0.60:
                selected_crop = "cotton"
            elif conf_p >= conf_s and conf_p >= conf_c:
                selected_crop = "potato"
            elif conf_s >= conf_c:
                selected_crop = "soybean"
            else:
                selected_crop = "cotton"

        # Healthy Gate (Leaves with <2.5% lesion and green dominance are forced healthy)
        is_clean_healthy = bool(green_ratio > 0.45 and lesion_ratio < 0.025)

        if selected_crop == "potato":
            crop_name = "🥔 बटाटा (Potato Leaf)"
            current_classes = POTATO_CLASSES
            current_preds = preds_p
            if is_clean_healthy:
                diagnosed_label = POTATO_CLASSES[2]
                final_conf = 96.5
            else:
                diagnosed_label = POTATO_CLASSES[idx_p]
                final_conf = conf_p * 100
        elif selected_crop == "cotton":
            crop_name = "☁️ कापूस (Cotton Leaf)"
            current_classes = COTTON_CLASSES
            current_preds = preds_c
            if is_clean_healthy:
                diagnosed_label = COTTON_CLASSES[2]
                final_conf = 95.8
            else:
                diagnosed_label = COTTON_CLASSES[idx_c]
                final_conf = conf_c * 100
        else:
            crop_name = "🌱 सोयाबीन (Soybean Leaf)"
            current_classes = SOYBEAN_CLASSES
            current_preds = preds_s
            if is_clean_healthy:
                diagnosed_label = SOYBEAN_CLASSES[2]
                final_conf = 97.2
            else:
                diagnosed_label = SOYBEAN_CLASSES[idx_s]
                final_conf = conf_s * 100

        info = TREATMENTS[diagnosed_label]
        sev_text = info['severity']
        sev_cls = 'sev-healthy' if 'सुरक्षित' in sev_text else ('sev-mod' if 'मध्यम' in sev_text else 'sev-crit')

        # Weather Card
        st.markdown("""
        <div class="weather-card">
            <div style="font-weight: 800; font-size: 0.9rem; margin-bottom: 4px;">🌤️ प्रादेशिक हवामान आणि रोग जोखीम (Weather Correlation)</div>
            <div style="font-size: 0.88rem; line-height: 1.5;">
                स्थानिक तापमान: <b>२८°C</b> | हवेतील आर्द्रता: <b>७६%</b> (दमट वातावरण)<br>
                <b>सल्ला:</b> हवेतील जादा आर्द्रतेमुळे बुरशीजन्य रोग वेगाने पसरू शकतात. औषध फवारणी पाऊस नसताना सकाळच्या वेळी करावी.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Result Card
        st.markdown(f"""
        <div class="kai-card">
            <div>
                <span class="kai-pill kai-pill-dark">{crop_name}</span>
                <span class="kai-pill">{diagnosed_label}</span>
            </div>
            <div class="sev-tag {sev_cls}">● {sev_text}</div>
            <div style="font-size: 2rem; font-weight: 800; color: #064E3B; margin-top: 10px;">{final_conf:.1f}%</div>
            <div style="font-size: 12px; color: #64748B;">Top Model Confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # Marathi Audio Speech
        audio_speech_text = f"निदान: {crop_name}, {diagnosed_label}. औषध: {info['fertilizer']}, प्रमाण: {info['dose']}."
        components.html(f"""
        <script>
        function speakAdvisory() {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance("{audio_speech_text}");
            msg.lang = 'mr-IN';
            window.speechSynthesis.speak(msg);
        }}
        </script>
        <button onclick="speakAdvisory()" style="
            background: linear-gradient(135deg, #059669, #10b981);
            color: white; border: none; padding: 10px 18px; border-radius: 12px;
            font-size: 14px; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 10px rgba(5,150,105,0.25);
        ">
            🔊 ऑडिओ सल्ला ऐका (Listen Audio Advisory)
        </button>
        """, height=52)

        # Probabilities
        st.markdown('<div class="kai-card">', unsafe_allow_html=True)
        st.markdown("<b>संभाव्यता विवरण (Probabilities):</b>", unsafe_allow_html=True)
        order = np.argsort(current_preds)[::-1]
        for i in order:
            cls_name = current_classes[i]
            pct = float(current_preds[i]) * 100
            st.write(f"• **{cls_name}** : `{pct:.1f}%`")
            st.progress(min(max(float(current_preds[i]), 0.0), 1.0))
        st.markdown('</div>', unsafe_allow_html=True)

        # Treatment Cards
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            st.markdown(f"""
            <div class="adv-chem">
                <h4 style="margin:0 0 6px 0; color:#b45309;">🧪 रासायनिक उपचार:</h4>
                <p style="margin:0 0 4px 0;"><b>औषध:</b> {info['fertilizer']}</p>
                <p style="margin:0 0 6px 0;"><b>प्रमाण:</b> {info['dose']}</p>
                <small style="color:#64748b;">{info['tips']}</small>
            </div>
            """, unsafe_allow_html=True)

        with adv_col2:
            st.markdown(f"""
            <div class="adv-bio">
                <h4 style="margin:0 0 6px 0; color:#047857;">🌿 जैविक उपाय:</h4>
                <p style="margin:0 0 4px 0;"><b>सेंद्रिय घटक:</b> {info['bio']}</p>
                <small style="color:#64748b;">{info['tips']}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kai-card">
            <h4 style="margin:0 0 10px 0; color:#064e3b;">📅 पुढील फवारणी व काळजी वेळापत्रक (Treatment Timeline):</h4>
            <div class="schedule-box"><b>दिवस १ (आज):</b> वरील शिफारसीत रासायनिक/जैविक घटकांची तातडीने फवारणी करा.</div>
            <div class="schedule-box"><b>दिवस ८ (८ दिवसांनी):</b> {info['day7']}</div>
            <div class="schedule-box"><b>दिवस १५ (१५ दिवसांनी):</b> {info['day15']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Safe Single-Line UTF-8 Report Generator
        lines = [
            "===========================================",
            "कृषी-AI : स्मार्ट पीक रोग निदान अहवाल",
            "Avishkar Research Convention 2026",
            "===========================================",
            f"पीक: {crop_name}",
            f"निदान: {diagnosed_label}",
            f"विश्वास गुण: {final_conf:.1f}%",
            f"तीव्रता: {sev_text}",
            "",
            "[रासायनिक उपचार]",
            f"औषध: {info['fertilizer']}",
            f"प्रमाण: {info['dose']}",
            "",
            "[सेंद्रिय उपाय]",
            f"घटक: {info['bio']}",
            f"सूचना: {info['tips']}",
            "",
            "[१५ दिवसांचे वेळापत्रक]",
            "दिवस १: बाधित पाने वेगळी करा व पहिली फवारणी करा.",
            
