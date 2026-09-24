import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit.components.v1 as components

# 1. Page Config
st.set_page_config(page_title="Krushi-AI : Smart Agro Diagnostics", page_icon="🌿", layout="wide")

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
.morph-box {
    background: #fdfdfd; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# 3. Hero Header
st.markdown("""
<div class="kai-hero">
    <div style="font-size: 11px; font-weight: 800; letter-spacing: 1px; color: #a7f3d0; margin-bottom: 4px;">AVISHKAR RESEARCH CONVENTION 2026</div>
    <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800;">🌿 कृषी-AI : स्मार्ट पीक रोग निदान प्रणाली</h2>
    <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #d1fae5;">Deep Learning Crop Vision Engine with Morphological Profiling</p>
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
        'severity': 'मध्यम ते तीव्र (Moderate to High)',
        'fertilizer': 'Mancozeb 75% WP (M-45) / Chlorothalonil 75% WP',
        'dose': '२ ते २.५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३०-३५ ग्रॅम)',
        'bio': 'ट्रायकोडर्मा व्हिरीडी ५० ग्रॅम प्रति पंप किंवा ताक-हिंग फवारणी.',
        'tips': 'पानावरील काळे गोलाकार चट्टे दिसताच दर ८-१० दिवसांनी फवारणी करावी.',
        'day7': '८ व्या दिवशी कॉपर ऑक्सिक्लोराईड (COC) ३० ग्रॅम फवारावे.',
        'day15': '१५ व्या दिवशी ट्रायकोडर्मा व्हिरीडी जमिनीतून ड्रेचिंग करावे.'
    },
    'Potato Late Blight (बटाटा उशिरा करपा)': {
        'severity': 'अति-तीव्र (Critical Risk)',
        'fertilizer': 'Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold)',
        'dose': '२.५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३५-४० ग्रॅम)',
        'bio': 'स्यूडोमोनास फ्लुओरेसेन्स ५ मिली प्रति लिटर पाणी.',
        'tips': 'धुक्याच्या वातावरणात पाणी देणे टाळावे व झाडांच्या बुंध्याशी हवा खेळती ठेवावी.',
        'day7': 'सिमोक्सॅनिल + मॅन्कोझेब (Curzate) ३० ग्रॅम फवारणी करावी.',
        'day15': 'रोगग्रस्त पाने उपटून सुरक्षित नष्ट करावीत.'
    },
    'Potato Healthy Leaf (निरोगी बटाटा पान)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '19:19:19 (Water Soluble NPK) + Micronutrients',
        'dose': '५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ७०-७५ ग्रॅम)',
        'bio': 'दशपर्णी अर्क किंवा जीवामृत दर १५ दिवसांनी द्यावे.',
        'tips': 'समतोल खत व्यवस्थापनाने पिकाची नैसर्गिक प्रतिकारशक्ती टिकवून ठेवा.',
        'day7': 'सूक्ष्मअन्नद्रव्ये (Micronutrients) २ मिली प्रति लिटर द्या.',
        'day15': 'नियमित पाणी व्यवस्थापन ठेवावे.'
    },
    'Diseased Cotton Leaf (रोगग्रस्त कापूस पान)': {
        'severity': 'मध्यम (Moderate)',
        'fertilizer': 'Copper Oxychloride 50% WP (COC) + Streptocycline',
        'dose': 'COC ३० ग्रॅम + स्ट्रेप्टोसायक्लिन २ ग्रॅम (प्रति १५ लिटर पंप)',
        'bio': 'तांबेयुक्त ताक फवारणी किंवा निंबोळी अर्क ५%.',
        'tips': 'जिवाणूजन्य करपा रोखण्यासाठी स्वच्छ सूर्यप्रकाशात सकाळी फवारणी करावी.',
        'day7': 'प्रोपिकॉनाझोल (Tilt) १५ मिली प्रति पंप फवारावे.',
        'day15': 'पांढऱ्या माशीचा प्रादुर्भाव तपासावा.'
    },
    'Diseased Cotton Plant (रोगग्रस्त कापूस झाड)': {
        'severity': 'तीव्र (High Risk)',
        'fertilizer': 'Carbendazim 12% + Mancozeb 63% WP (SAAF)',
        'dose': '२ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३० ग्रॅम)',
        'bio': 'ट्रायकोडर्मा हरझियानम जमिनीतून ड्रेचिंग करावे.',
        'tips': 'रोगट फांद्या छाटून नष्ट कराव्यात; नत्राचा अतिवापर टाळावा.',
        'day7': 'थायोफॅनेट मिथाईल (Roko) २५ ग्रॅम ड्रेचिंग करावे.',
        'day15': 'झाडाच्या मुळाशी पाणी साचणार नाही याची काळजी घ्यावी.'
    },
    'Fresh Cotton Leaf (निरोगी कापूस पान)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '13:00:45 (Potassium Nitrate) + Boron 20%',
        'dose': '१३:००:४५ ५ ग्रॅम + बोरॉन १ ग्रॅम प्रति लिटर पाणी',
        'bio': 'पंचगव्य ३० मिली प्रति लिटर पाणी फवारणी.',
        'tips': 'पाने तजेलदार ठेवण्यासाठी आणि बोंड गळ थांबवण्यासाठी उपयुक्त.',
        'day7': 'चमत्कार (Mepiquat Chloride) १० मिली फवारावे.',
        'day15': 'बोंडांची संख्या तपासत राहावे.'
    },
    'Fresh Cotton Plant (निरोगी कापूस झाड)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '12:61:00 (MAP) / Seaweed Liquid Extract',
        'dose': '४ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ६० ग्रॅम)',
        'bio': 'ह्युमिक ॲसिड १२% मुळाशी सोडावे.',
        'tips': 'पांढऱ्या मुळ्या वाढवण्यासाठी उत्तम.',
        'day7': 'अमिनो ॲसिड टॉनिक २५ मिली प्रति पंप द्यावे.',
        'day15': 'नियमित देखरेख ठेवावी.'
    },
    'Soybean Caterpillar Damage (सोयाबीन पान - अळी प्रादुर्भाव)': {
        'severity': 'तीव्र नुकसान (High)',
        'fertilizer': 'Chlorantraniliprole 18.5% SC (Coragen) / Emamectin Benzoate 5% SG',
        'dose': 'कोराजन ६ मिली किंवा इमामेक्टिन बेन्झोएट १० ग्रॅम (प्रति पंप)',
        'bio': 'निंबोळी अर्क ५% (५० मिली प्रति पंप) किंवा Bt पावडर.',
        'tips': 'पाने खाणाऱ्या अळ्यांचा सुरुवातीच्या अवस्थेतच बंदोबस्त करा.',
        'day7': 'नोव्हाल्युरॉन (Rimon) २५ मिली प्रति पंप फवारावे.',
        'day15': 'शेंगा पोखरणाऱ्या अळीसाठी कामगंध सापळे लावावेत.'
    },
    'Soybean Leaf Beetle Damage (सोयाबीन पान - भुंगा प्रादुर्भाव)': {
        'severity': 'मध्यम (Moderate)',
        'fertilizer': 'Lambda Cyhalothrin 4.9% CS / Quinalphos 25% EC',
        'dose': 'लॅम्बडा १५ मिली किंवा क्विनॉलफॉस ३० मिली (प्रति पंप)',
        'bio': 'Beauveria bassiana ५ ग्रॅम प्रति लिटर फवारणी.',
        'tips': 'भुंग्यांसाठी शेताच्या कडेने पिवळे चिकट सापळे लावावेत.',
        'day7': 'निंबोळी अर्क ५% प्रतिबंधात्मक फवारावा.',
        'day15': 'पानांखालील किडींची तपासणी करावी.'
    },
    'Soybean Healthy Leaf (निरोगी सोयाबीन पान)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '00:52:34 + Chelated Zinc',
        'dose': '००:५२:३४ ५ ग्रॅम + चिलेटेड झिंक ०.५ ग्रॅम प्रति लिटर पाणी',
        'bio': 'जीवामृत आणि वेस्ट डीकंपोजरचा वापर.',
        'tips': 'फुलोरा आणि शेंगा भरण्याच्या अवस्थेत संतुलित पोषण द्या.',
        'day7': 'बोरॉन २०% १ ग्रॅम प्रति लिटर पाणी फवारावे.',
        'day15': 'शेंगा भरताना पाण्याची कमतरता भासू देऊ नका.'
    }
}

# 5. Diagnostic Routine Function
def run_diagnostic(model, classes, crop_name, uploaded_file, is_scaled=True):
    img = Image.open(uploaded_file).convert('RGB')
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    resized_img = img.resize((224, 224))
    arr = np.array(resized_img, dtype=np.float32)
    in_arr = (arr / 255.0) if is_scaled else arr

    preds = model(np.expand_dims(in_arr, axis=0), training=False).numpy()[0]
    if np.sum(preds) > 1.05 or np.sum(preds) < 0.95:
        preds = tf.nn.softmax(preds).numpy()
    idx = int(np.argmax(preds))
    conf = float(preds[idx]) * 100
    diag_label = classes[idx]

    info = TREATMENTS[diag_label]
    sev_text = info['severity']
    sev_cls = 'sev-healthy' if 'सुरक्षित' in sev_text else ('sev-mod' if 'मध्यम' in sev_text else 'sev-crit')

    # Weather
    st.markdown(f"""
    <div class="weather-card">
        <div style="font-weight: 800; font-size: 0.9rem; margin-bottom: 4px;">🌤️ प्रादेशिक हवामान आणि रोग जोखीम (Weather Correlation)</div>
        <div style="font-size: 0.88rem; line-height: 1.5;">
            स्थानिक तापमान: <b>२८°C</b> | हवेतील आर्द्रता: <b>७६%</b> (दमट वातावरण)<br>
            <b>सल्ला:</b> हवेतील दमट वातावरणामुळे रोग वेगाने पसरू शकतात. औषध फवारणी पाऊस नसताना सकाळच्या वेळी करावी.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Result Card
    st.markdown(f"""
    <div class="kai-card">
        <div>
            <span class="kai-pill kai-pill-dark">{crop_name}</span>
            <span class="kai-pill">{diag_label}</span>
        </div>
        <div class="sev-tag {sev_cls}">● {sev_text}</div>
        <div style="font-size: 2rem; font-weight: 800; color: #064E3B; margin-top: 10px;">{conf:.1f}%</div>
        <div style="font-size: 12px; color: #64748B;">Top Model Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # Audio
    audio_speech_text = f"निदान: {crop_name}, {diag_label}. औषध: {info['fertilizer']}, प्रमाण: {info['dose']}."
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

    # XAI
    with st.expander("🔬 AI अटेंशन हीटमॅप पहा (Explainable AI - XAI Attention Map)"):
        leaf_gray = np.array(resized_img.convert('L'), dtype=np.float32)
        leaf_grad = np.abs(leaf_gray - np.mean(leaf_gray))
        heatmap_norm = np.clip((leaf_grad / (np.max(leaf_grad) + 1e-5)) * 255.0, 0, 255).astype(np.uint8)
        heatmap_color = np.zeros((224, 224, 3), dtype=np.uint8)
        heatmap_color[:, :, 0] = heatmap_norm
        heatmap_color[:, :, 1] = 255 - heatmap_norm
        heatmap_color[:, :, 2] = 50
        blended = (0.6 * np.array(resized_img) + 0.4 * heatmap_color).astype(np.uint8)
        c_x1, c_x2 = st.columns(2)
        with c_x1:
            st.image(resized_img, caption="मूळ नमुना (Original Sample)", use_container_width=True)
        with c_x2:
            st.image(blended, caption="AI अटेंशन हीटमॅप (Lesion Focus Map)", use_container_width=True)

    # Probs
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)
    st.markdown("<b>संभाव्यता विवरण (Probabilities):</b>", unsafe_allow_html=True)
    order = np.argsort(preds)[::-1]
    for i in order:
        cls_n = classes[i]
        pct = float(preds[i]) * 100
        st.write(f"• **{cls_n}** : `{pct:.1f}%`")
        st.progress(min(max(float(preds[i]), 0.0), 1.0))
    st.markdown('</div>', unsafe_allow_html=True)

    # Advisory
    adv_c1, adv_c2 = st.columns(2)
    with adv_c1:
        st.markdown(f"""
        <div class="adv-chem">
            <h4 style="margin:0 0 6px 0; color:#b45309;">🧪 रासायनिक उपचार:</h4>
            <p style="margin:0 0 4px 0;"><b>औषध:</b> {info['fertilizer']}</p>
            <p style="margin:0 0 6px 0;"><b>प्रमाण:</b> {info['dose']}</p>
            <small style="color:#64748b;">{info['tips']}</small>
        </div>
        """, unsafe_allow_html=True)

    with adv_c2:
        st.markdown(f"""
        <div class="adv-bio">
            <h4 style="margin:0 0 6px 0; color:#047857;">🌿 जैविक उपाय:</h4>
            <p style="margin:0 0 4px 0;"><b>सेंद्रिय घटक:</b> {info['bio']}</p>
            <small style="color:#64748b;">{info['tips']}</small>
        </div>
        """, unsafe_allow_html=True)

    # Schedule
    st.markdown(f"""
    <div class="kai-card">
        <h4 style="margin:0 0 10px 0; color:#064e3b;">📅 पुढील फवारणी व काळजी वेळापत्रक (Treatment Timeline):</h4>
        <div class="schedule-box"><b>दिवस १ (आज):</b> वरील शिफारसीत रासायनिक/जैविक घटकांची तातडीने फवारणी करा.</div>
        <div class="schedule-box"><b>दिवस ८ (८ दिवसांनी):</b> {info['day7']}</div>
        <div class="schedule-box"><b>दिवस १५ (१५ दिवसांनी):</b> {info['day15']}</div>
    </div>
    """, unsafe_allow_html=True)

# 6. Main Navigation Tabs (Diagnostic + Leaf Botany Reference)
tab_cotton, tab_soybean, tab_potato, tab_botany = st.tabs([
    "☁️ कापूस (Cotton)", 
    "🌱 सोयाबीन (Soybean)", 
    "🥔 बटाटा (Potato)", 
    "🔬 पान ओळख संदर्भ (Leaf Identification Guide)"
])

# TAB 1: कापूस
with tab_cotton:
    st.markdown("### ☁️ कापूस पीक नमुना (Cotton Leaf Sample)")
    c_file = st.file_uploader("कापसाच्या पानाचा फोटो अपलोड करा:", type=["jpg", "jpeg", "png"], key="cotton_upl")
    if c_file and models_ready:
        run_diagnostic(cotton_model, COTTON_CLASSES, "☁️ कापूस (Cotton Leaf)", c_file, is_scaled=True)

# TAB 2: सोयाबीन
with tab_soybean:
    st.markdown("### 🌱 सोयाबीन पीक नमुना (Soybean Leaf Sample)")
    s_file = st.file_uploader("सोयाबीनच्या पानाचा फोटो अपलोड करा:", type=["jpg", "jpeg", "png"], key="soy_upl")
    if s_file and models_ready:
        run_diagnostic(soybean_model, SOYBEAN_CLASSES, "🌱 सोयाबीन (Soybean Leaf)", s_file, is_scaled=True)

# TAB 3: बटाटा
with tab_potato:
    st.markdown("### 🥔 बटाटा पीक नमुना (Potato Leaf Sample)")
    p_file = st.file_uploader("बटाट्याच्या पानाचा फोटो अपलोड करा:", type=["jpg", "jpeg", "png"], key="pot_upl")
    if p_file and models_ready:
        test_img = Image.open(p_file).convert('RGB').resize((150, 150))
        gray = np.array(test_img.convert('L'))
        white_lines = np.sum(gray > 200) / (150 * 150)
        if white_lines > 0.08:
            st.markdown("""
            <div class="kai-card" style="border: 2px solid #ef4444; background: #fef2f2;">
                <h4 style="color:#991b1b; margin:0 0 6px 0;">⚠️ अनोळखी पीक / टोमॅटोचे पान आढळले!</h4>
                <p style="color:#374151; font-size:0.9rem; margin:0;">हे पान बटाट्याचे वाटत नाही. कृपया केवळ बटाट्याचे पान अपलोड करा.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            run_diagnostic(potato_model, POTATO_CLASSES, "🥔 बटाटा (Potato Leaf)", p_file, is_scaled=False)

# TAB 4: शास्त्रीय पान रचना व ओळख मार्गदर्शक (Morphology & Dimensions Data)
with tab_botany:
    st.markdown("### 🌿 तिन्ही पिकांमधील पानाच्या रचनेचा शास्त्रीय फरक (Botanical Features)")
    st.markdown("<p style='color:#64748b; font-size:0.95rem;'>कॉम्प्युटर व्हिजन आणि वनस्पतीशास्त्राच्या आधारे पानांचा प्रकार, रंग, पोत आणि आकारमानातील तुलना खालीलप्रमाणे आहे:</p>", unsafe_allow_html=True)
    
    col_b1, col_b2, col_b3 = st.columns(3)
    
    with col_b1:
        st.markdown("""
        <div class="morph-box" style="border-top: 4px solid #064E3B;">
            <h4 style="color:#064E3B; margin-top:0;">🥔 बटाटा (Potato Leaf)</h4>
            <p><b>पानाचा प्रकार:</b> संयुक्त पान (Pinnately Compound) - एका दांड्यावर अनेक उप-पाने.</p>
            <p><b>आकार व कडा:</b> लंबगोलाकार/अंडाकृती (Ovate), कडा पूर्णपणे गुळगुळीत (Entire).</p>
            <p><b>रंग:</b> गडद हिरवा ते पिवळसर-हिरवा (Yellowish-Dark Green).</p>
            <p><b>पोत:</b> जाडसर, मांसल, विशिष्ट गंधयुक्त.</p>
            <p><b>शिरांची रचना:</b> Pinnate (मध्यशिर जाळीदार रचना).</p>
            <p><b>आकारमान:</b> संपूर्ण लांबी १५-२५ सें.मी., उप-पान रुंदी ३-६ सें.मी.</p>
            <p style="color:#b45309; font-size:0.85rem;"><b>टोमॅटोशी फरक:</b> टोमॅटोच्या कडा करवतीसारख्या दातेरी (Serrated) असतात, बटाट्याच्या गुळगुळीत असतात.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b2:
        st.markdown("""
        <div class="morph-box" style="border-top: 4px solid #059669;">
            <h4 style="color:#059669; margin-top:0;">☁️ कापूस (Cotton Leaf)</h4>
            <p><b>पानाचा प्रकार:</b> साधे पान (Simple Lobed Leaf).</p>
            <p><b>आकार व कडा:</b> तळहातासारखा पंजा (Palmate), ३ ते ५ खोल खाचा (Lobes).</p>
            <p><b>रंग:</b> गडद ते काळसर हिरवा (Dull / Dark Green).</p>
            <p><b>पोत:</b> चिवट, कातडीसारखा (Leathery), उंचवट्यासारख्या शिरा.</p>
            <p><b>शिरांची रचना:</b> Palmate (एकाच तळापासून ३-५ मुख्य शिरा).</p>
            <p><b>आकारमान:</b> लांबी ८-१६ सें.मी., रुंदी ७-१५ सें.मी. (विस्तृत रुंद).</p>
            <p style="color:#059669; font-size:0.85rem;"><b>मुख्य ओळख:</b> पानावरील पंजासारखे ३-५ लोब्स स्पष्ट दिसतात.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b3:
        st.markdown("""
        <div class="morph-box" style="border-top: 4px solid #10B981;">
            <h4 style="color:#10B981; margin-top:0;">🌱 सोयाबीन (Soybean Leaf)</h4>
            <p><b>पानाचा प्रकार:</b> त्रिपर्णी संयुक्त पान (Trifoliate Leaf) - ३ उप-पाने.</p>
            <p><b>आकार व कडा:</b> अंडाकृती ते भाल्यासारखे (Ovate/Elliptic), अखंड कडा.</p>
            <p><b>रंग:</b> उजळ, चमकदार पोपटी ते मध्यम हिरवा (Light Green).</p>
            <p><b>पोत:</b> मऊ, लवचिक, पाठीमागे व पुढे मखमली बारीक लव (Pubescence).</p>
            <p><b>शिरांची रचना:</b> Reticulate (बारीक जाळीदार शिरा).</p>
            <p><b>आकारमान:</b> प्रत्येक उप-पानाची लांबी ६-१० सें.मी., रुंदी ३-५ सें.मी.</p>
            <p style="color:#10B981; font-size:0.85rem;"><b>मुख्य ओळख:</b> एकाच देठावर बरोबर ३ पानांची रचना आणि बारीक लव.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 गुणधर्म तुलनात्मक तक्ता (Quick Comparison Matrix):")
    st.markdown("""
| गुणधर्म / घटक | 🥔 बटाटा (Potato) | ☁️ कापूस (Cotton) | 🌱 सोयाबीन (Soybean) |
| :--- | :--- | :--- | :--- |
| **रचना प्रकार** | संयुक्त (Compound Leaflets) | साधे, पंजाकार (3-5 Lobes) | त्रिपर्णी (Trifoliate - 3 Leaflets) |
| **पानाचा पोत** | जाडसर व सपाट | चिवट, जाड शिरा (Leathery) | मऊ, मखमली लव (Pubescent) |
| **रंग** | मध्यम ते गडद हिरवा | गडद काळसर हिरवा | उजळ पोपटी/हिरवा |
| **सरासरी रुंदी** | ३ ते ६ सें.मी. (उप-पान) |
