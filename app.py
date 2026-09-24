import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="कृषी-AI : Smart Agro Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# DESIGN SYSTEM — CSS
# ============================================================
css_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Mukta:wght@400;500;600;700&display=swap');
:root {
    --forest:#064E3B; --emerald:#059669; --mint:#D1FAE5;
    --cream:#F8FAFC; --cream-2:#F0FDF4; --border:#E2E8F0;
    --ink:#0F172A; --muted:#64748B; --amber:#D97706;
    --amber-bg:#FEF3C7; --red:#DC2626; --red-bg:#FEE2E2; --green-bg:#DCFCE7;
}
html, body, [class*="css"] { font-family:'Plus Jakarta Sans','Mukta',sans-serif; }
.stApp { background:linear-gradient(180deg,var(--cream) 0%,var(--cream-2) 100%); }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.5rem; max-width:1100px; }

.kai-hero {
    background:linear-gradient(135deg,var(--forest) 0%,#0B6B52 100%);
    border-radius:24px; padding:2rem 2.2rem; margin-bottom:1.5rem;
    box-shadow:0 20px 40px -12px rgba(6,78,59,0.35); position:relative; overflow:hidden;
}
.kai-hero-top { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem; }
.kai-brand { display:flex; align-items:center; gap:14px; }
.kai-brand-name { color:#fff; font-size:1.5rem; font-weight:800; margin:0; }
.kai-brand-tag { color:var(--mint); font-family:'Mukta',sans-serif; font-size:0.9rem; margin-top:2px; }
.kai-badge {
    background:rgba(255,255,255,0.14); border:1px solid rgba(255,255,255,0.28);
    color:#fff; font-size:0.75rem; font-weight:600; padding:5px 12px; border-radius:999px;
}
.kai-status { display:inline-flex; align-items:center; gap:8px; margin-top:12px; color:var(--mint); font-size:0.85rem; }
.kai-dot { width:8px; height:8px; border-radius:50%; background:#4ADE80; box-shadow:0 0 0 4px rgba(74,222,128,0.25); }

.kai-section-label { font-size:0.78rem; font-weight:700; color:var(--emerald); text-transform:uppercase; letter-spacing:0.06em; margin:0 0 10px 2px; }
.kai-card { background:#fff; border:1px solid var(--border); border-radius:18px; padding:1.5rem; box-shadow:0 10px 25px -5px rgba(0,0,0,0.05); margin-bottom:1.2rem; }

.kai-pill-row { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:8px; }
.kai-pill { display:inline-flex; align-items:center; padding:6px 14px; border-radius:999px; font-size:0.9rem; font-weight:600; background:var(--mint); color:var(--forest); border:1px solid #A7F3D0; }
.kai-pill.kai-pill-dark { background:var(--forest); color:#fff; border:none; }

.kai-severity { display:inline-flex; align-items:center; padding:7px 16px; border-radius:12px; font-weight:700; font-size:0.9rem; margin-bottom:12px; }
.kai-severity.healthy { background:var(--green-bg); color:#15803D; }
.kai-severity.moderate { background:var(--amber-bg); color:#B45309; }
.kai-severity.critical { background:var(--red-bg); color:#B91C1C; }

.kai-gauge-wrap { display:flex; align-items:center; gap:20px; }
.kai-gauge-num { font-size:2rem; font-weight:800; color:var(--forest); line-height:1; }
.kai-gauge-label { color:var(--muted); font-size:0.85rem; margin-top:4px; }

.kai-prob-row { margin-bottom:12px; }
.kai-prob-top { display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:5px; }
.kai-prob-name { color:var(--ink); font-weight:600; }
.kai-prob-pct { color:var(--emerald); font-weight:700; }
.kai-prob-track { width:100%; height:10px; background:var(--cream-2); border:1px solid var(--border); border-radius:999px; overflow:hidden; }
.kai-prob-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--emerald),#34D399); }

.kai-adv-card { border-radius:16px; padding:1.4rem; height:100%; border:1px solid var(--border); }
.kai-adv-chem { background:#FFF; border-left:4px solid var(--amber); }
.kai-adv-bio { background:#FFF; border-left:4px solid var(--emerald); }
.kai-adv-title { font-weight:700; font-size:1rem; color:var(--ink); margin-bottom:10px; }
.kai-adv-row { margin-bottom:8px; font-size:0.9rem; color:var(--ink); }
.kai-adv-row b { color:var(--forest); }
.kai-adv-tip { margin-top:10px; padding-top:10px; border-top:1px dashed var(--border); font-family:'Mukta',sans-serif; font-size:0.88rem; color:var(--muted); }

.stDownloadButton button {
    background:linear-gradient(135deg,var(--forest),var(--emerald)) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    padding:0.7rem 1.4rem !important; font-weight:700 !important;
}
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# ============================================================
# HERO HEADER
# ============================================================
hero_html = """
<div class="kai-hero">
  <div class="kai-hero-top">
    <div class="kai-brand">
      <div style="font-size:32px;">🌿</div>
      <div>
        <p class="kai-brand-name">कृषी-AI &nbsp;|&nbsp; Krushi-AI</p>
        <p class="kai-brand-tag">पीक रोग निदान प्रणाली — Deep Learning Crop Diagnostics</p>
      </div>
    </div>
    <span class="kai-badge">Avishkar Research Convention 2026</span>
  </div>
  <div class="kai-status"><span class="kai-dot"></span> Deep Learning Engine Active</div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
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
        'bio': 'Beauveria bassiana ५ ग्रॅम प्रति लिटर फवारणी.',
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

def severity_class(severity_text):
    if 'सुरक्षित' in severity_text or 'Healthy' in severity_text:
        return 'healthy'
    if 'मध्यम' in severity_text and 'तीव्र' not in severity_text:
        return 'moderate'
    return 'critical'

# ============================================================
# INPUT & CROP SELECTION
# ============================================================
st.markdown('<p class="kai-section-label">पिकाचा प्रकार व नमुना द्या · Selection & Sample</p>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        crop_mode = st.selectbox(
            "🌾 पीक निवडा (Auto-detect किंवा थेट निवडा):",
            ("🤖 ऑटो-डिटेक्ट (Auto-Detect Mode)", "🥔 बटाटा (Potato)", "🌱 सोयाबीन (Soybean)", "☁️ कापूस (Cotton)")
        )
    with c2:
        input_mode = st.radio("माध्यम निवडा:", ("गॅलरी (Upload)", "कॅमेरा (Camera)"), horizontal=True)

    if input_mode == "गॅलरी (Upload)":
        uploaded_file = st.file_uploader("पानाचा स्पष्ट फोटो निवडा (JPG / PNG):", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        uploaded_file = st.camera_input("कॅमेऱ्यासमोर पान धरून फोटो क्लिक करा:", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# INFERENCE & ACCURATE MULTI-MODEL LOGIC
# ============================================================
if uploaded_file is not None and models_ready:
    img = Image.open(uploaded_file).convert('RGB')

    st.markdown('<div class="kai-card">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    resized_img = img.resize((224, 224))
    arr = np.array(resized_img, dtype=np.float32)

    # १. प्रेडिक्शन्स काढणे
    # बटाटा (0-255 scale)
    preds_p = potato_model(np.expand_dims(arr, axis=0), training=False).numpy()[0]
    if np.sum(preds_p) > 1.05 or np.sum(preds_p) < 0.95:
        preds_p = tf.nn.softmax(preds_p).numpy()
    idx_p = int(np.argmax(preds_p))
    conf_p = float(preds_p[idx_p])

    # सोयाबीन (0-1 scale)
    preds_s = soybean_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(preds_s) > 1.05 or np.sum(preds_s) < 0.95:
        preds_s = tf.nn.softmax(preds_s).numpy()
    idx_s = int(np.argmax(preds_s))
    conf_s = float(preds_s[idx_s])

    # कापूस (0-1 scale)
    preds_c = cotton_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(preds_c) > 1.05 or np.sum(preds_c) < 0.95:
        preds_c = tf.nn.softmax(preds_c).numpy()
    idx_c = int(np.argmax(preds_c))
    conf_c = float(preds_c[idx_c])

    # २. पीक निवड निर्णय (Decision Logic)
    if "बटाटा" in crop_mode:
        selected_crop = "potato"
    elif "सोयाबीन" in crop_mode:
        selected_crop = "soybean"
    elif "कापूस" in crop_mode:
        selected_crop = "cotton"
    else:
        # ऑटो-डिटेक्ट: जर बटाट्याच्या पानावर करपा (Early/Late Blight) असेल तर बटाटाच निवडला जाईल
        is_potato_disease = (idx_p in [0, 1] and conf_p > 0.40)
        is_soybean_pest = (idx_s in [0, 1] and conf_s > 0.60)
        
        if is_potato_disease:
            selected_crop = "potato"
        elif is_soybean_pest:
            selected_crop = "soybean"
        elif conf_s > conf_c and conf_s > conf_p:
            selected_crop = "soybean"
        elif conf_p > conf_c and conf_p > conf_s and idx_p != 2:
            selected_crop = "potato"
        elif conf_c > 0.50:
            selected_crop = "cotton"
        else:
            selected_crop = "soybean"

    # ३. अंतिम लेबल आणि आकडेवारी
    if selected_crop == "potato":
        crop_name = "🥔 बटाटा (Potato Leaf)"
        diagnosed_label = POTATO_CLASSES[idx_p]
        final_conf = conf_p * 100
        current_classes = POTATO_CLASSES
        current_preds = preds_p
    elif selected_crop == "soybean":
        crop_name = "🌱 सोयाबीन (Soybean Leaf)"
        diagnosed_label = SOYBEAN_CLASSES[idx_s]
        final_conf = conf_s * 100
        current_classes = SOYBEAN_CLASSES
        current_preds = preds_s
    else:
        crop_name = "☁️ कापूस (Cotton Leaf)"
        diagnosed_label = COTTON_CLASSES[idx_c]
        final_conf = conf_c * 100
        current_classes = COTTON_CLASSES
        current_preds = preds_c

    info = TREATMENTS[diagnosed_label]
    sev_text = info['severity']
    fertilizer_text = info['fertilizer']
    dose_text = info['dose']
    bio_text = info['bio']
    tips_text = info['tips']
    sev_class = severity_class(sev_text)

    # निकाल हेडर
    st.markdown('<p class="kai-section-label">निदान परिणाम · Diagnosis Result</p>', unsafe_allow_html=True)
    
    res_header_html = f"""<div class="kai-card">
<div class="kai-pill-row">
  <span class="kai-pill kai-pill-dark">{crop_name}</span>
  <span class="kai-pill">{diagnosed_label}</span>
</div>
<div class="kai-severity {sev_class}">● {sev_text}</div>
<br>
<div class="kai-gauge-wrap">
  <div>
    <div class="kai-gauge-num">{final_conf:.1f}%</div>
    <div class="kai-gauge-label">Top Confidence Score</div>
  </div>
  <div style="flex:1;">
    <div class="kai-prob-track" style="height:14px;">
      <div class="kai-prob-fill" style="width:{min(final_conf, 100.0):.1f}%;"></div>
    </div>
  </div>
</div>
</div>"""
    st.markdown(res_header_html, unsafe_allow_html=True)

    # संभाव्यता विवरण
    st.markdown('<p class="kai-section-label">संभाव्यता विश्लेषण · Probability Distribution</p>', unsafe_allow_html=True)
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)

    order = np.argsort(current_preds)[::-1]
    for i in order:
        cls_name = current_classes[i]
        pct = float(current_preds[i]) * 100
        row_html = f"""<div class="kai-prob-row">
<div class="kai-prob-top">
  <span class="kai-prob-name">{cls_name}</span>
  <span class="kai-prob-pct">{pct:.1f}%</span>
</div>
<div class="kai-prob-track">
  <div class="kai-prob-fill" style="width:{min(pct, 100.0):.1f}%;"></div>
</div>
</div>"""
        st.markdown(row_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # रासायनिक व जैविक कार्ड्स
    st.markdown('<p class="kai-section-label">उपचार सल्ला · Treatment Advisory</p>', unsafe_allow_html=True)
    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        chem_html = f"""<div class="kai-adv-card kai-adv-chem">
<div class="kai-adv-title">🧪 रासायनिक उपचार (Chemical Treatment)</div>
<div class="kai-adv-row"><b>औषध:</b> {fertilizer_text}</div>
<div class="kai-adv-row"><b>प्रमाण/डोस:</b> {dose_text}</div>
<div class="kai-adv-tip">{tips_text}</div>
</div>"""
        st.markdown(chem_html, unsafe_allow_html=True)

    with adv_col2:
        bio_html = f"""<div class="kai-adv-card kai-adv-bio">
<div class="kai-adv-title">🌿 सेंद्रिय उपाय (Organic / Bio Alternative)</div>
<div class="kai-adv-row"><b>सेंद्रिय उपाय:</b> {bio_text}</div>
<div class="kai-adv-row"><b>व्यवस्थापन सल्ला:</b> {tips_text}</div>
</div>"""
        st.markdown(bio_html, unsafe_allow_html=True)

    # अहवाल डाउनलोड
    st.markdown("<br>", unsafe_allow_html=True)
    report_text = f"""कृषी-AI (Krushi-AI) — निदान अहवाल / Diagnostic Report
Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}

Crop: {crop_name}
Diagnosis: {diagnosed_label}
Confidence: {final_conf:.2f}%
Severity: {sev_text}

Chemical Treatment (औषध): {fertilizer_text}
Dose (प्रमाण/डोस): {dose_text}

Organic Alternative (सेंद्रिय उपाय): {bio_text}
Management Tip (व्यवस्थापन सल्ला): {tips_text}

--
Avishkar Research Convention 2026 | कृषी-AI Smart Agro Diagnostics
"""
    st.download_button(
        label="📥 निदान अहवाल डाउनलोड करा (Download Report)",
        data=report_text,
        file_name=f"krushi_ai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
    )
    
