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
    initial_sidebar_state="collapsed",
)

# ============================================================
# DESIGN SYSTEM — CSS
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Mukta:wght@400;500;600;700&display=swap');

    :root{
        --forest:#064E3B;
        --emerald:#059669;
        --mint:#D1FAE5;
        --cream:#F8FAFC;
        --cream-2:#F0FDF4;
        --border:#E2E8F0;
        --ink:#0F172A;
        --muted:#64748B;
        --amber:#D97706;
        --amber-bg:#FEF3C7;
        --red:#DC2626;
        --red-bg:#FEE2E2;
        --green-bg:#DCFCE7;
    }

    html, body, [class*="css"]{
        font-family:'Plus Jakarta Sans','Mukta',sans-serif;
    }

    .stApp{
        background:linear-gradient(180deg,var(--cream) 0%,var(--cream-2) 100%);
    }

    #MainMenu, footer, header{visibility:hidden;}
    .block-container{padding-top:1.5rem; max-width:1100px;}

    /* ---------- HERO ---------- */
    .kai-hero{
        background:linear-gradient(135deg,var(--forest) 0%,#0B6B52 100%);
        border-radius:24px;
        padding:2.2rem 2.4rem;
        margin-bottom:1.6rem;
        box-shadow:0 20px 40px -12px rgba(6,78,59,0.35);
        position:relative;
        overflow:hidden;
    }
    .kai-hero::after{
        content:"";
        position:absolute; top:-40%; right:-10%;
        width:320px; height:320px; border-radius:50%;
        background:radial-gradient(circle,rgba(209,250,229,0.18) 0%,transparent 70%);
    }
    .kai-hero-top{
        display:flex; align-items:center; justify-content:space-between;
        flex-wrap:wrap; gap:1rem; position:relative; z-index:1;
    }
    .kai-brand{display:flex; align-items:center; gap:14px;}
    .kai-brand-name{
        color:#fff; font-size:1.55rem; font-weight:800; letter-spacing:-0.02em; margin:0;
    }
    .kai-brand-tag{
        color:var(--mint); font-family:'Mukta',sans-serif; font-size:0.92rem; margin-top:2px;
    }
    .kai-badge{
        background:rgba(255,255,255,0.14);
        border:1px solid rgba(255,255,255,0.28);
        color:#fff; font-size:0.78rem; font-weight:600;
        padding:6px 14px; border-radius:999px; backdrop-filter:blur(6px);
    }
    .kai-status{
        display:inline-flex; align-items:center; gap:8px;
        margin-top:14px; color:var(--mint); font-size:0.85rem; font-weight:500;
        position:relative; z-index:1;
    }
    .kai-dot{
        width:8px; height:8px; border-radius:50%; background:#4ADE80;
        box-shadow:0 0 0 4px rgba(74,222,128,0.25);
    }

    /* ---------- SECTION LABEL ---------- */
    .kai-section-label{
        font-size:0.78rem; font-weight:700; color:var(--emerald);
        text-transform:uppercase; letter-spacing:0.06em; margin:0 0 10px 2px;
    }

    /* ---------- CARD ---------- */
    .kai-card{
        background:#fff;
        border:1px solid var(--border);
        border-radius:18px;
        padding:1.6rem 1.7rem;
        box-shadow:0 10px 25px -5px rgba(0,0,0,0.05);
        margin-bottom:1.2rem;
    }

    /* ---------- SEGMENTED CONTROL (radio) ---------- */
    div[role="radiogroup"]{
        background:var(--cream-2);
        border:1px solid var(--border);
        border-radius:14px;
        padding:5px;
        display:inline-flex;
        gap:4px;
    }
    div[role="radiogroup"] label{
        border-radius:10px !important;
        padding:6px 18px !important;
        transition:all .15s ease;
    }

    /* ---------- FILE UPLOADER ---------- */
    [data-testid="stFileUploaderDropzone"]{
        background:var(--cream-2) !important;
        border:2px dashed #A7F3D0 !important;
        border-radius:16px !important;
    }

    /* ---------- PILL TAGS ---------- */
    .kai-pill-row{display:flex; flex-wrap:wrap; gap:10px; margin-bottom:6px;}
    .kai-pill{
        display:inline-flex; align-items:center; gap:6px;
        padding:8px 16px; border-radius:999px;
        font-size:0.92rem; font-weight:600;
        background:var(--mint); color:var(--forest);
        border:1px solid #A7F3D0;
    }
    .kai-pill.kai-pill-dark{
        background:var(--forest); color:#fff; border:none;
    }

    /* ---------- SEVERITY BADGE ---------- */
    .kai-severity{
        display:inline-flex; align-items:center; gap:8px;
        padding:9px 18px; border-radius:12px; font-weight:700; font-size:0.95rem;
    }
    .kai-severity.healthy{ background:var(--green-bg); color:#15803D; }
    .kai-severity.moderate{ background:var(--amber-bg); color:#B45309; }
    .kai-severity.critical{ background:var(--red-bg); color:#B91C1C; }

    /* ---------- CONFIDENCE GAUGE ---------- */
    .kai-gauge-wrap{ display:flex; align-items:center; gap:22px; }
    .kai-gauge-num{ font-size:2.1rem; font-weight:800; color:var(--forest); line-height:1; }
    .kai-gauge-label{ color:var(--muted); font-size:0.85rem; margin-top:4px; }

    /* ---------- PROBABILITY BARS ---------- */
    .kai-prob-row{ margin-bottom:14px; }
    .kai-prob-top{ display:flex; justify-content:space-between; font-size:0.87rem; margin-bottom:6px; }
    .kai-prob-name{ color:var(--ink); font-weight:600; }
    .kai-prob-pct{ color:var(--emerald); font-weight:700; }
    .kai-prob-track{
        width:100%; height:10px; background:var(--cream-2);
        border:1px solid var(--border); border-radius:999px; overflow:hidden;
    }
    .kai-prob-fill{
        height:100%; border-radius:999px;
        background:linear-gradient(90deg,var(--emerald),#34D399);
    }

    /* ---------- ADVISORY CARDS ---------- */
    .kai-adv-card{
        border-radius:16px; padding:1.4rem 1.5rem; height:100%;
        border:1px solid var(--border);
    }
    .kai-adv-chem{ background:linear-gradient(160deg,#FFF 0%,#FEF9F3 100%); border-left:4px solid var(--amber); }
    .kai-adv-bio{ background:linear-gradient(160deg,#FFF 0%,var(--cream-2) 100%); border-left:4px solid var(--emerald); }
    .kai-adv-title{ font-weight:700; font-size:1.02rem; color:var(--ink); margin-bottom:10px; }
    .kai-adv-row{ margin-bottom:8px; font-size:0.9rem; color:var(--ink); }
    .kai-adv-row b{ color:var(--forest); }
    .kai-adv-tip{
        margin-top:10px; padding-top:10px; border-top:1px dashed var(--border);
        font-family:'Mukta',sans-serif; font-size:0.88rem; color:var(--muted);
    }

    /* ---------- DOWNLOAD BUTTON ---------- */
    .stDownloadButton button{
        background:linear-gradient(135deg,var(--forest),var(--emerald)) !important;
        color:#fff !important; border:none !important;
        border-radius:12px !important; padding:0.7rem 1.4rem !important;
        font-weight:700 !important; box-shadow:0 10px 20px -6px rgba(5,150,105,0.45) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="kai-hero">
      <div class="kai-hero-top">
        <div class="kai-brand">
          <svg width="46" height="46" viewBox="0 0 46 46" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="23" cy="23" r="23" fill="#ffffff" fill-opacity="0.12"/>
            <path d="M23 34C23 34 10 29 10 18.5C10 13.8056 13.8056 10 18.5 10C20.6 10 22.4 10.9 23.7 12.3C24.0 12.6 24.3 12.6 24.6 12.3C25.9 10.9 27.7 10 29.8 10C34.5 10 38.3 13.8056 38.3 18.5C38.3 22.5 35.5 26 32 28.5" stroke="#D1FAE5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M23 34V20" stroke="#D1FAE5" stroke-width="2" stroke-linecap="round"/>
          </svg>
          <div>
            <p class="kai-brand-name">कृषी-AI &nbsp;|&nbsp; Krushi-AI</p>
            <p class="kai-brand-tag">पीक रोग निदान प्रणाली — Deep Learning Crop Diagnostics</p>
          </div>
        </div>
        <span class="kai-badge">Avishkar Research Convention 2026</span>
      </div>
      <div class="kai-status"><span class="kai-dot"></span> Deep Learning Engine Active</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_all_models():
    p_m = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False)
    c_m = tf.keras.models.load_model('cotton_model.h5', compile=False)
    s_m = tf.keras.models.load_model('soybean_model.h5', compile=False)
    return p_m, c_m, s_m

potato_model, cotton_model, soybean_model = load_all_models()

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

def severity_class(severity_text: str) -> str:
    if 'सुरक्षित' in severity_text or 'Healthy' in severity_text:
        return 'healthy'
    if 'मध्यम' in severity_text and 'तीव्र' not in severity_text:
        return 'moderate'
    return 'critical'

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown('<p class="kai-section-label">पान अपलोड करा · Upload Leaf Sample</p>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)
    mode = st.radio("Input Mode:", ("Upload", "Camera"), horizontal=True, label_visibility="collapsed")
    uploaded_file = st.file_uploader(
        "Upload Leaf", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    ) if mode == "Upload" else st.camera_input("Take Photo", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# INFERENCE + RESULTS
# ============================================================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    col_img, col_gap = st.columns([1, 0.001])
    with col_img:
        st.markdown('<div class="kai-card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    resized_img = img.resize((224, 224))
    arr = np.array(resized_img, dtype=np.float32)

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

    info = TREATMENTS[diagnosed_label]
    sev_text = info['severity']
    fertilizer_text = info['fertilizer']
    dose_text = info['dose']
    bio_text = info['bio']
    tips_text = info['tips']
    sev_class = severity_class(sev_text)

    # ---------------- RESULTS HEADER ----------------
    st.markdown('<p class="kai-section-label">निदान परिणाम · Diagnosis Result</p>', unsafe_allow_html=True)
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="kai-pill-row">
          <span class="kai-pill kai-pill-dark">{crop_name}</span>
          <span class="kai-pill">{diagnosed_label}</span>
        </div>
        <div class="kai-severity {sev_class}">● {sev_text}</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence meter
    st.markdown(
        f"""
        <div class="kai-gauge-wrap">
          <div>
            <div class="kai-gauge-num">{final_conf:.1f}%</div>
            <div class="kai-gauge-label">Top Confidence Score</div>
          </div>
          <div style="flex:1;">
            <div class="kai-prob-track" style="height:14px;">
              <div class="kai-prob-fill" style="width:{final_conf:.1f}%;"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PROBABILITY DISTRIBUTION ----------------
    st.markdown('<p class="kai-section-label">संभाव्यता विश्लेषण · Probability Distribution</p>', unsafe_allow_html=True)
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)

    order = np.argsort(current_preds)[::-1]
    bars_html = ""
    for i in order:
        cls_name = current_classes[i]
        pct = float(current_preds[i]) * 100
        bars_html += f"""
        <div class="kai-prob-row">
          <div class="kai-prob-top">
            <span class="kai-prob-name">{cls_name}</span>
            <span class="kai-prob-pct">{pct:.1f}%</span>
          </div>
          <div class="kai-prob-track">
            <div class="kai-prob-fill" style="width:{pct:.1f}%;"></div>
          </div>
        </div>
        """
    st.markdown(bars_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- ADVISORY CARDS ----------------
    st.markdown('<p class="kai-section-label">उपचार सल्ला · Treatment Advisory</p>', unsafe_allow_html=True)
    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        st.markdown(
            f"""
            <div class="kai-adv-card kai-adv-chem">
              <div class="kai-adv-title">🧪 रासायनिक उपचार (Chemical Treatment)</div>
              <div class="kai-adv-row"><b>औषध:</b> {fertilizer_text}</div>
              <div class="kai-adv-row"><b>प्रमाण/डोस:</b> {dose_text}</div>
              <div class="kai-adv-tip">{tips_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with adv_col2:
        st.markdown(
            f"""
            <div class="kai-adv-card kai-adv-bio">
         
