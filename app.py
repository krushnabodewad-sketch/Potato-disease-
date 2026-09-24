import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

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
</style>
""", unsafe_allow_html=True)

# 3. Header
st.markdown("""
<div class="kai-hero">
    <div style="font-size: 11px; font-weight: 800; letter-spacing: 1px; color: #a7f3d0; margin-bottom: 4px;">AVISHKAR RESEARCH CONVENTION 2026</div>
    <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800;">🌿 कृषी-AI : स्मार्ट पीक रोग निदान प्रणाली</h2>
    <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #d1fae5;">Deep Learning Automated Leaf Identification (Cotton • Soybean • Potato)</p>
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
        'tips': 'पानावरील काळे गोलाकार चट्टे दिसताच दर ८-१० दिवसांनी फवारणी करावी.'
    },
    'Potato Late Blight (बटाटा उशिरा करपा)': {
        'severity': 'अति-तीव्र (Critical Risk)',
        'fertilizer': 'Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold)',
        'dose': '२.५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३५-४० ग्रॅम)',
        'bio': 'स्यूडोमोनास फ्लुओरेसेन्स ५ मिली प्रति लिटर पाणी.',
        'tips': 'धुक्याच्या वातावरणात पाणी देणे टाळावे व झाडांच्या बुंध्याशी हवा खेळती ठेवावी.'
    },
    'Potato Healthy Leaf (निरोगी बटाटा पान)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '19:19:19 (Water Soluble NPK) + Micronutrients',
        'dose': '५ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ७०-७५ ग्रॅम)',
        'bio': 'दशपर्णी अर्क किंवा जीवामृत दर १५ दिवसांनी द्यावे.',
        'tips': 'समतोल खत व्यवस्थापनाने पिकाची नैसर्गिक प्रतिकारशक्ती टिकवून ठेवा.'
    },
    'Diseased Cotton Leaf (रोगग्रस्त कापूस पान)': {
        'severity': 'मध्यम (Moderate)',
        'fertilizer': 'Copper Oxychloride 50% WP (COC) + Streptocycline',
        'dose': 'COC ३० ग्रॅम + स्ट्रेप्टोसायक्लिन २ ग्रॅम (प्रति १५ लिटर पंप)',
        'bio': 'तांबेयुक्त ताक फवारणी किंवा निंबोळी अर्क ५%.',
        'tips': 'जिवाणूजन्य करपा रोखण्यासाठी स्वच्छ सूर्यप्रकाशात सकाळी फवारणी करावी.'
    },
    'Diseased Cotton Plant (रोगग्रस्त कापूस झाड)': {
        'severity': 'तीव्र (High Risk)',
        'fertilizer': 'Carbendazim 12% + Mancozeb 63% WP (SAAF)',
        'dose': '२ ग्रॅम प्रति लिटर पाणी (१५ लिटर पंपासाठी ३० ग्रॅम)',
        'bio': 'ट्रायकोडर्मा हरझियानम जमिनीतून ड्रेचिंग करावे.',
        'tips': 'रोगट फांद्या छाटून नष्ट कराव्यात; नत्राचा अतिवापर टाळावा.'
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
        'bio': 'ह्युमिक ॲसिड १२% मुळाशी सोडावे.',
        'tips': 'पांढऱ्या मुळ्या वाढवण्यासाठी उत्तम.'
    },
    'Soybean Caterpillar Damage (सोयाबीन पान - अळी प्रादुर्भाव)': {
        'severity': 'तीव्र नुकसान (High)',
        'fertilizer': 'Chlorantraniliprole 18.5% SC (Coragen) / Emamectin Benzoate 5% SG',
        'dose': 'कोराजन ६ मिली किंवा इमामेक्टिन बेन्झोएट १० ग्रॅम (प्रति पंप)',
        'bio': 'निंबोळी अर्क ५% (५० मिली प्रति पंप) किंवा Bt पावडर.',
        'tips': 'पाने खाणाऱ्या अळ्यांचा सुरुवातीच्या अवस्थेतच बंदोबस्त करा.'
    },
    'Soybean Leaf Beetle Damage (सोयाबीन पान - भुंगा प्रादुर्भाव)': {
        'severity': 'मध्यम (Moderate)',
        'fertilizer': 'Lambda Cyhalothrin 4.9% CS / Quinalphos 25% EC',
        'dose': 'लॅम्बडा १५ मिली किंवा क्विनॉलफॉस ३० मिली (प्रति पंप)',
        'bio': 'Beauveria bassiana ५ ग्रॅम प्रति लिटर फवारणी.',
        'tips': 'भुंग्यांसाठी शेताच्या कडेने पिवळे चिकट सापळे लावावेत.'
    },
    'Soybean Healthy Leaf (निरोगी सोयाबीन पान)': {
        'severity': 'सुरक्षित (Healthy)',
        'fertilizer': '00:52:34 + Chelated Zinc',
        'dose': '००:५२:३४ ५ ग्रॅम + चिलेटेड झिंक ०.५ ग्रॅम प्रति लिटर पाणी',
        'bio': 'जीवामृत आणि वेस्ट डीकंपोजरचा वापर.',
        'tips': 'फुलोरा आणि शेंगा भरण्याच्या अवस्थेत संतुलित पोषण द्या.'
    }
}

# 5. Input Section (आधीसारखा एकच साधा इनपुट बॉक्स)
st.markdown('<p style="font-size:0.85rem; font-weight:700; color:#059669; text-transform:uppercase;">पानाचा नमुना द्या · Upload Leaf Sample</p>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)
    input_mode = st.radio("माध्यम निवडा:", ("गॅलरी (Upload)", "कॅमेरा (Camera)"), horizontal=True)
    if input_mode == "गॅलरी (Upload)":
        uploaded_file = st.file_uploader("पानाचा स्पष्ट फोटो निवडा:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        uploaded_file = st.camera_input("फोटो काढा:", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# 6. Leaf Identification & Diagnostic Engine
if uploaded_file is not None and models_ready:
    img = Image.open(uploaded_file).convert('RGB')
    
    st.markdown('<div class="kai-card">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    resized_img = img.resize((224, 224))
    arr = np.array(resized_img, dtype=np.float32)

    # तिन्ही मॉडेल्सचे प्रेडिक्शन्स
    preds_p = potato_model(np.expand_dims(arr, axis=0), training=False).numpy()[0]
    if np.sum(preds_p) > 1.05 or np.sum(preds_p) < 0.95:
        preds_p = tf.nn.softmax(preds_p).numpy()

    preds_s = soybean_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(preds_s) > 1.05 or np.sum(preds_s) < 0.95:
        preds_s = tf.nn.softmax(preds_s).numpy()

    preds_c = cotton_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(preds_c) > 1.05 or np.sum(preds_c) < 0.95:
        preds_c = tf.nn.softmax(preds_c).numpy()

    idx_p, conf_p = int(np.argmax(preds_p)), float(np.max(preds_p))
    idx_s, conf_s = int(np.argmax(preds_s)), float(np.max(preds_s))
    idx_c, conf_c = int(np.argmax(preds_c)), float(np.max(preds_c))

    # =========================================================
    # वनस्पतीशास्त्रीय वैशिष्ट्ये (Morphology & Feature Extraction)
    # =========================================================
    gray_arr = np.array(resized_img.convert('L'), dtype=np.float32)
    
    # १. पानावरील पांढऱ्या नागमोडी रेषा (टोमॅटो लीफ मायनर / अनोळखी लक्षणे)
    white_miner_lines = np.sum(gray_arr > 210) / (224 * 224)
    
    # २. रंगांचे गुणोत्तर (Color Ratios)
    r_chan, g_chan, b_chan = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mean_r, mean_g, mean_b = np.mean(r_chan), np.mean(g_chan), np.mean(b_chan)
    cyan_green_ratio = (mean_g - mean_r) / (mean_b + 1e-5)
    
    # ३. कापसाचे पंजाकार लोब्स आणि शिरांचे ग्रेडियंट (Cotton Leaf Texture / Contour Sharpness)
    sobel_h = np.abs(gray_arr[1:, :] - gray_arr[:-1, :])
    sobel_w = np.abs(gray_arr[:, 1:] - gray_arr[:, :-1])
    edge_density = (np.mean(sobel_h) + np.mean(sobel_w)) / 2.0

    # =========================================================
    # स्वयंचलित पान ओळख निर्णय (Automatic Leaf Classifier Logic)
    # =========================================================
    is_out_of_scope = False

    # जर टोमॅटोचे पान असेल (पांढऱ्या नागमोडी रेषा किंवा विशिष्ट निळसर-हिरवी छटा)
    if (white_miner_lines > 0.05 and idx_p == 0) or (cyan_green_ratio > 0.22 and mean_b > 68.0 and idx_p == 0):
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
                <b>कृषी-AI</b> सध्या केवळ <b>कापूस, सोयाबीन आणि बटाटा</b> या ३ पिकांच्या अचूक रोगनिदानासाठी तयार केले आहे.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # १. कापसाचे पान ओळखणे:
        # कापसाचे पान चिवट, गडद आणि जाड शिरा/लोब्स असलेले असते (Edge density व कापूस मॉडेलचे संकेत)
        is_cotton_leaf = (conf_c > 0.55 and (edge_density > 10.5 or conf_c > conf_s))
        
        # २. सोयाबीन पान ओळखणे:
        # सोयाबीनचे पान मऊ, अंडाकृती आणि कीटकांचे चट्टे (Caterpillar/Beetle) सहज पकडणारे असते
        is_soybean_leaf = (conf_s > 0.50 and idx_s in [0, 1] and edge_density <= 10.5)

        # अंतिम पीक निश्चिती
        if is_cotton_leaf and not is_soybean_leaf:
            detected_crop = "cotton"
        elif is_soybean_leaf:
            detected_crop = "soybean"
        elif conf_p > 0.70 and idx_p in [0, 1]:
            detected_crop = "potato"
        elif conf_c >= conf_s:
            detected_crop = "cotton"
        else:
            detected_crop = "soybean"

        # निवडलेल्या पिकाचे अंतिम निकाल
        if detected_crop == "cotton":
            crop_name = "☁️ कापूस (Cotton Leaf)"
            diagnosed_label = COTTON_CLASSES[idx_c]
            final_conf = conf_c * 100
            current_classes = COTTON_CLASSES
            current_preds = preds_c
        elif detected_crop == "soybean":
            crop_name = "🌱 सोयाबीन (Soybean Leaf)"
            diagnosed_label = SOYBEAN_CLASSES[idx_s]
            final_conf = conf_s * 100
            current_classes = SOYBEAN_CLASSES
            current_preds = preds_s
        else:
            crop_name = "🥔 बटाटा (Potato Leaf)"
            diagnosed_label = POTATO_CLASSES[idx_p]
            final_conf = conf_p * 100
            current_classes = POTATO_CLASSES
            current_preds = preds_p

        info = TREATMENTS[diagnosed_label]
        sev_text = info['severity']
        sev_cls = 'sev-healthy' if 'सुरक्षित' in sev_text else ('sev-mod' if 'मध्यम' in sev_text else 'sev-crit')

        # निकाल कार्ड
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

        # संभाव्यता विवरण
        st.markdown('<div class="kai-card">', unsafe_allow_html=True)
        st.markdown("<b>संभाव्यता विवरण (Probabilities):</b>", unsafe_allow_html=True)
        order = np.argsort(current_preds)[::-1]
        for i in order:
            cls_name = current_classes[i]
            pct = float(current_preds[i]) * 100
            st.write(f"• **{cls_name}** : `{pct:.1f}%`")
            st.progress(min(max(float(current_preds[i]), 0.0), 1.0))
        st.markdown('</div>', unsafe_allow_html=True)

        # सल्ला व औषधोपचार कार्ड्स
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
            
