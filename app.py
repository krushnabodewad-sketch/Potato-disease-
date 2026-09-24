import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# १. पेज संरचना
st.set_page_config(
    page_title="कृषी-AI: स्वयंचलित पीक व रोग निदान",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# २. आधुनिक कृषी UI स्टायलिंग
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700;800&display=swap');
    * { font-family: 'Mukta', sans-serif; }
    .stApp { background-color: #f5f8f5; }
    
    .hero-banner {
        background: linear-gradient(135deg, #0d532d 0%, #1b7a43 60%, #2ea35f 100%);
        border-radius: 18px;
        padding: 22px 18px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 22px rgba(27, 122, 67, 0.2);
        margin-bottom: 20px;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 25px;
        font-weight: 800;
        margin: 0;
    }
    .hero-banner p {
        color: #d1f2e2 !important;
        font-size: 13.5px;
        margin-top: 5px;
    }

    .card {
        background: #ffffff;
        border-radius: 15px;
        padding: 18px;
        border: 1px solid #dceede;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }

    .detection-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .chip-crop { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
    .chip-part { background: #fff8e1; color: #f57f17; border: 1px solid #ffe082; }

    .result-danger {
        background: #fff5f5;
        border-left: 6px solid #e53935;
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 14px;
        box-shadow: 0 4px 14px rgba(229, 57, 53, 0.08);
    }
    .result-success {
        background: #f1f8e9;
        border-left: 6px solid #2e7d32;
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 14px;
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.08);
    }
    
    .remedy-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 12px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 3px 10px rgba(0,0,0,0.03);
    }
    .remedy-title {
        font-weight: 700;
        font-size: 16px;
        color: #1b5e20;
        margin-bottom: 5px;
    }
    .remedy-text {
        font-size: 14.5px;
        color: #263238;
        line-height: 1.55;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ३. मॉडेल्स लोड करणे
@st.cache_resource
def load_all_models():
    p_m = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False)
    c_m = tf.keras.models.load_model('cotton_model.h5', compile=False)
    s_m = tf.keras.models.load_model('soybean_model.h5', compile=False)
    feat_m = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
    return p_m, c_m, s_m, feat_m

try:
    potato_model, cotton_model, soybean_model, feature_model = load_all_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"मॉडेल लोड करताना त्रुटी आली: {e}")

# ४. रोग वर्ग व कृषी सल्ला
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
POTATO_REMEDIES = {
    'Potato___Early_blight': {
        'title': 'अर्ली ब्लाइट (अगाती करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'मॅन्कोझेब (Mancozeb 75% WP) २ ते २.५ ग्रॅम प्रति लिटर पाण्यात मिसळून संपूर्ण पानांवर फवारावे.',
        'prevention': 'पिकांची फेरपालट करा व रोगट पाने गोळा करून नष्ट करा.'
    },
    'Potato___Late_blight': {
        'title': 'लेट ब्लाइट (उशिरा येणारा करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'मेटॅलॅक्सिल + मॅन्कोझेब (रिडोमिल गोल्ड) २ ग्रॅम प्रति लिटर पाण्यात फवारावे.',
        'prevention': 'ढगाळ व दमट हवामानात त्वरित प्रतिबंधात्मक फवारणी करावी.'
    },
    'Potato___healthy': {
        'title': 'निरोगी पीक (Healthy)',
        'type': 'निरोगी',
        'cure': 'पीक पूर्णपणे सुदृढ आहे. कोणत्याही औषधाची गरज नाही.',
        'prevention': 'वेळोवेळी योग्य खत आणि पाणी व्यवस्थापन ठेवा.'
    }
}

COTTON_CLASSES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
COTTON_REMEDIES = {
    'diseased cotton leaf': {
        'title': 'रोगग्रस्त पान / बोंड (Bacterial Blight / करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'कॉपर ऑक्सिक्लोराईड (COC) २५ ग्रॅम + स्ट्रेप्टोसायक्लिन १ ते २ ग्रॅम प्रति १० लिटर पाण्यात फवारावे.',
        'prevention': 'शेतात पाणी साचू देऊ नका आणि गळून पडलेले रोगट भाग नष्ट करा.'
    },
    'diseased cotton plant': {
        'title': 'रोगग्रस्त झाड (Infected Plant)',
        'type': 'रोगग्रस्त',
        'cure': 'बाधित झाडे नष्ट करावीत व मुळांशी ट्रायकोडर्मा किंवा बुरशीनाशकाचे आळवणी करावी.',
        'prevention': 'रोगप्रतिकारक वाण वापरा.'
    },
    'fresh cotton leaf': {
        'title': 'निरोगी कापूस पान / बोंड (Fresh)',
        'type': 'निरोगी',
        'cure': 'कापसाचे पीक निरोगी आहे. अनावश्यक फवारणी टाळावी.',
        'prevention': 'रसशोषक किडींचे नियमित निरीक्षण ठेवा.'
    },
    'fresh cotton plant': {
        'title': 'निरोगी कापूस पीक (Fresh Plant)',
        'type': 'निरोगी',
        'cure': 'कापूस झाड सशक्त व निरोगी आहे.',
        'prevention': 'संतुलित खतांचा वापर ठेवा.'
    }
}

SOYBEAN_CLASSES = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']
SOYBEAN_REMEDIES = {
    'Caterpillar': {
        'title': 'लष्करी अळी / पाने खाणारी अळी (Caterpillar)',
        'type': 'रोगग्रस्त',
        'cure': 'इमामेक्टिन बेन्झोएट ५% एस.जी. ४ ग्रॅम किंवा कोराजन ३ मिली प्रति १० लिटर पाण्यात फवारावे.',
        'prevention': 'शेतात एकरी ३ कामगंध सापळे (Pheromone Traps) लावा.'
    },
    'Diabrotica speciosa': {
        'title': 'पानांवरील भुंगा/किडे (Leaf Beetle)',
        'type': 'रोगग्रस्त',
        'cure': 'अलिका (थायमेथॉक्सम + लॅम्बडा सायहॅलोथ्रीन) ३ ते ४ मिली प्रति १० लिटर पाण्यात फवारावे.',
        'prevention': 'सुरुवातीला निंबोळी अर्क ५% फवारा.'
    },
    'Healthy': {
        'title': 'निरोगी सोयाबीन पीक (Healthy Leaf)',
        'type': 'निरोगी',
        'cure': 'सोयाबीनचे पीक निरोगी आहे! अतिरिक्त फवारण्या टाळा.',
        'prevention': 'पाण्याचा ताण पडू देऊ नका.'
    }
}

# ५. अवयव (पान/फूल/फळ) ओळखणे
def get_part_and_features(img_pil):
    resized = img_pil.resize((224, 224))
    pre = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(np.array(resized, dtype=np.float32), axis=0))
    raw_pred = feature_model(pre, training=False)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(raw_pred.numpy(), top=8)[0]
    
    flower_kw = ['flower', 'daisy', 'rose', 'blossom', 'petal', 'sunflower', 'dahlia', 'marigold', 'tulip', 'pot']
    fruit_kw = ['fruit', 'boll', 'berry', 'apple', 'orange', 'pomegranate', 'pod', 'cotton', 'seed', 'vegetable', 'cucumber']
    
    part = "🍃 पान (Leaf / Foliage)"
    for _, name, score in decoded:
        n = name.lower()
        if any(k in n for k in flower_kw) and score > 0.15:
            part = "🌸 फूल (Flower)"
            break
        elif any(k in n for k in fruit_kw) and score > 0.15:
            part = "🍏 फळ / बोंड / शेंग (Fruit / Boll / Pod)"
            break
            
    return part

# ६. हेडर
st.markdown("""
<div class="hero-banner">
    <h1>🌱 कृषी-AI : स्वयंचलित पीक व रोग ओळख प्रणाली</h1>
    <p>आविष्कार संशोधन प्रकल्प • १००% ऑटो-डिटेक्ट • नो मॅन्युअल सिलेक्शन</p>
</div>
""", unsafe_allow_html=True)

# ७. इनपुट कार्ड
st.markdown('<div class="card">', unsafe_allow_html=True)
col_opt, col_file = st.columns([1, 1.2])

with col_opt:
    source_option = st.radio("📷 फोटो कसा घ्यायचा?", ("गॅलरी (Upload)", "कॅमेरा (Camera)"), horizontal=True)

with col_file:
    uploaded_file = None
    if source_option == "गॅलरी (Upload)":
        uploaded_file = st.file_uploader("कोणत्याही पिकाचा फोटो निवडा", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("कॅमेरा समोर धरून फोटो काढा")
st.markdown('</div>', unsafe_allow_html=True)

# ८. प्रेडिक्शन व स्वयंचलित निर्णय
if uploaded_file is not None and models_ready:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

    with st.spinner("🤖 AI आपोआप पीक, अवयव आणि रोग शोधत आहे..."):
        try:
            detected_part = get_part_and_features(image)

            # १. कापूस प्रेडिक्शन
            img_c = image.resize((224, 224))
            arr_c = tf.convert_to_tensor(np.expand_dims(np.array(img_c, dtype=np.float32) / 255.0, axis=0))
            raw_c = cotton_model(arr_c, training=False).numpy()[0]
            prob_c = tf.nn.softmax(raw_c).numpy() if (np.sum(raw_c) > 1.05 or np.sum(raw_c) < 0.95) else raw_c
            idx_c = int(np.argmax(prob_c))
            conf_c = float(prob_c[idx_c])

            # २. सोयाबीन प्रेडिक्शन
            img_s = image.resize((224, 224))
            arr_s = tf.convert_to_tensor(np.expand_dims(np.array(img_s, dtype=np.float32) / 255.0, axis=0))
            raw_s = soybean_model(arr_s, training=False).numpy()[0]
            prob_s = tf.nn.softmax(raw_s).numpy() if (np.sum(raw_s) > 1.05 or np.sum(raw_s) < 0.95) else raw_s
            idx_s = int(np.argmax(prob_s))
            conf_s = float(prob_s[idx_s])

            # ३. बटाटा प्रेडिक्शन
            img_p = image.resize((224, 224))
            arr_p = tf.convert_to_tensor(np.expand_dims(np.array(img_p, dtype=np.float32), axis=0))
            try:
                raw_p = potato_model(arr_p / 255.0, training=False).numpy()[0]
            except Exception:
                raw_p = potato_model(arr_p, training=False).numpy()[0]
            prob_p = tf.nn.softmax(raw_p).numpy() if (np.sum(raw_p) > 1.05 or np.sum(raw_p) < 0.95) else raw_p
            idx_p = int(np.argmax(prob_p))
            conf_p = float(prob_p[idx_p])

            # स्मार्ट व्हिज्युअल राऊटिंग:
            # बटाटा मॉडेल इतर हिरव्या पानांना चुकीचा जास्त स्कोअर देत असल्याने त्याला कॅलिब्रेट करणे
            adj_p = conf_p * 0.72
            adj_s = conf_s * 1.25
            adj_c = conf_c * 1.15

            if adj_s >= adj_c and adj_s >= adj_p:
                chosen_crop = "soybean"
            elif adj_c >= adj_s and adj_c >= adj_p:
                chosen_crop = "cotton"
            else:
                chosen_crop = "potato"

            # निकाल मॅपिंग
            if chosen_crop == "soybean":
                crop_title = "सोयाबीन (Soybean)"
                label = SOYBEAN_CLASSES[idx_s]
                info = SOYBEAN_REMEDIES[label]
                conf = conf_s * 100
            elif chosen_crop == "cotton":
                crop_title = "कापूस (Cotton)"
                label = COTTON_CLASSES[idx_c]
                info = COTTON_REMEDIES[label]
                conf = conf_c * 100
            else:
                crop_title = "बटाटा (Potato)"
                label = POTATO_CLASSES[idx_p]
                info = POTATO_REMEDIES[label]
                conf = conf_p * 100

            # चिप्स
            st.markdown(f"""
            <div style="margin-top: 15px;">
                <span class="detection-chip chip-crop">🌾 ओळखलेले पीक: <b>{crop_title}</b></span>
                <span class="detection-chip chip-part">🔍 अवयव: <b>{detected_part}</b></span>
            </div>
            """, unsafe_allow_html=True)

            # रिझल्ट कार्ड
            if info['type'] == 'निरोगी':
                st.markdown(f"""
                <div class="result-success">
                    <h3 style="color:#1b5e20; margin:0 0 5px 0; font-size:22px;">✅ निष्कर्ष: {crop_title} — {info['title']}</h3>
                    <p style="margin:0; color:#2e7d32; font-weight:700; font-size:16px;">अचूकता (Confidence): {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-danger">
                    <h3 style="color:#b71c1c; margin:0 0 5px 0; font-size:22px;">⚠️ आढळलेला रोग/कीड: {crop_title} — {info['title']}</h3>
                    <p style="margin:0; color:#c62828; font-weight:700; font-size:16px;">अचूकता (Confidence): {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # उपाय
            st.markdown(f"""
            <div class="remedy-card">
                <div class="remedy-title">💊 शिफारस केलेले तात्काळ उपाय:</div>
                <p class="remedy-text">{info['cure']}</p>
            </div>
            <div class="remedy-card">
                <div class="remedy-title">🛡️ प्रतिबंधात्मक शेती व्यवस्थापन:</div>
                <p class="remedy-text">{info['prevention']}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📊 AI मॉडेल तुलनात्मक पडताळणी"):
                st.write(f"• सोयाबीन स्कोअर: {conf_s*100:.2f}%")
                st.write(f"• कापूस स्कोअर: {conf_c*100:.2f}%")
                st.write(f"• बटाटा स्कोअर: {conf_p*100:.2f}%")

        except Exception as pred_err:
            st.error(f"विश्लेषण करताना अडचण आली: {pred_err}")
            
