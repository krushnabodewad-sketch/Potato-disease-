import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Krushi-AI: Smart Peek & Rog Nidan",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Modern Ultra-Clean Agro UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700;800&family=Poppins:wght@500;700&display=swap');
    * { font-family: 'Mukta', 'Poppins', sans-serif; }
    .stApp { background-color: #f4f8f4; }
    
    .hero-banner {
        background: linear-gradient(135deg, #0b4f30 0%, #167a49 60%, #22a061 100%);
        border-radius: 20px;
        padding: 24px 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(22, 122, 73, 0.22);
        margin-bottom: 22px;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
    }
    .hero-banner p {
        color: #d1f2e2 !important;
        font-size: 14px;
        margin-top: 6px;
    }

    .auto-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        color: #fff;
        margin-top: 8px;
    }

    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #d9edd9;
        box-shadow: 0 4px 16px rgba(0,0,0,0.03);
        margin-bottom: 18px;
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
        padding: 16px 20px;
        margin-top: 15px;
        box-shadow: 0 4px 14px rgba(229, 57, 53, 0.08);
    }
    .result-success {
        background: #f1f8e9;
        border-left: 6px solid #2e7d32;
        border-radius: 14px;
        padding: 16px 20px;
        margin-top: 15px;
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.08);
    }
    
    .remedy-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px;
        margin-top: 14px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .remedy-title {
        font-weight: 700;
        font-size: 16px;
        color: #1b5e20;
        margin-bottom: 6px;
    }
    .remedy-text {
        font-size: 15px;
        color: #263238;
        line-height: 1.6;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Models Safely
@st.cache_resource
def load_all_models():
    p_m = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False)
    c_m = tf.keras.models.load_model('cotton_model.h5', compile=False)
    s_m = tf.keras.models.load_model('soybean_model.h5', compile=False)
    # Pretrained ImageNet model for Part/Feature Detection
    feat_m = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
    return p_m, c_m, s_m, feat_m

try:
    potato_model, cotton_model, soybean_model, feature_model = load_all_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"Models load kartana samasya aali: {e}")

# 4. Knowledge Database
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
POTATO_REMEDIES = {
    'Potato___Early_blight': {
        'crop': 'बटाटा (Potato)',
        'title': 'अर्ली ब्लाइट (अगाती करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'मॅन्कोझेब (Mancozeb 75% WP) २ ते २.५ ग्रॅम प्रति लिटर पाण्यात मिसळून संपूर्ण पानांवर फवारावे.',
        'prevention': 'पिकांची फेरपालट करा, रोगट पाने नष्ट करा आणि नत्राचा अतिवापर टाळा.'
    },
    'Potato___Late_blight': {
        'crop': 'बटाटा (Potato)',
        'title': 'लेट ब्लाइट (उशिरा येणारा करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'मेटॅलॅक्सिल + मॅन्कोझेब (रिडोमिल गोल्ड) २ ग्रॅम प्रति लिटर पाण्यात मिसळून तात्काळ फवारणी करावी.',
        'prevention': 'दमट हवामानात प्रतिबंधात्मक फवारणी वेळेवर करा.'
    },
    'Potato___healthy': {
        'crop': 'बटाटा (Potato)',
        'title': 'निरोगी पीक (Healthy)',
        'type': 'निरोगी',
        'cure': 'पीक पूर्णपणे सुदृढ आहे. कोणत्याही औषधाची गरज नाही.',
        'prevention': 'नियमित पाणी आणि खत व्यवस्थापन ठेवा.'
    }
}

COTTON_CLASSES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
COTTON_REMEDIES = {
    'diseased cotton leaf': {
        'crop': 'कापूस (Cotton)',
        'title': 'रोगग्रस्त पान / बोंड (Bacterial Blight / करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'कॉपर ऑक्सिक्लोराईड (COC) २५ ग्रॅम + स्ट्रेप्टोसायक्लिन १ ते २ ग्रॅम प्रति १० लिटर पाण्यात मिसळून फवारावे.',
        'prevention': 'शेतात पाणी साचू देऊ नका आणि गळून पडलेली रोगट पाने/बोंडे नष्ट करा.'
    },
    'diseased cotton plant': {
        'crop': 'कापूस (Cotton)',
        'title': 'रोगग्रस्त कापूस झाड (Infected Plant)',
        'type': 'रोगग्रस्त',
        'cure': 'बाधित झाडांचे अवशेष नष्ट करावेत व मुळांशी ट्रायकोडर्मा किंवा बुरशीनाशकाचे आळवणी करावी.',
        'prevention': 'रोगप्रतिकारक वाणांची निवड करा.'
    },
    'fresh cotton leaf': {
        'crop': 'कापूस (Cotton)',
        'title': 'निरोगी कापूस पान (Fresh Leaf)',
        'type': 'निरोगी',
        'cure': 'कापसाचे पान निरोगी आहे. अनावश्यक फवारणी टाळा.',
        'prevention': 'रसशोषक किडींचे नियमित निरीक्षण ठेवा.'
    },
    'fresh cotton plant': {
        'crop': 'कापूस (Cotton)',
        'title': 'निरोगी कापूस झाड (Fresh Plant)',
        'type': 'निरोगी',
        'cure': 'झाड सशक्त व निरोगी आहे.',
        'prevention': 'संतुलित खतांचा वापर ठेवा.'
    }
}

SOYBEAN_CLASSES = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']
SOYBEAN_REMEDIES = {
    'Caterpillar': {
        'crop': 'सोयाबीन (Soybean)',
        'title': 'लष्करी अळी / पाने खाणारी अळी (Caterpillar)',
        'type': 'रोगग्रस्त',
        'cure': 'इमामेक्टिन बेन्झोएट ५% एस.जी. ४ ग्रॅम किंवा कोराजन ३ मिली प्रति १० लिटर पाण्यात फवारावे.',
        'prevention': 'शेतात एकरी २ ते ३ कामगंध सापळे (Pheromone Traps) लावा.'
    },
    'Diabrotica speciosa': {
        'crop': 'सोयाबीन (Soybean)',
        'title': 'पानांवरील भुंगा/किडे (Leaf Beetle)',
        'type': 'रोगग्रस्त',
        'cure': 'अलिका (थायमेथॉक्सम + लॅम्बडा सायहॅलोथ्रीन) ३ ते ४ मिली प्रति १० लिटर पाण्यात फवारावे.',
        'prevention': 'सुरुवातीला निंबोळी अर्क ५% ची फवारणी करा.'
    },
    'Healthy': {
        'crop': 'सोयाबीन (Soybean)',
        'title': 'निरोगी सोयाबीन पान (Healthy Leaf)',
        'type': 'निरोगी',
        'cure': 'पीक निरोगी आहे! अतिरिक्त फवारण्या टाळा.',
        'prevention': 'पाण्याचा ताण पडू देऊ नका.'
    }
}

# 5. Plant Part Detection (Leaf / Flower / Fruit / Boll)
def identify_plant_part(image_pil):
    img_eval = image_pil.resize((224, 224))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(np.array(img_eval, dtype=np.float32), axis=0))
    preds = feature_model(x, training=False)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds.numpy(), top=5)[0]
    
    flower_kw = ['flower', 'daisy', 'rose', 'blossom', 'petal', 'sunflower', 'dahlia', 'marigold', 'tulip']
    fruit_kw = ['fruit', 'boll', 'berry', 'apple', 'orange', 'pomegranate', 'pod', 'cotton', 'seed', 'vegetable']
    
    is_flower = any(any(k in label.lower() for k in flower_kw) for _, label, _ in decoded)
    is_fruit = any(any(k in label.lower() for k in fruit_kw) for _, label, _ in decoded)
    
    if is_flower:
        return "🌸 फूल (Flower)"
    elif is_fruit:
        return "🍏 फळ / बोंड / शेंग (Fruit / Boll / Pod)"
    else:
        return "🍃 पान (Leaf / Foliage)"

# 6. UI Header
st.markdown("""
<div class="hero-banner">
    <h1>🌱 कृषी-AI : स्वयंचलित पीक व रोग ओळख प्रणाली</h1>
    <p>आविष्कार संशोधन प्रकल्प • १००% ऑटो-डिटेक्ट • नो मॅन्युअल सिलेक्शन</p>
    <div class="auto-badge">⚡ Auto Crop & Plant Part Identification Active</div>
</div>
""", unsafe_allow_html=True)

# 7. Upload Box (No crop dropdown needed)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📷 पीक किंवा झाडाचा फोटो टाका:")
source_option = st.radio("फोटोचा स्रोत निवडा:", ("गॅलरीतून निवडा (Upload)", "थेट कॅमेऱ्याने फोटो काढा (Camera)"), horizontal=True)

uploaded_file = None
if source_option == "गॅलरीतून निवडा (Upload)":
    uploaded_file = st.file_uploader("कोणत्याही पिकाचे पान, फूल किंवा फळ निवडा", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("कॅमेरा समोर धरून फोटो क्लिक करा")
st.markdown('</div>', unsafe_allow_html=True)

# 8. Automated Multi-Model Prediction
if uploaded_file is not None and models_ready:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

    with st.spinner("🤖 AI आपोआप पीक, अवयव (पान/फूल/फळ) आणि रोग शोधत आहे..."):
        try:
            # 1. Identify Plant Part
            detected_part = identify_plant_part(image)

            # 2. Cotton Prediction
            img_c = image.resize((224, 224))
            arr_c = tf.convert_to_tensor(np.expand_dims(np.array(img_c, dtype=np.float32) / 255.0, axis=0))
            raw_c = cotton_model(arr_c, training=False).numpy()[0]
            prob_c = tf.nn.softmax(raw_c).numpy() if (np.sum(raw_c) > 1.05 or np.sum(raw_c) < 0.95) else raw_c
            idx_c = int(np.argmax(prob_c))
            conf_c = float(prob_c[idx_c])

            # 3. Soybean Prediction
            img_s = image.resize((224, 224))
            arr_s = tf.convert_to_tensor(np.expand_dims(np.array(img_s, dtype=np.float32) / 255.0, axis=0))
            raw_s = soybean_model(arr_s, training=False).numpy()[0]
            prob_s = tf.nn.softmax(raw_s).numpy() if (np.sum(raw_s) > 1.05 or np.sum(raw_s) < 0.95) else raw_s
            idx_s = int(np.argmax(prob_s))
            conf_s = float(prob_s[idx_s])

            # 4. Potato Prediction
            img_p = image.resize((224, 224))
            arr_p = tf.convert_to_tensor(np.expand_dims(np.array(img_p, dtype=np.float32), axis=0))
            try:
                raw_p = potato_model(arr_p / 255.0, training=False).numpy()[0]
            except Exception:
                raw_p = potato_model(arr_p, training=False).numpy()[0]
            prob_p = tf.nn.softmax(raw_p).numpy() if (np.sum(raw_p) > 1.05 or np.sum(raw_p) < 0.95) else raw_p
            idx_p = int(np.argmax(prob_p))
            conf_p = float(prob_p[idx_p])

            # 5. Smart Multi-Factor Ensemble (Automatic Routing)
            # Kapus cha photo potato kade jau naye mhanun cotton visual signatures la preference
            cotton_bias = 1.05
            soybean_bias = 1.0
            potato_bias = 0.95

            scores = {
                'cotton': conf_c * cotton_bias,
                'soybean': conf_s * soybean_bias,
                'potato': conf_p * potato_bias
            }

            best_crop = max(scores, key=scores.get)

            if best_crop == 'cotton':
                label = COTTON_CLASSES[idx_c]
                info = COTTON_REMEDIES[label]
                conf = conf_c * 100
                crop_title = "कापूस (Cotton)"
            elif best_crop == 'soybean':
                label = SOYBEAN_CLASSES[idx_s]
                info = SOYBEAN_REMEDIES[label]
                conf = conf_s * 100
                crop_title = "सोयाबीन (Soybean)"
            else:
                label = POTATO_CLASSES[idx_p]
                info = POTATO_REMEDIES[label]
                conf = conf_p * 100
                crop_title = "बटाटा (Potato)"

            # 6. Display Auto Identified Chips
            st.markdown(f"""
            <div style="margin-top: 15px;">
                <span class="detection-chip chip-crop">🌾 ओळखलेले पीक: <b>{crop_title}</b></span>
                <span class="detection-chip chip-part">🔍 ओळखलेला अवयव: <b>{detected_part}</b></span>
            </div>
            """, unsafe_allow_html=True)

            # 7. Disease Verdict
            if info['type'] == 'निरोगी':
                st.markdown(f"""
                <div class="result-success">
                    <h3 style="color:#1b5e20; margin:0 0 6px 0; font-size:22px;">✅ निष्कर्ष: {crop_title} — {info['title']}</h3>
                    <p style="margin:0; color:#2e7d32; font-weight:700; font-size:16px;">अचूकता (Confidence): {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-danger">
                    <h3 style="color:#b71c1c; margin:0 0 6px 0; font-size:22px;">⚠️ आढळलेला रोग/कीड: {crop_title} — {info['title']}</h3>
                    <p style="margin:0; color:#c62828; font-weight:700; font-size:16px;">अचूकता (Confidence): {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # 8. Expert Remedies
            st.markdown(f"""
            <div class="remedy-card">
                <div class="remedy-title">💊 शिफारस केलेले तात्काळ उपाय:</div>
                <p class="remedy-text">{info['cure']}</p>
            </div>
            <div class="remedy-card">
                <div class="remedy-title">🛡️ प्रतिबंधात्मक व्यवस्थापन:</div>
                <p class="remedy-text">{info['prevention']}</p>
            </div>
            """, unsafe_allow_html=True)

            # 9. Comparison Confidence
            with st.expander("📊 AI मॉडेल तुलनात्मक पडताळणी"):
                st.write(f"• कापूस मॉडेल स्कोअर: {conf_c*100:.2f}%")
                st.write(f"• सोयाबीन मॉडेल स्कोअर: {conf_s*100:.2f}%")
                st.write(f"• बटाटा मॉडेल स्कोअर: {conf_p*100:.2f}%")

        except Exception as pred_err:
            st.error(f"विश्लेषण करताना तांत्रिक त्रुटी आली: {pred_err}")
            
