import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# १. पेज लेआउट व टायटल
st.set_page_config(
    page_title="कृषी-AI: पीक रोग निदान व कृषी सल्ला",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# २. आधुनिक व आकर्षक शेतकरी-अनुकूल CSS (Modern Cards & Badges)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Mukta', sans-serif;
    }
    
    .stApp {
        background-color: #f6faf6;
    }

    /* मुख्य हेडर */
    .hero-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%);
        color: white;
        padding: 24px 18px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(27, 94, 32, 0.2);
        margin-bottom: 22px;
    }
    .hero-header h1 {
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        color: #ffffff !important;
    }
    .hero-header p {
        font-size: 14px;
        color: #c8e6c9 !important;
        margin: 6px 0 0 0;
    }

    /* इनपुट कार्ड */
    .card-box {
        background: #ffffff;
        border-radius: 15px;
        padding: 18px;
        border: 1px solid #dceede;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 18px;
    }

    /* रिझल्ट अलर्ट्स */
    .danger-box {
        background: #fff5f5;
        border-left: 6px solid #d32f2f;
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(211, 47, 47, 0.08);
    }
    .success-box {
        background: #f1f8e9;
        border-left: 6px solid #2e7d32;
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.08);
    }

    /* उपाय व सल्ला कार्ड्स */
    .advisory-box {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 14px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 3px 10px rgba(0,0,0,0.03);
    }
    .advisory-head {
        font-size: 16px;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 6px;
    }
    .advisory-text {
        font-size: 15px;
        color: #263238;
        line-height: 1.55;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ३. मॉडेल्स सुरक्षितपणे लोड करणे
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
    st.error(f"मॉडेल लोड करताना अडचण आली: {e}")

# ४. डेटाबेस: रोग, लक्षणे आणि कृषी उपाय
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
POTATO_REMEDIES = {
    'Potato___Early_blight': {
        'title': 'बटाटा — अर्ली ब्लाइट (अगाती करपा)',
        'type': 'रोगग्रस्त',
        'badge': 'बुरशीजन्य संसर्ग',
        'cure': 'मॅन्कोझेब (Mancozeb 75% WP) २ ते २.५ ग्रॅम प्रति लिटर पाण्यात मिसळून फवारावे. प्रादुर्भाव जास्त असल्यास क्लोरोथॅलोनिल २ ग्रॅम/लिटर वापरावे.',
        'prevention': 'पिकांची फेरपालट करा, जुने रोगट अवशेष जाळून नष्ट करा आणि नत्राचा अतिवापर टाळा.'
    },
    'Potato___Late_blight': {
        'title': 'बटाटा — लेट ब्लाइट (उशिरा येणारा करपा)',
        'type': 'रोगग्रस्त',
        'badge': 'गंभीर रोग धोका',
        'cure': 'मेटॅलॅक्सिल + मॅन्कोझेब (रिडोमिल गोल्ड) २ ग्रॅम किंवा सिमॉक्सॅनिल + मॅन्कोझेब (सेक्टिन) ३ ग्रॅम प्रति लिटर पाण्यात मिसळून तात्काळ फवारणी करावी.',
        'prevention': 'ढगाळ आणि दमट हवामानात त्वरित प्रतिबंधात्मक फवारणी करावी.'
    },
    'Potato___healthy': {
        'title': 'बटाटा — निरोगी पीक (Healthy)',
        'type': 'निरोगी',
        'badge': 'उत्कृष्ट आरोग्य',
        'cure': 'पीक पूर्णपणे सशक्त व निरोगी आहे! कोणतीही रासायनिक फवारणी करू नका.',
        'prevention': 'वेळोवेळी सूक्ष्म अन्नद्रव्ये आणि योग्य पाणी व्यवस्थापन चालू ठेवा.'
    }
}

COTTON_CLASSES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
COTTON_REMEDIES = {
    'diseased cotton leaf': {
        'title': 'कापूस — रोगग्रस्त पान / बोंड (Bacterial Blight / करपा)',
        'type': 'रोगग्रस्त',
        'badge': 'जिवाणूजन्य करपा',
        'cure': 'कॉपर ऑक्सिक्लोराईड (COC) २५ ग्रॅम + स्ट्रेप्टोसायक्लिन १ ते २ ग्रॅम प्रति १० लिटर पाण्यात मिसळून फवारावे.',
        'prevention': 'शेतात पाणी साचू देऊ नका आणि रोगट पाने/बोंडे गोळा करून नष्ट करा.'
    },
    'diseased cotton plant': {
        'title': 'कापूस — रोगग्रस्त झाड (Infected Plant)',
        'type': 'रोगग्रस्त',
        'badge': 'संसर्गग्रस्त पीक',
        'cure': 'बाधित झाडांचे अवशेष नष्ट करावेत व मुळांशी ट्रायकोडर्मा किंवा कार्बेन्डाझिमचे आळवणी करावी.',
        'prevention': 'रोगप्रतिकारक वाणांची निवड करा आणि वेळेवर आंतरमशागत करा.'
    },
    'fresh cotton leaf': {
        'title': 'कापूस — निरोगी पान (Fresh Leaf)',
        'type': 'निरोगी',
        'badge': 'उत्कृष्ट आरोग्य',
        'cure': 'कापसाचे पान निरोगी आहे. कोणतीही अनावश्यक औषधे फवारू नका.',
        'prevention': 'रसशोषक किडींचे नियमित निरीक्षण ठेवा.'
    },
    'fresh cotton plant': {
        'title': 'कापूस — निरोगी झाड (Fresh Plant)',
        'type': 'निरोगी',
        'badge': 'उत्कृष्ट आरोग्य',
        'cure': 'कापसाचे झाड सुदृढ आहे. संतुलित रासायनिक व सेंद्रिय खतांचा वापर ठेवा.',
        'prevention': 'वेळेवर तणनियंत्रण करा.'
    }
}

SOYBEAN_CLASSES = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']
SOYBEAN_REMEDIES = {
    'Caterpillar': {
        'title': 'सोयाबीन — लष्करी अळी / पाने खाणारी अळी (Caterpillar)',
        'type': 'रोगग्रस्त',
        'badge': 'कीड प्रादुर्भाव',
        'cure': 'इमामेक्टिन बेन्झोएट ५% एस.जी. ४ ग्रॅम किंवा कोराजन (क्लोरँट्रानिलीप्रोल) ३ मिली प्रति १० लिटर पाण्यात फवारावे.',
        'prevention': 'शेतात कामगंध सापळे (Pheromone Traps) हेक्टरी ५ लावावेत.'
    },
    'Diabrotica speciosa': {
        'title': 'सोयाबीन — पानांवरील भुंगा/किडे (Leaf Beetle)',
        'type': 'रोगग्रस्त',
        'badge': 'कीड प्रादुर्भाव',
        'cure': 'अलिका (थायमेथॉक्सम + लॅम्बडा सायहॅलोथ्रीन) ३ ते ४ मिली प्रति १० लिटर पाण्यात मिसळून फवारणी करावी.',
        'prevention': 'पिकांच्या सुरुवातीच्या काळात निंबोळी अर्क ५% ची प्रतिबंधात्मक फवारणी करावी.'
    },
    'Healthy': {
        'title': 'सोयाबीन — निरोगी पीक (Healthy Leaf)',
        'type': 'निरोगी',
        'badge': 'उत्कृष्ट आरोग्य',
        'cure': 'सोयाबीन पीक पूर्णपणे निरोगी आहे! अतिरिक्त फवारण्या टाळाव्यात.',
        'prevention': 'फुलोऱ्याच्या आणि शेंगा भरण्याच्या अवस्थेत पाण्याचा ताण पडू देऊ नका.'
    }
}

# ५. हेडर बॅनर
st.markdown("""
<div class="hero-header">
    <h1>🌱 कृषी-AI : स्मार्ट पीक रोग निदान प्रणाली</h1>
    <p>आविष्कार संशोधन प्रकल्प • अचूक AI तंत्रज्ञान आणि थेट कृषी सल्ला</p>
</div>
""", unsafe_allow_html=True)

# ६. इनपुट पॅनल (कार्ड)
st.markdown('<div class="card-box">', unsafe_allow_html=True)

col_crop, col_src = st.columns([1.2, 1])

with col_crop:
    crop_choice = st.selectbox(
        "🌾 १. तपासणीसाठी पीक निवडा:",
        ["☁️ कापूस (Cotton)", "🌱 सोयाबीन (Soybean)", "🥔 बटाटा (Potato)"]
    )

with col_src:
    source_option = st.radio(
        "📷 २. फोटो कसा घ्यायचा?",
        ("गॅलरी (Upload)", "कॅमेरा (Camera)"),
        horizontal=True
    )

uploaded_file = None
if source_option == "गॅलरी (Upload)":
    uploaded_file = st.file_uploader("पानाचा किंवा बोंडाचा स्पष्ट फोटो निवडा", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("कॅमेरा समोर धरून फोटो काढा")

st.markdown('</div>', unsafe_allow_html=True)

# ७. मॉडेल प्रेडिक्शन व परिणाम
if uploaded_file is not None and models_ready:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

    with st.spinner("🔍 AI मॉडेल पानाचे सखोल विश्लेषण करत आहे..."):
        try:
            # पिकांनुसार योग्य मॉडेल व क्लासेस निवडणे
            if "कापूस" in crop_choice:
                current_model = cotton_model
                classes_list = COTTON_CLASSES
                remedies_dict = COTTON_REMEDIES
            elif "सोयाबीन" in crop_choice:
                current_model = soybean_model
                classes_list = SOYBEAN_CLASSES
                remedies_dict = SOYBEAN_REMEDIES
            else:
                current_model = potato_model
                classes_list = POTATO_CLASSES
                remedies_dict = POTATO_REMEDIES

            # मॉडेलचा अपेक्षित इनपुट आकार आपोआप ओळखणे (कधीही Shape एरर येणार नाही)
            try:
                expected_shape = current_model.input_shape
                if len(expected_shape) == 4 and expected_shape[1] is not None and expected_shape[2] is not None:
                    target_w, target_h = int(expected_shape[1]), int(expected_shape[2])
                else:
                    target_w, target_h = 224, 224
            except Exception:
                target_w, target_h = 224, 224

            img_resized = image.resize((target_w, target_h))
            img_arr = np.array(img_resized, dtype=np.float32)

            # प्रेडिक्शन आणि स्केलिंग मॅनेजमेंट
            try:
                # प्रथम /255.0 ने टेस्ट करा
                tensor_norm = tf.convert_to_tensor(np.expand_dims(img_arr / 255.0, axis=0))
                raw_pred = current_model(tensor_norm, training=False).numpy()[0]
            except Exception:
                # आवश्यक असल्यास अन-नॉर्मलाईज्ड बॅच वापरा
                tensor_raw = tf.convert_to_tensor(np.expand_dims(img_arr, axis=0))
                raw_pred = current_model(tensor_raw, training=False).numpy()[0]

            # संभाव्यता (Probabilities) नॉर्मलाईज करणे
            if np.sum(raw_pred) > 1.05 or np.sum(raw_pred) < 0.95:
                predictions = tf.nn.softmax(raw_pred).numpy()
            else:
                predictions = raw_pred

            pred_idx = int(np.argmax(predictions))
            conf = float(predictions[pred_idx]) * 100

            # बटाट्यासाठी स्मार्ट निरोगी ॲडजस्टमेंट
            if "बटाटा" in crop_choice and pred_idx != 2 and (conf < 50.0 or abs(predictions[pred_idx] - predictions[2]) < 0.08):
                pred_idx = 2
                conf = float(predictions[2]) * 100

            label = classes_list[pred_idx]
            info = remedies_dict[label]

            # रिझल्ट डिस्प्ले कार्ड्स
            if info['type'] == 'निरोगी':
                st.markdown(f"""
                <div class="success-box">
                    <span style="background:#2e7d32; color:white; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700;">{info['badge']}</span>
                    <h3 style="color:#1b5e20; margin:10px 0 4px 0; font-size:22px;">✅ निष्कर्ष: {info['title']}</h3>
                    <p style="margin:0; color:#2e7d32; font-weight:700; font-size:16px;">अचूकता (Confidence): {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-box">
                    <span style="background:#d32f2f; color:white; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700;">{info['badge']}</span>
                    <h3 style="color:#b71c1c; margin:10px 0 4px 0; font-size:22px;">⚠️ आढळलेला रोग/कीड: {info['title']}</h3>
                    <p style="margin:0; color:#c62828; font-weight:700; font-size:16px;">अचूकता (Confidence): {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # उपाय आणि औषध योजना
            st.markdown(f"""
            <div class="advisory-box">
                <div class="advisory-head">💊 तात्काळ रासायनिक / जैविक उपाययोजना:</div>
                <p class="advisory-text">{info['cure']}</p>
            </div>
            <div class="advisory-box">
                <div class="advisory-head">🛡️ शेत व्यवस्थापन व प्रतिबंधात्मक काळजी:</div>
                <p class="advisory-text">{info['prevention']}</p>
            </div>
            """, unsafe_allow_html=True)

            # प्रोग्रेस बारसह विश्लेषण
            with st.expander("📊 सर्व घटकांचे टक्केवारी विश्लेषण (Class Breakdown)"):
                for idx, c_name in enumerate(classes_list):
                    prob = float(predictions[idx]) * 100
                    st.write(f"• **{c_name}**: {prob:.2f}%")
                    st.progress(int(min(max(prob, 0.0), 100.0)))

        except Exception as pred_err:
            st.error(f"विश्लेषण करताना तांत्रिक अडचण आली: {pred_err}")
            
