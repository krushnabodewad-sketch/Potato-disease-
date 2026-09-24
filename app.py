import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# १. पेज सेटअप
st.set_page_config(
    page_title="कृषी-AI | स्मार्ट पीक संरक्षण",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# २. प्रीमियम आणि आकर्षक मॉडर्न CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700;800&family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Mukta', 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #f4fbf5 0%, #eef6f0 100%);
    }

    /* मुख्य हेडर बॅनर */
    .hero-container {
        background: linear-gradient(135deg, #0d5c3a 0%, #178a55 50%, #2bb673 100%);
        border-radius: 20px;
        padding: 28px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(23, 138, 85, 0.25);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .hero-subtitle {
        font-size: 14px;
        color: #d1f2e2 !important;
        margin-top: 6px;
        font-weight: 500;
    }

    /* निवड पॅनल कार्ड */
    .control-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #d8ebd9;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        margin-bottom: 20px;
    }

    /* रिझल्ट कार्ड्स */
    .danger-badge {
        background: linear-gradient(135deg, #fff1f0 0%, #ffeae8 100%);
        border-left: 6px solid #e53935;
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 15px;
        box-shadow: 0 6px 18px rgba(229, 57, 53, 0.08);
    }
    .success-badge {
        background: linear-gradient(135deg, #f0fdf4 0%, #e8fbee 100%);
        border-left: 6px solid #2e7d32;
        border-radius: 14px;
        padding: 18px 20px;
        margin-top: 15px;
        box-shadow: 0 6px 18px rgba(46, 125, 50, 0.08);
    }

    /* सल्ला कार्ड */
    .advisory-box {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        margin-top: 15px;
        border: 1px solid #d0e7d5;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    }
    .advisory-title {
        color: #0d5c3a;
        font-weight: 700;
        font-size: 17px;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }

    /* इमेज फ्रेम */
    [data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        border: 2px solid #e0eee2;
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
    st.error(f"मॉडेल लोड करताना त्रुटी आली: {e}")

# ४. डेटाबेस
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
POTATO_REMEDIES = {
    'Potato___Early_blight': {
        'title': 'बटाटा — अर्ली ब्लाइट (अगाती करपा)',
        'type': 'रोगग्रस्त',
        'status': 'तात्काळ लक्ष देणे आवश्यक',
        'cure': 'मॅन्कोझेब (Mancozeb 75% WP) २ ते २.५ ग्रॅम प्रति लिटर पाण्यात मिसळून संपूर्ण पानांवर फवारावे.'
    },
    'Potato___Late_blight': {
        'title': 'बटाटा — लेट ब्लाइट (उशिरा येणारा करपा)',
        'type': 'रोगग्रस्त',
        'status': 'गंभीर रोग धोका',
        'cure': 'मेटॅलॅक्सिल + मॅन्कोझेब (उदा. रिडोमिल गोल्ड) २ ग्रॅम प्रति लिटर पाण्यात मिसळून तात्काळ फवारणी करावी.'
    },
    'Potato___healthy': {
        'title': 'बटाटा — निरोगी पीक (Healthy)',
        'type': 'निरोगी',
        'status': 'उत्कृष्ट आरोग्य',
        'cure': 'पीक पूर्णपणे सुदृढ आहे. कोणत्याही रासायनिक फवारणीची गरज नाही.'
    }
}

COTTON_CLASSES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
COTTON_REMEDIES = {
    'diseased cotton leaf': {
        'title': 'कापूस — रोगग्रस्त पान / बोंड (Bacterial Blight / करपा)',
        'type': 'रोगग्रस्त',
        'status': 'जीवाणू संसर्ग',
        'cure': 'कॉपर ऑक्सिक्लोराईड (COC) २५ ग्रॅम + स्ट्रेप्टोसायक्लिन १ ते २ ग्रॅम प्रति १० लिटर पाण्यात मिसळून फवारावे.'
    },
    'diseased cotton plant': {
        'title': 'कापूस — रोगग्रस्त झाड (Infected Plant)',
        'type': 'रोगग्रस्त',
        'status': 'संसर्गग्रस्त पीक',
        'cure': 'बाधित झाडांचे अवशेष नष्ट करावेत व मुळांशी ट्रायकोडर्मा किंवा योग्य बुरशीनाशकाचे आळवणी करावी.'
    },
    'fresh cotton leaf': {
        'title': 'कापूस — निरोगी पान (Fresh Leaf)',
        'type': 'निरोगी',
        'status': 'उत्कृष्ट आरोग्य',
        'cure': 'कापसाचे पान निरोगी आहे. संतुलित पाणी आणि खत व्यवस्थापन ठेवा.'
    },
    'fresh cotton plant': {
        'title': 'कापूस — निरोगी झाड (Fresh Plant)',
        'type': 'निरोगी',
        'status': 'उत्कृष्ट आरोग्य',
        'cure': 'झाड सशक्त आहे. अतिरिक्त फवारण्या टाळाव्यात.'
    }
}

SOYBEAN_CLASSES = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']
SOYBEAN_REMEDIES = {
    'Caterpillar': {
        'title': 'सोयाबीन — लष्करी अळी / पाने खाणारी अळी (Caterpillar)',
        'type': 'रोगग्रस्त',
        'status': 'कीड प्रादुर्भाव',
        'cure': 'इमामेक्टिन बेन्झोएट ५% एस.जी. ४ ग्रॅम किंवा कोराजन ३ मिली प्रति १० लिटर पाण्यात मिसळून फवारावे.'
    },
    'Diabrotica speciosa': {
        'title': 'सोयाबीन — पानांवरील भुंगा/किडे (Leaf Beetle)',
        'type': 'रोगग्रस्त',
        'status': 'कीड प्रादुर्भाव',
        'cure': 'अलिका (थायमेथॉक्सम + लॅम्बडा सायहॅलोथ्रीन) ३ ते ४ मिली प्रति १० लिटर पाण्यात फवारणी करावी.'
    },
    'Healthy': {
        'title': 'सोयाबीन — निरोगी पीक (Healthy Leaf)',
        'type': 'निरोगी',
        'status': 'उत्कृष्ट आरोग्य',
        'cure': 'सोयाबीन पीक निरोगी आहे. नियमित देखरेख ठेवावी.'
    }
}

# ५. हेडर सेक्शन
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌱 कृषी-AI : स्मार्ट पीक संरक्षण प्रणाली</div>
    <div class="hero-subtitle">आविष्कार संशोधन प्रकल्प • AI तंत्रज्ञानाने त्वरित रोग निदान व अचूक कृषी सल्ला</div>
</div>
""", unsafe_allow_html=True)

# ६. इनपुट पॅनल
st.markdown('<div class="control-card">', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    crop_choice = st.selectbox(
        "🌾 १. तपासणीसाठी पीक निवडा:",
        ["☁️ कापूस (Cotton)", "🌱 सोयाबीन (Soybean)", "🥔 बटाटा (Potato)"]
    )

with col2:
    source_option = st.radio(
        "📷 २. फोटो कसा घ्यायचा?",
        ("गॅलरीतून निवडा (Upload)", "कॅमेरा वापरा (Camera)"),
        horizontal=True
    )

uploaded_file = None
if source_option == "गॅलरीतून निवडा (Upload)":
    uploaded_file = st.file_uploader("पानाचा किंवा बोंडाचा सुस्पष्ट फोटो निवडा", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("कॅमेरा समोर धरून फोटो क्लिक करा")

st.markdown('</div>', unsafe_allow_html=True)

# ७. मॉडेल प्रेडिक्शन व रिझल्ट
if uploaded_file is not None and models_ready:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.image(image, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

    with st.spinner("🔍 AI मॉडेल सखोल तपासणी करत आहे..."):
        try:
            if "कापूस" in crop_choice:
                current_model = cotton_model
                classes_list = COTTON_CLASSES
                remedies_dict = COTTON_REMEDIES
                target_size = (224, 224)
                img_resized = image.resize(target_size)
                img_arr = np.array(img_resized, dtype=np.float32) / 255.0

            elif "सोयाबीन" in crop_choice:
                current_model = soybean_model
                classes_list = SOYBEAN_CLASSES
                remedies_dict = SOYBEAN_REMEDIES
                target_size = (224, 224)
                img_resized = image.resize(target_size)
                img_arr = np.array(img_resized, dtype=np.float32) / 255.0

            else:  # बटाटा
                current_model = potato_model
                classes_list = POTATO_CLASSES
                remedies_dict = POTATO_REMEDIES
                target_size = (256, 256)
                img_resized = image.resize(target_size)
                img_arr = np.array(img_resized, dtype=np.float32)

            img_tensor = tf.convert_to_tensor(np.expand_dims(img_arr, axis=0))
            raw_output = current_model(img_tensor, training=False).numpy()[0]

            if np.sum(raw_output) > 1.05 or np.sum(raw_output) < 0.95:
                predictions = tf.nn.softmax(raw_output).numpy()
            else:
                predictions = raw_output

            pred_idx = int(np.argmax(predictions))
            conf = float(predictions[pred_idx]) * 100

            if "बटाटा" in crop_choice and pred_idx != 2 and (conf < 50.0 or abs(predictions[pred_idx] - predictions[2]) < 0.08):
                pred_idx = 2
                conf = float(predictions[2]) * 100

            label = classes_list[pred_idx]
            info = remedies_dict[label]

            # रिझल्ट डिस्प्ले
            if info['type'] == 'निरोगी':
                st.markdown(f"""
                <div class="success-badge">
                    <span style="background:#2e7d32; color:white; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:700;">{info['status']}</span>
                    <h3 style="color:#1b5e20; margin:8px 0 4px 0; font-size:22px;">✅ {info['title']}</h3>
                    <p style="margin:0; color:#388e3c; font-weight:600; font-size:16px;">अचूकता (Confidence): {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-badge">
                    <span style="background:#d32f2f; color:white; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:700;">{info['status']}</span>
                    <h3 style="color:#b71c1c; margin:8px 0 4px 0; font-size:22px;">⚠️ {info['title']}</h3>
                    <p style="margin:0; color:#c62828; font-weight:600; font-size:16px;">अचूकता (Confidence): {conf:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # उपाय बॉक्स
            st.markdown(f"""
            <div class="advisory-box">
                <div class="advisory-title">
                    <span>💡</span> तज्ज्ञ कृषी शिफारसी व उपाययोजना
                </div>
                <p style="margin:0; font-size:15.5px; color:#2c3e50; line-height:1.6;">
                    {info['cure']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # प्रोग्रेस बारसह विश्लेषण
            with st.expander("📊 सर्व घटकांचे सविस्तर विश्लेषण (Probabilities)"):
                for idx, c_name in enumerate(classes_list):
                    prob = float(predictions[idx]) * 100
                    st.write(f"**{c_name}** — {prob:.2f}%")
                    st.progress(int(prob))

        except Exception as pred_err:
            st.error(f"विश्लेषण करताना त्रुटी आली: {pred_err}")
            
