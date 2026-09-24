import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# १. पेज कॉन्फिगरेशन
st.set_page_config(
    page_title="कृषी-AI: बहुविध पीक रोग निदान",
    page_icon="🌱",
    layout="centered"
)

# २. आधुनिक व आकर्षक कस्टम CSS स्टायलिंग
st.markdown("""
<style>
    /* मुख्य बॅकग्राउंड व फॉन्ट */
    .main {
        background-color: #f7faf7;
    }
    /* मुख्य हेडर कार्ड */
    .header-box {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: white;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }
    .header-box h1 {
        color: #ffffff !important;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .header-box p {
        color: #c8e6c9 !important;
        font-size: 14px;
        margin: 0;
    }
    /* निकाल दाखवणारे कार्ड्स */
    .result-card-danger {
        background-color: #fff5f5;
        border-left: 6px solid #e53935;
        border-radius: 12px;
        padding: 18px;
        margin-top: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .result-card-success {
        background-color: #f1f8e9;
        border-left: 6px solid #43a047;
        border-radius: 12px;
        padding: 18px;
        margin-top: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .remedy-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 18px;
        margin-top: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# ३. तिन्ही मॉडेल्स लोड करणे
@st.cache_resource
def load_all_models():
    potato_m = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False)
    cotton_m = tf.keras.models.load_model('cotton_model.h5', compile=False)
    soybean_m = tf.keras.models.load_model('soybean_model.h5', compile=False)
    return potato_m, cotton_m, soybean_m

try:
    potato_model, cotton_model, soybean_model = load_all_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"मॉडेल लोड करताना त्रुटी आली: {e}")

# ४. रोग वर्ग व उपाय डेटाबेस
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
POTATO_REMEDIES = {
    'Potato___Early_blight': {
        'title': 'बटाटा — अर्ली ब्लाइट (अगाती करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'मॅन्कोझेब (Mancozeb 75% WP) २ ते २.५ ग्रॅम प्रति लिटर पाण्यात मिसळून संपूर्ण पानांवर फवारावे.'
    },
    'Potato___Late_blight': {
        'title': 'बटाटा — लेट ब्लाइट (उशिरा येणारा करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'मेटॅलॅक्सिल + मॅन्कोझेब (उदा. रिडोमिल गोल्ड) २ ग्रॅम प्रति लिटर पाण्यात मिसळून तात्काळ फवारणी करावी.'
    },
    'Potato___healthy': {
        'title': 'बटाटा — निरोगी पान (Healthy Leaf)',
        'type': 'निरोगी',
        'cure': 'पीक सुदृढ व निरोगी आहे! कोणत्याही रासायनिक फवारणीची आवश्यकता नाही.'
    }
}

COTTON_CLASSES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
COTTON_REMEDIES = {
    'diseased cotton leaf': {
        'title': 'कापूस — रोगग्रस्त पान / बोंड (Bacterial Blight / करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'कॉपर ऑक्सिक्लोराईड (COC) २५ ग्रॅम + स्ट्रेप्टोसायक्लिन १ ते २ ग्रॅम प्रति १० लिटर पाण्यात मिसळून फवारावे.'
    },
    'diseased cotton plant': {
        'title': 'कापूस — रोगग्रस्त झाड (Infected Plant)',
        'type': 'रोगग्रस्त',
        'cure': 'बाधित भागांचे अवशेष गोळा करून नष्ट करावेत आणि मुळांशी ट्रायकोडर्मा किंवा बुरशीनाशकाचे आळवणी करावी.'
    },
    'fresh cotton leaf': {
        'title': 'कापूस — निरोगी पान (Fresh Leaf)',
        'type': 'निरोगी',
        'cure': 'कापसाचे पान निरोगी आहे. संतुलित खत व पाण्याचे योग्य व्यवस्थापन ठेवा.'
    },
    'fresh cotton plant': {
        'title': 'कापूस — निरोगी पीक (Fresh Plant)',
        'type': 'निरोगी',
        'cure': 'कापसाचे झाड पूर्णपणे सुदृढ आहे. अनावश्यक कीटकनाशके वापरू नका.'
    }
}

SOYBEAN_CLASSES = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']
SOYBEAN_REMEDIES = {
    'Caterpillar': {
        'title': 'सोयाबीन — लष्करी अळी / पाने खाणारी अळी (Caterpillar)',
        'type': 'रोगग्रस्त',
        'cure': 'इमामेक्टिन बेन्झोएट ५% एस.जी. ४ ग्रॅम किंवा कोराजन (क्लोरँट्रानिलीप्रोल) ३ मिली प्रति १० लिटर पाण्यात मिसळून फवारावे.'
    },
    'Diabrotica speciosa': {
        'title': 'सोयाबीन — पानांवरील भुंगा/कीड (Leaf Beetle)',
        'type': 'रोगग्रस्त',
        'cure': 'अलिका (थायमेथॉक्सम + लॅम्बडा सायहॅलोथ्रीन) ३ ते ४ मिली प्रति १० लिटर पाण्यात फवारणी करावी.'
    },
    'Healthy': {
        'title': 'सोयाबीन — निरोगी पान (Healthy Leaf)',
        'type': 'निरोगी',
        'cure': 'सोयाबीनचे पान निरोगी आहे. नियमित शेताची पाहणी चालू ठेवा.'
    }
}

# ५. ॲप हेडर
st.markdown("""
<div class="header-box">
    <h1>🌱 कृषी-AI: स्मार्ट पीक रोग निदान</h1>
    <p>आविष्कार संशोधन प्रकल्प — कृत्रिम बुद्धिमत्ता आधारित त्वरित पीक संरक्षण</p>
</div>
""", unsafe_allow_html=True)

# ६. ठळक पीक निवड (Tabs मुळे निवड चुकणार नाही)
crop_tabs = st.tabs(["🥔 बटाटा (Potato)", "☁️ कापूस (Cotton)", "🌱 सोयाबीन (Soybean)"])

def process_and_display(crop_name, model, classes_list, remedies_dict, target_size, normalize_factor):
    st.subheader(f"{crop_name} पीक विश्लेषण")
    
    col1, col2 = st.columns(2)
    with col1:
        source = st.radio(f"{crop_name} - फोटोचा स्रोत:", ("गॅलरी (Upload)", "कॅमेरा (Camera)"), key=f"src_{crop_name}")
    
    uploaded_img = None
    if source == "गॅलरी (Upload)":
        uploaded_img = st.file_uploader(f"{crop_name} पानाचा फोटो निवडा", type=["jpg", "jpeg", "png"], key=f"up_{crop_name}")
    else:
        uploaded_img = st.camera_input(f"{crop_name} पाण्याचा फोटो काढा", key=f"cam_{crop_name}")

    if uploaded_img is not None and models_ready:
        image = Image.open(uploaded_img).convert('RGB')
        st.image(image, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

        with st.spinner("AI मॉडेल विश्लेषण करत आहे..."):
            img_resized = image.resize(target_size)
            img_arr = np.array(img_resized, dtype=np.float32) * normalize_factor
            img_batch = np.expand_dims(img_arr, axis=0)

            raw_pred = model.predict(img_batch)[0]
            if np.sum(raw_pred) > 1.05 or np.sum(raw_pred) < 0.95:
                predictions = tf.nn.softmax(raw_pred).numpy()
            else:
                predictions = raw_pred

            pred_idx = int(np.argmax(predictions))
            conf = float(predictions[pred_idx]) * 100

            # बटाट्यासाठी स्मार्ट थ्रेशोल्ड
            if crop_name == "बटाटा" and pred_idx != 2 and (conf < 50.0 or abs(predictions[pred_idx] - predictions[2]) < 0.08):
                pred_idx = 2
                conf = float(predictions[2]) * 100

            label = classes_list[pred_idx]
            info = remedies_dict[label]

            # रिझल्ट कार्ड
            if info['type'] == 'निरोगी':
                st.markdown(f"""
                <div class="result-card-success">
                    <h3 style="color:#2e7d32; margin:0;">✅ {info['title']}</h3>
                    <p style="margin:5px 0 0 0; color:#555;">अचूकता (Confidence): <b>{conf:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-danger">
                    <h3 style="color:#c62828; margin:0;">⚠️ {info['title']}</h3>
                    <p style="margin:5px 0 0 0; color:#555;">अचूकता (Confidence): <b>{conf:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

            # उपाय कार्ड
            st.markdown(f"""
            <div class="remedy-card">
                <h4 style="margin:0 0 8px 0; color:#1b5e20;">💡 तज्ज्ञ कृषी सल्ला व उपाय:</h4>
                <p style="margin:0; font-size:15px; color:#333; line-height:1.5;">{info['cure']}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📊 सविस्तर घटक विश्लेषण"):
                for idx, c_name in enumerate(classes_list):
                    st.write(f"• **{c_name}**: {float(predictions[idx])*100:.2f}%")

# बटाटा टॅब
with crop_tabs[0]:
    process_and_display("बटाटा", potato_model, POTATO_CLASSES, POTATO_REMEDIES, (256, 256), 1.0)

# कापूस टॅब
with crop_tabs[1]:
    process_and_display("कापूस", cotton_model, COTTON_CLASSES, COTTON_REMEDIES, (224, 224), 1.0/255.0)

# सोयाबीन टॅब
with crop_tabs[2]:
    process_and_display("सोयाबीन", soybean_model, SOYBEAN_CLASSES, SOYBEAN_REMEDIES, (224, 224), 1.0/255.0)
    
