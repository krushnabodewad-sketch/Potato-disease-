import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="कृषी-AI: बहुविध पीक रोग निदान", page_icon="🌿", layout="centered")

# तिन्ही मॉडेल्स सुरक्षितपणे लोड करणे
@st.cache_resource
def load_models():
    potato_m = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False)
    cotton_m = tf.keras.models.load_model('cotton_model.h5', compile=False)
    soybean_m = tf.keras.models.load_model('soybean_model.h5', compile=False)
    return potato_m, cotton_m, soybean_m

try:
    potato_model, cotton_model, soybean_model = load_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error("मॉडेल फाईल्स लोड करताना अडचण आली. कृपया तिन्ही .h5 फाईल्स रिपॉझिटरीमध्ये असल्याची खात्री करा.")

# १. बटाटा रोग वर्ग व उपाय
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
POTATO_REMEDIES = {
    'Potato___Early_blight': {
        'title': 'बटाटा - अर्ली ब्लाइट (अगाती करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'मॅन्कोझेब (Mancozeb 75% WP) २ ते २.५ ग्रॅम प्रति लिटर पाण्यात मिसळून फवारावे.'
    },
    'Potato___Late_blight': {
        'title': 'बटाटा - लेट ब्लाइट (उशिरा येणारा करपा)',
        'type': 'रोगग्रस्त',
        'cure': 'मेटॅलॅक्सिल + मॅन्कोझेब (उदा. रिडोमिल गोल्ड) २ ग्रॅम प्रति लिटर पाण्यात फवारावे.'
    },
    'Potato___healthy': {
        'title': 'निरोगी बटाटा पान (Healthy Leaf)',
        'type': 'निरोगी',
        'cure': 'पीक पूर्णपणे निरोगी आहे! कोणत्याही रासायनिक फवारणीची गरज नाही.'
    }
}

# २. कापूस रोग वर्ग व उपाय
COTTON_CLASSES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
COTTON_REMEDIES = {
    'diseased cotton leaf': {
        'title': 'कापूस - रोगग्रस्त पान (Bacterial Blight / Leaf Spot)',
        'type': 'रोगग्रस्त',
        'cure': 'कॉपर ऑक्सिक्लोराईड (COC) २५ ग्रॅम + स्ट्रेप्टोसायक्लिन २ ग्रॅम १० लिटर पाण्यात मिसळून फवारणी करावी.'
    },
    'diseased cotton plant': {
        'title': 'कापूस - रोगग्रस्त झाड (Infected Plant)',
        'type': 'रोगग्रस्त',
        'cure': 'बाधित झाडांचे अवशेष नष्ट करावेत व योग्य बुरशीनाशकाची फवारणी करावी.'
    },
    'fresh cotton leaf': {
        'title': 'निरोगी कापूस पान (Fresh Leaf)',
        'type': 'निरोगी',
        'cure': 'कापसाचे पान पूर्णपणे निरोगी आहे. संतुलित खत व्यवस्थापन ठेवा.'
    },
    'fresh cotton plant': {
        'title': 'निरोगी कापूस पीक (Fresh Plant)',
        'type': 'निरोगी',
        'cure': 'पीक सुदृढ आहे. अनावश्यक फवारणी टाळावी.'
    }
}

# ३. सोयाबीन रोग वर्ग व उपाय
SOYBEAN_CLASSES = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']
SOYBEAN_REMEDIES = {
    'Caterpillar': {
        'title': 'सोयाबीन - लष्करी अळी / पाने खाणारी अळी (Caterpillar)',
        'type': 'कीड/रोगग्रस्त',
        'cure': 'इमामेक्टिन बेन्झोएट ५% एस.जी. ४ ग्रॅम किंवा कोराजन ३ मिली प्रति १० लिटर पाण्यात फवारावे.'
    },
    'Diabrotica speciosa': {
        'title': 'सोयाबीन - पानांवरील किडे/भुंगा (Leaf Beetle)',
        'type': 'कीड/रोगग्रस्त',
        'cure': 'अलिका (थायमेथॉक्सम + लॅम्बडा सायहॅलोथ्रीन) ३ ते ४ मिली प्रति १० लिटर पाण्यात फवारावे.'
    },
    'Healthy': {
        'title': 'निरोगी सोयाबीन पान (Healthy Leaf)',
        'type': 'निरोगी',
        'cure': 'सोयाबीनचे पान पूर्णपणे निरोगी आहे! नियमित निरीक्षण ठेवा.'
    }
}

# मुख्य युझर इंटरफेस
st.title("🌿 कृषी-AI: बहुविध पीक रोग निदान प्रणाली")
st.caption("आविष्कार संशोधन प्रकल्प — बटाटा, कापूस व सोयाबीन पीक संरक्षण")

crop_choice = st.selectbox("🌱 तुमचे पीक निवडा (Select Crop):", ["बटाटा (Potato)", "कापूस (Cotton)", "सोयाबीन (Soybean)"])

source_option = st.radio("फोटो कसा निवडायचा?", ("गॅलरीतून निवडा (Upload)", "थेट कॅमेऱ्याने फोटो काढा (Camera)"))

uploaded_file = None
if source_option == "गॅलरीतून निवडा (Upload)":
    uploaded_file = st.file_uploader("पानाचा फोटो निवडा (JPG / PNG)", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("पानावर कॅमेरा रोखून फोटो काढा")

if uploaded_file is not None and models_ready:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)
    
    with st.spinner("AI मॉडेल विश्लेषण करत आहे..."):
        if crop_choice == "बटाटा (Potato)":
            img = image.resize((256, 256))
            img_arr = np.array(img, dtype=np.float32)
            img_batch = np.expand_dims(img_arr, axis=0)
            
            raw_pred = potato_model.predict(img_batch)[0]
            
            # सॉफ्टमॅक्स व नॉर्मलायझेशन हँडलिंग
            if np.sum(raw_pred) > 1.05 or np.sum(raw_pred) < 0.95:
                predictions = tf.nn.softmax(raw_pred).numpy()
            else:
                predictions = raw_pred
                
            pred_idx = int(np.argmax(predictions))
            conf = float(predictions[pred_idx]) * 100
            
            if pred_idx != 2 and (conf < 50.0 or abs(predictions[pred_idx] - predictions[2]) < 0.08):
                pred_idx = 2
                conf = float(predictions[2]) * 100
                
            label = POTATO_CLASSES[pred_idx]
            info = POTATO_REMEDIES[label]
            classes_list = POTATO_CLASSES

        elif crop_choice == "कापूस (Cotton)":
            img = image.resize((224, 224))
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_arr, axis=0)
            
            raw_pred = cotton_model.predict(img_batch)[0]
            if np.sum(raw_pred) > 1.05 or np.sum(raw_pred) < 0.95:
                predictions = tf.nn.softmax(raw_pred).numpy()
            else:
                predictions = raw_pred
                
            pred_idx = int(np.argmax(predictions))
            conf = float(predictions[pred_idx]) * 100
            label = COTTON_CLASSES[pred_idx]
            info = COTTON_REMEDIES[label]
            classes_list = COTTON_CLASSES

        else:  # सोयाबीन (Soybean)
            img = image.resize((224, 224))
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_arr, axis=0)
            
            raw_pred = soybean_model.predict(img_batch)[0]
            if np.sum(raw_pred) > 1.05 or np.sum(raw_pred) < 0.95:
                predictions = tf.nn.softmax(raw_pred).numpy()
            else:
                predictions = raw_pred
                
            pred_idx = int(np.argmax(predictions))
            conf = float(predictions[pred_idx]) * 100
            label = SOYBEAN_CLASSES[pred_idx]
            info = SOYBEAN_REMEDIES[label]
            classes_list = SOYBEAN_CLASSES

        st.divider()
        if info['type'] == 'निरोगी':
            st.success(f"### निष्कर्ष: {info['title']}")
        else:
            st.error(f"### आढळलेला रोग/कीड: {info['title']}")
        st.metric("अचूकता (Confidence)", f"{conf:.2f}%")
        
        with st.expander("सर्व घटकांचे टक्केवारी विश्लेषण"):
            for idx, c_name in enumerate(classes_list):
                st.write(f"• {c_name}: {float(predictions[idx])*100:.2f}%")

        st.subheader("💡 शिफारस केलेले कृषी उपाय:")
        st.info(info['cure'])
        
