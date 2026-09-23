import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="AI बटाटा रोग ओळख", page_icon="🥔", layout="centered")

@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model('potato_disease_model (1).h5')

try:
    model = load_disease_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error("मॉडेल लोड होऊ शकले नाही. कृपया मॉडेल फाईल तपासा.")

CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

REMEDIES = {
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
        'cure': 'पीक निरोगी आहे. कोणत्याही रासायनिक फवारणीची गरज नाही.'
    }
}

st.title("🥔 कृषी-AI: वनस्पती रोग ओळख प्रणाली")
st.caption("आविष्कार संशोधन स्पर्धा — थेट कॅमेरा किंवा फोटोद्वारे विश्लेषण")

option = st.radio("फोटो कसा निवडायचा?", ("गॅलरीतून निवडा (Upload)", "थेट कॅमेऱ्याने फोटो काढा (Camera)"))

uploaded_file = None
if option == "गॅलरीतून निवडा (Upload)":
    uploaded_file = st.file_uploader("पानाचा फोटो निवडा (JPG / PNG)", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("पानावर कॅमेरा रोखून फोटो काढा")

if uploaded_file is not None and model_ready:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="निवडलेले पान", use_container_width=True)
    
    with st.spinner("AI मॉडेल विश्लेषण करत आहे..."):
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        
        predictions = model.predict(img_array)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx]) * 100
        
        predicted_label = CLASS_NAMES[predicted_idx]
        info = REMEDIES[predicted_label]
        
        st.divider()
        if info['type'] == 'निरोगी':
            st.success(f"### निष्कर्ष: {info['title']}")
        else:
            st.error(f"### आढळलेला रोग: {info['title']}")
            
        st.metric(label="अचूकता (Confidence Rate)", value=f"{confidence:.2f}%")
        st.subheader("💡 शिफारस केलेले कृषी उपाय:")
        st.info(info['cure'])
      
