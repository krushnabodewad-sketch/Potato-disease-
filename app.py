Act as a world-class Senior UI/UX Designer and Lead Streamlit Frontend Engineer. 

I have a multi-crop disease detection AI web application ("कृषी-AI") built with TensorFlow/Keras and Streamlit, designed for Indian farmers and academic presentation (Avishkar Research Convention). The app automatically diagnoses diseases across Potato, Cotton, and Soybean crops with plant part detection (Leaf, Flower, Fruit/Boll) and provides expert agro-remedies in Marathi.

### UI/UX REDESIGN GOALS:
1. Aesthetic & Modern Agri-Tech Theme:
   - Use modern typography (e.g., Google Fonts 'Mukta', 'Inter', 'Poppins').
   - Premium nature/agri palette: Deep emerald greens (#0B4F30, #167A49), soft mint backgrounds (#F4FAF6), clean white glassmorphism cards with subtle borders (#D8EBD9) and soft drop shadows.
   - Clean badges/pills for plant parts and crops (Flower 🌸, Fruit/Boll 🍏, Leaf 🍃).

2. Component & Layout Structure:
   - Hero Header: Clean banner with app branding, subtitle, and an aesthetic status badge ("⚡ AI Active").
   - Input Section: Sleek upload area with smooth radio selector (Upload vs Camera), drag-and-drop file styling.
   - Result Presentation:
     * Dynamic Alert Cards: Vibrant red-accented card for diseased/pests (with warning icon, severity status, disease title, and confidence score metric), and lush green card for healthy crops.
     * Interactive Probability Breakdown: Polished progress bars / visual metrics for class distribution.
     * Advisory / Remedy Section: Two distinct, beautifully styled cards for (1) "तात्काळ रासायनिक / जैविक उपाययोजना" and (2) "शेत व्यवस्थापन व प्रतिबंधात्मक काळजी".
   - Responsive & Mobile First: Ensure perfect rendering on mobile browsers since farmers access this on smartphones.

3. CRITICAL TECHNICAL CONSTRAINTS (DO NOT BREAK):
   - Keep all core TensorFlow prediction logic, tensor conversions, shape handling, and the ensemble routing untouched to prevent runtime ValueError / Shape mismatch.
   - Maintain the complete Marathi language dictionary and remedies.
   - Return the FULL, ready-to-deploy single-file `app.py` script without cutting any sections.

Here is the current functional Python code:

```python
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Config
st.set_page_config(
    page_title="कृषी-AI: स्वयंचलित पीक व रोग निदान",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Models Loading
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

# 3. Remedies Database
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
POTATO_REMEDIES = {
    'Potato___Early_blight': {
        'title': 'अर्ली ब्लाइट (अगाती करपा)',
        'type': 'रोगग्रस्त',
        'badge': 'बुरशीजन्य संसर्ग',
        'cure': 'मॅन्कोझेब (Mancozeb 75% WP) २ ते २.५ ग्रॅम प्रति लिटर पाण्यात मिसळून संपूर्ण पानांवर फवारावे.',
        'prevention': 'पिकांची फेरपालट करा व रोगट पाने गोळा करून नष्ट करा.'
    },
    'Potato___Late_blight': {
        'title': 'लेट ब्लाइट (उशिरा येणारा करपा)',
        'type': 'रोगग्रस्त',
        'badge': 'गंभीर रोग धोका',
        'cure': 'मेटॅलॅक्सिल + मॅन्कोझेब (रिडोमिल गोल्ड) २ ग्रॅम प्रति लिटर पाण्यात फवारावे.',
        'prevention': 'ढगाळ व दमट हवामानात त्वरित प्रतिबंधात्मक फवारणी करावी.'
    },
    'Potato___healthy': {
        'title': 'निरोगी पीक (Healthy)',
        'type': 'निरोगी',
        'badge': 'उत्कृष्ट आरोग्य',
        'cure': 'पीक पूर्णपणे सुदृढ आहे. कोणत्याही औषधाची गरज नाही.',
        'prevention': 'वेळोवेळी योग्य खत आणि पाणी व्यवस्थापन ठेवा.'
    }
}

COTTON_CLASSES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
COTTON_REMEDIES = {
    'diseased cotton leaf': {
        'title': 'रोगग्रस्त पान / बोंड (Bacterial Blight / करपा)',
        'type': 'रोगग्रस्त',
        'badge': 'जिवाणू संसर्ग',
        'cure': 'कॉपर ऑक्सिक्लोराईड (COC) २५ ग्रॅम + स्ट्रेप्टोसायक्लिन १ ते २ ग्रॅम प्रति १० लिटर पाण्यात फवारावे.',
        'prevention': 'शेतात पाणी साचू देऊ नका आणि गळून पडलेले रोगट भाग नष्ट करा.'
    },
    'diseased cotton plant': {
        'title': 'रोगग्रस्त झाड (Infected Plant)',
        'type': 'रोगग्रस्त',
        'badge': 'संसर्गग्रस्त पीक',
        'cure': 'बाधित झाडे नष्ट करावीत व मुळांशी ट्रायकोडर्मा किंवा बुरशीनाशकाचे आळवणी करावी.',
        'prevention': 'रोगप्रतिकारक वाण वापरा.'
    },
    'fresh cotton leaf': {
        'title': 'निरोगी कापूस पान / बोंड (Fresh)',
        'type': 'निरोगी',
        'badge': 'उत्कृष्ट आरोग्य',
        'cure': 'कापसाचे पीक निरोगी आहे. अनावश्यक फवारणी टाळावी.',
        'prevention': 'रसशोषक किडींचे नियमित निरीक्षण ठेवा.'
    },
    'fresh cotton plant': {
        'title': 'निरोगी कापूस पीक (Fresh Plant)',
        'type': 'निरोगी',
        'badge': 'उत्कृष्ट आरोग्य',
        'cure': 'कापूस झाड सशक्त व निरोगी आहे.',
        'prevention': 'संतुलित खतांचा वापर ठेवा.'
    }
}

SOYBEAN_CLASSES = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']
SOYBEAN_REMEDIES = {
    'Caterpillar': {
        'title': 'लष्करी अळी / पाने खाणारी अळी (Caterpillar)',
        'type': 'रोगग्रस्त',
        'badge': 'कीड प्रादुर्भाव',
        'cure': 'इमामेक्टिन बेन्झोएट ५% एस.जी. ४ ग्रॅम किंवा कोराजन ३ मिली प्रति १० लिटर पाण्यात फवारावे.',
        'prevention': 'शेतात एकरी ३ कामगंध सापळे (Pheromone Traps) लावा.'
    },
    'Diabrotica speciosa': {
        'title': 'पानांवरील भुंगा/किडे (Leaf Beetle)',
        'type': 'रोगग्रस्त',
        'badge': 'कीड प्रादुर्भाव',
        'cure': 'अलिका (थायमेथॉक्सम + लॅम्बडा सायहॅलोथ्रीन) ३ ते ४ मिली प्रति १० लिटर पाण्यात फवारावे.',
        'prevention': 'सुरुवातीला निंबोळी अर्क ५% फवारा.'
    },
    'Healthy': {
        'title': 'निरोगी सोयाबीन पीक (Healthy Leaf)',
        'type': 'निरोगी',
        'badge': 'उत्कृष्ट आरोग्य',
        'cure': 'सोयाबीनचे पीक निरोगी आहे! अतिरिक्त फवारण्या टाळा.',
        'prevention': 'पाण्याचा ताण पडू देऊ नका.'
    }
}

# 4. Plant Part Detection
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

# UI Layout & Auto Routing
# (Please completely revamp the UI layout and CSS below into an ultra-modern, aesthetic frontend)
