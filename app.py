import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Krushi-AI: Bahuvidh Peek Rog Nidan", page_icon="🌿", layout="centered")

# Teeni models load karne
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
    st.error(f"Model load kartana samasya aali: {e}")

# Classes aani Upay
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
POTATO_REMEDIES = {
    'Potato___Early_blight': {
        'title': 'Batata - Early Blight (Agati Karpa)',
        'type': 'Roggrast',
        'cure': 'Mancozeb 75% WP 2 te 2.5 gram prati liter panyat mislun fawarave.'
    },
    'Potato___Late_blight': {
        'title': 'Batata - Late Blight (Ushira Yenara Karpa)',
        'type': 'Roggrast',
        'cure': 'Metalaxyl + Mancozeb (Ridomil Gold) 2 gram prati liter panyat fawarave.'
    },
    'Potato___healthy': {
        'title': 'Nirogi Batata Paan (Healthy)',
        'type': 'Nirogi',
        'cure': 'Peek purnapane nirogi ahe! Kahihi fawaranyachi garaj nahi.'
    }
}

COTTON_CLASSES = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
COTTON_REMEDIES = {
    'diseased cotton leaf': {
        'title': 'Kapus - Roggrast Paan / Bond (Bacterial Blight)',
        'type': 'Roggrast',
        'cure': 'Copper Oxychloride (COC) 25g + Streptocycline 1-2g prati 10L panyat fawarave.'
    },
    'diseased cotton plant': {
        'title': 'Kapus - Roggrast Zhaad (Infected Plant)',
        'type': 'Roggrast',
        'cure': 'Badhit zhadanche awashesh nashta karawe aani bursheenashak drenching karawe.'
    },
    'fresh cotton leaf': {
        'title': 'Nirogi Kapus Paan (Fresh Leaf)',
        'type': 'Nirogi',
        'cure': 'Paan purnapane nirogi ahe.'
    },
    'fresh cotton plant': {
        'title': 'Nirogi Kapus Zhaad (Fresh Plant)',
        'type': 'Nirogi',
        'cure': 'Kapus zhaad sudhrudha ahe.'
    }
}

SOYBEAN_CLASSES = ['Caterpillar', 'Diabrotica speciosa', 'Healthy']
SOYBEAN_REMEDIES = {
    'Caterpillar': {
        'title': 'Soyabean - Lashkari Aali / Paane Khanari Aali',
        'type': 'Keed/Roggrast',
        'cure': 'Emamectin Benzoate 5% SG 4g kiva Coragen 3ml prati 10L panyat fawarave.'
    },
    'Diabrotica speciosa': {
        'title': 'Soyabean - Paanvaril Bhunga/Keed (Leaf Beetle)',
        'type': 'Keed/Roggrast',
        'cure': 'Alika (Thiamethoxam + Lambda cyhalothrin) 3-4 ml prati 10L panyat fawarave.'
    },
    'Healthy': {
        'title': 'Nirogi Soyabean Paan (Healthy Leaf)',
        'type': 'Nirogi',
        'cure': 'Soyabean paan purnapane nirogi ahe.'
    }
}

# UI
st.title("🌿 Krushi-AI: Bahuvidh Peek Rog Nidan")
crop_choice = st.selectbox("🌱 Tumche Peek Niwada (Select Crop):", ["Batata (Potato)", "Kapus (Cotton)", "Soyabean (Soybean)"])
source_option = st.radio("Photo kasa niwdaycha?", ("Gallerytun Niwada (Upload)", "Camera ne Photo Kadha (Camera)"))

uploaded_file = None
if source_option == "Gallerytun Niwada (Upload)":
    uploaded_file = st.file_uploader("Paancha photo niwada (JPG / PNG)", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Paana var camera thevun photo kadha")

if uploaded_file is not None and models_ready:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Nivadlele Chhayachitra", use_container_width=True)
    
    with st.spinner("AI Vishleshan karat ahe..."):
        try:
            if crop_choice == "Batata (Potato)":
                target_size = potato_model.input_shape[1:3]
                if None in target_size or len(target_size) != 2:
                    target_size = (256, 256)
                img = image.resize(target_size)
                img_arr = np.array(img, dtype=np.float32)
                # Kahi models /255 require kartat
                img_batch = np.expand_dims(img_arr, axis=0)
                
                try:
                    raw_pred = potato_model.predict(img_batch)[0]
                except Exception:
                    raw_pred = potato_model.predict(img_batch / 255.0)[0]
                    
                classes_list = POTATO_CLASSES
                remedies_dict = POTATO_REMEDIES
                
            elif crop_choice == "Kapus (Cotton)":
                img = image.resize((224, 224))
                img_batch = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
                raw_pred = cotton_model.predict(img_batch)[0]
                classes_list = COTTON_CLASSES
                remedies_dict = COTTON_REMEDIES
                
            else: # Soyabean
                img = image.resize((224, 224))
                img_batch = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
                raw_pred = soybean_model.predict(img_batch)[0]
                classes_list = SOYBEAN_CLASSES
                remedies_dict = SOYBEAN_REMEDIES

            # Normalization / Softmax check
            if np.sum(raw_pred) > 1.05 or np.sum(raw_pred) < 0.95:
                predictions = tf.nn.softmax(raw_pred).numpy()
            else:
                predictions = raw_pred

            pred_idx = int(np.argmax(predictions))
            conf = float(predictions[pred_idx]) * 100
            
            # Potato threshold adjustment
            if crop_choice == "Batata (Potato)" and pred_idx != 2 and (conf < 50.0 or abs(predictions[pred_idx] - predictions[2]) < 0.08):
                pred_idx = 2
                conf = float(predictions[2]) * 100
                
            label = classes_list[pred_idx]
            info = remedies_dict[label]

            st.divider()
            if info['type'] == 'Nirogi':
                st.success(f"### Nishkarsh: {info['title']}")
            else:
                st.error(f"### Aadhallela Rog/Keed: {info['title']}")
            st.metric("Achookta (Confidence)", f"{conf:.2f}%")

            with st.expander("Sarv ghatakanche takkewari vishleshan"):
                for idx, c_name in enumerate(classes_list):
                    st.write(f"• {c_name}: {float(predictions[idx])*100:.2f}%")

            st.subheader("💡 Shefarash kelele Krushi Upay:")
            st.info(info['cure'])

        except Exception as err:
            st.error(f"Prediction chya veles error aala: {err}")
            
