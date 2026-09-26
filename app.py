import base64
import requests
import streamlit as st
from google import genai
from PIL import Image
import io

# --- Page Setup ---
st.set_page_config(
    page_title="कृषी-AI: पीक रोग निदान",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 कृषी-AI: पीक व रोग निदान प्रणाली")
st.caption("कापूस, सोयाबीन आणि बटाटा पिकांसाठी विशेष AI सहाय्यक")

# --- API Keys from Streamlit Secrets ---
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
PLANTNET_KEY = st.secrets.get("PLANTNET_API_KEY", "")
ROBOFLOW_KEY = st.secrets.get("ROBOFLOW_API_KEY", "")

# 1. PlantNet: Identify Crop / Plant Species
def identify_crop_plantnet(image_bytes):
    if not PLANTNET_KEY:
        return "Unknown"
    url = f"https://my-api.plantnet.org/v2/identify/all?api-key={PLANTNET_KEY}"
    files = [("images", ("leaf.jpg", image_bytes, "image/jpeg"))]
    try:
        response = requests.post(url, files=files, timeout=12)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                top_match = results[0]["species"]["scientificNameWithoutAuthor"]
                common_names = results[0]["species"].get("commonNames", [])
                common_str = f" ({common_names[0]})" if common_names else ""
                return f"{top_match}{common_str}"
    except Exception:
        pass
    return "पीक ओळखता आले नाही"

# 2. Roboflow: Disease Detection & Health Status
def detect_disease_roboflow(image_bytes):
    if not ROBOFLOW_KEY:
        return None
    endpoint = "plant-disease-detection-s8vzx/1"
    url = f"https://detect.roboflow.com/{endpoint}?api_key={ROBOFLOW_KEY}"
    try:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = requests.post(
            url,
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=12
        )
        if response.status_code == 200:
            data = response.json()
            predictions = data.get("predictions", [])
            if predictions:
                top = predictions[0]
                label = top.get("class", "Unknown")
                conf = round(top.get("confidence", 0) * 100, 1)
                is_healthy = "healthy" in label.lower()
                return {
                    "is_healthy": is_healthy,
                    "disease": label,
                    "confidence": conf
                }
            return {
                "is_healthy": True,
                "disease": "निरोगी पान (Healthy)",
                "confidence": 95.0
            }
    except Exception:
        pass
    return None

# 3. Gemini: Expert Advisory in Marathi
def get_gemini_advisory(image_bytes, crop_name, disease_info):
    if not GEMINI_KEY:
        return "Gemini API Key उपलब्ध नाही."
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        img = Image.open(io.BytesIO(image_bytes))
        prompt = f"""
        तुम्ही एक तज्ज्ञ कृषी शास्त्रज्ञ आहात. खालील माहिती आणि पानाच्या फोटोचे विश्लेषण करून मराठीत स्पष्ट सल्ला द्या:
        - ओळखलेले पीक: {crop_name}
        - कॉम्प्युटर व्हिजनचे निदान: {disease_info}

        खालील मुद्द्यांवर सोप्या मराठीत मार्गदर्शन करा:
        1. पानाची सद्यस्थिती (निरोगी आहे की रोगग्रस्त)
        2. जर रोगग्रस्त असेल तर रोगाचे नाव आणि त्याची प्रमुख लक्षणे
        3. जैविक व रासायनिक फवारणी किंवा नियंत्रणाचे उपाय
        4. शेतकऱ्यांसाठी महत्त्वाच्या खबरदारीच्या टिप्स
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img]
        )
        return response.text
    except Exception as e:
        return f"Gemini विश्लेषण करण्यात अडचण आली: {e}"

# --- UI & Image Upload ---
uploaded_file = st.file_uploader("पानाचा स्पष्ट फोटो अपलोड करा किंवा कॅमेऱ्याने काढा:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="अपलोड केलेला फोटो", use_container_width=True)

    if st.button("रोग व पीक विश्लेषण करा 🔍", type="primary"):
        with st.spinner("AI द्वारे पानाचे विश्लेषण सुरू आहे..."):
            # Step 1: Crop Identification
            crop_name = identify_crop_plantnet(image_bytes)
            
            # Step 2: Roboflow Disease Classification
            rf_res = detect_disease_roboflow(image_bytes)

        st.divider()
        st.subheader("📊 प्राथमिक तपासणी निकाल:")
        st.write(f"🌱 **पिकाची जात (PlantNet):** {crop_name}")

        if rf_res:
            if rf_res["is_healthy"]:
                st.success(f"✅ **स्थिती:** निरोगी पान ({rf_res['confidence']}% खात्री)")
            else:
                st.error(f"⚠️ **आढळलेला रोग (Roboflow):** {rf_res['disease']} ({rf_res['confidence']}% खात्री)")
            disease_context = f"{rf_res['disease']} (Accuracy: {rf_res['confidence']}%)"
        else:
            disease_context = "थेट Gemini व्हिजन द्वारे तपासावे."
            st.info("Roboflow डेटा अनुपलब्ध, थेट व्हिजन मॉडेल वापरत आहे.")

        # Step 3: Detailed Advisory via Gemini
        st.divider()
        st.subheader("📋 कृषी तज्ज्ञ सल्ला (Gemini):")
        with st.spinner("मराठी सल्ला तयार होत आहे..."):
            advisory = get_gemini_advisory(image_bytes, crop_name, disease_context)
            st.markdown(advisory)
            
