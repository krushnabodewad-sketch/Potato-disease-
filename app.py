import streamlit as st
import google.generativeai as genai
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="कृषी-AI: स्मार्ट पीक संरक्षण",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Modern Ultra-Clean Emerald Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700;800&family=Poppins:wght@500;600;700&display=swap');
    
    * {
        font-family: 'Mukta', 'Poppins', sans-serif;
    }
    
    .stApp {
        background-color: #f4f8f4;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0b4f30 0%, #167a49 55%, #22a061 100%);
        border-radius: 20px;
        padding: 24px 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(22, 122, 73, 0.22);
        margin-bottom: 20px;
    }
    
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 26px;
        font-weight: 800;
        margin: 10px 0 4px 0;
    }
    
    .hero-banner p {
        color: #d1f2e2 !important;
        font-size: 14px;
        margin: 0;
    }

    .badge-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        color: #ffffff;
        margin-top: 10px;
    }

    /* Card styling */
    .main-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #d9edd9;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
        margin-bottom: 18px;
    }

    .result-box {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
        margin-top: 18px;
        line-height: 1.7;
    }

    [data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 2px solid #e0eee2;
    }
</style>
""", unsafe_allow_html=True)

# 3. Google Gemini Setup
# Ethe tumchi Google AI Studio chi key paste kara:
API_KEY = "TUMCHI_API_KEY_ETHE_PASTE_KARA"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# 4. Inline Modern SVG Logo + Hero Banner
logo_svg = """
<svg width="64" height="64" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="leafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#A7F3D0"/>
      <stop offset="100%" stop-color="#10B981"/>
    </linearGradient>
  </defs>
  <path d="M50 10C25 25 15 50 25 75C35 95 65 95 75 75C85 50 75 25 50 10Z" fill="url(#leafGrad)" opacity="0.9"/>
  <path d="M50 20V85" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
  <path d="M50 40L35 30M50 55L30 48M50 70L38 68" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
  <path d="M50 40L65 30M50 55L70 48M50 70L62 68" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
  <circle cx="35" cy="30" r="4" fill="#FFFFFF"/>
  <circle cx="65" cy="30" r="4" fill="#FFFFFF"/>
  <circle cx="30" cy="48" r="4" fill="#FFFFFF"/>
  <circle cx="70" cy="48" r="4" fill="#FFFFFF"/>
</svg>
"""

st.markdown(f"""
<div class="hero-banner">
    <div>{logo_svg}</div>
    <h1>🌱 कृषी-AI : स्मार्ट पीक व रोग निदान प्रणाली</h1>
    <p>आविष्कार संशोधन प्रकल्प • ऑटो-डिटेक्ट व्हिजन प्रणाली</p>
    <div class="badge-pill">⚡ Gemini Vision AI 1.5 Flash Active</div>
</div>
""", unsafe_allow_html=True)

# 5. File Upload Card
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("#### 📷 पिकाचे पान, फूल किंवा फळाचा फोटो द्या:")
source_option = st.radio("फोटो स्रोत निवडा:", ("गॅलरीतून निवडा (Upload)", "कॅमेरा वापरा (Camera)"), horizontal=True)

uploaded_file = None
if source_option == "गॅलरीतून निवडा (Upload)":
    uploaded_file = st.file_uploader("पानाचा/पिकाचा फोटो निवडा (JPG/PNG)", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("कॅमेरा समोर धरून फोटो क्लिक करा")
st.markdown('</div>', unsafe_allow_html=True)

# 6. Automatic Analysis Pipeline
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

    with st.spinner("🤖 AI आपोआप पीक, अवयव व रोग शोधून कृषी सल्ला तयार करत आहे..."):
        try:
            analysis_prompt = """
            तुम्ही एक उच्च दर्जाचे ज्येष्ठ कृषी शास्त्रज्ञ (Chief Agricultural Scientist & Plant Pathologist AI) आहात.
            दिलेल्या छायाचित्राचे काळजीपूर्वक निरीक्षण करा आणि खालील फॉरमॅटमध्ये शुद्ध व सुटसुटीत मराठीत उत्तर द्या:

            ### 🌾 १. पीक व अवयव ओळख:
            * **ओळखलेले पीक:** [कापूस / सोयाबीन / बटाटा किंवा इतर पिकाचे नाव]
            * **ओळखलेला अवयव:** [पान / फूल / फळ / बोंड / शेंग / खोड]
            * **अचूकता (Confidence):** [उदा. ९५%]

            ---
            ### 🔍 २. आरोग्य स्थिती व रोग निदान:
            * **स्थिती:** [निरोगी / रोगग्रस्त / कीड प्रादुर्भाव]
            * **आढळलेला रोग / कीड:** [रोगाचे नाव किंवा 'पीक पूर्णपणे निरोगी आहे']
            * **लक्षणे:** [दिसून येणाऱ्या लक्षणांचे १-२ ओळीत संक्षिप्त वर्णन]

            ---
            ### 💊 ३. शिफारस केलेले तात्काळ उपाय:
            * **रासायनिक/जैविक औषधे:** [नेमकी औषधांची नावे]
            * **फवारणीचे प्रमाण:** [प्रति लिटर पाण्यात किंवा प्रति १५ लिटर पंपासाठी प्रमाण]

            ---
            ### 🛡️ ४. शेतकरी प्रतिबंधात्मक सल्ला व काळजी:
            * [रोग पसरू नये म्हणून आंतरमशागत, खत व पाणी व्यवस्थापनाची खबरदारी]

            (टीप: छायाचित्र कोणत्याही पिकाचे नसल्यास कृपया 'हा फोटो कोणत्याही शेती पिकाचा दिसत नाही' असा स्पष्ट संदेश द्यावा.)
            """

            response = model.generate_content([analysis_prompt, image])

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(response.text)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"विश्लेषण करताना तांत्रिक त्रुटी आली: {e}")
