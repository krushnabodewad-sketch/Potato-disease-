import streamlit as st
import google.generativeai as genai
from PIL import Image

# १. पेज संरचना
st.set_page_config(
    page_title="कृषी-AI : स्मार्ट पीक संरक्षण",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# २. कॉम्पॅक्ट आणि प्रीमियम ॲग्री-टेक CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700;800&family=Poppins:wght@500;600;700&display=swap');
    * { font-family: 'Mukta', sans-serif; }
    .stApp { background-color: #f4f8f4; }

    .hero-banner {
        background: linear-gradient(135deg, #073b22 0%, #0f6639 55%, #1e874b 100%);
        border-radius: 18px;
        padding: 16px 18px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 20px rgba(15, 102, 57, 0.2);
        margin-bottom: 16px;
    }
    .hero-banner h1 { color: #ffffff !important; font-size: 22px; font-weight: 800; margin: 6px 0 2px 0; }
    .hero-banner p { color: #d1f2e2 !important; font-size: 13px; margin: 0; }

    .badge-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2px 12px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: 600;
        color: #ffffff;
        margin-top: 6px;
    }

    .main-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 16px 18px;
        border: 1px solid #d9edd9;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
    }

    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
        margin-top: 14px;
        line-height: 1.65;
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        border: 1px solid #dceede;
    }
</style>
""", unsafe_allow_html=True)

# ३. API Key Secrets मधून मिळवणे
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ कृपया Streamlit Settings -> Secrets मध्ये 'GEMINI_API_KEY' जोडा.")
    st.stop()

api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# ॲक्टिव्ह मॉडेल शोधणारे ऑटोमॅटिक फंक्शन
@st.cache_resource
def get_supported_model():
    preferred = [
        "gemini-1.5-flash",
        "models/gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro-vision"
    ]
    try:
        available = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        # उपलब्ध मॉडेलमधून मॅच करणे
        for p in preferred:
            for a in available:
                if p in a:
                    return genai.GenerativeModel(a)
        # fallback
        return genai.GenerativeModel(available[0])
    except Exception:
        return genai.GenerativeModel('gemini-1.5-flash')

gemini_model = get_supported_model()

# ४. लोगो व हेडर बॅनर
logo_svg = """
<svg width="44" height="44" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="leafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#34D399"/>
      <stop offset="100%" stop-color="#059669"/>
    </linearGradient>
  </defs>
  <path d="M50 12C28 26 18 50 28 74C37 92 63 92 72 74C82 50 72 26 50 12Z" fill="url(#leafGrad)"/>
  <path d="M50 24V80" stroke="#FFFFFF" stroke-width="3.5" stroke-linecap="round"/>
  <path d="M50 42L36 32M50 56L32 50M50 70L38 66" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M50 42L64 32M50 56L68 50M50 70L62 66" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="36" cy="32" r="3.5" fill="#FFFFFF"/>
  <circle cx="64" cy="32" r="3.5" fill="#FFFFFF"/>
  <circle cx="32" cy="50" r="3.5" fill="#FFFFFF"/>
  <circle cx="68" cy="50" r="3.5" fill="#FFFFFF"/>
</svg>
"""

st.markdown(f"""
<div class="hero-banner">
    <div>{logo_svg}</div>
    <h1>🌱 कृषी-AI : स्मार्ट पीक व रोग निदान प्रणाली</h1>
    <p>आविष्कार संशोधन प्रकल्प • ऑटो-डिटेक्ट व्हिजन प्रणाली</p>
    <div class="badge-pill">⚡ Gemini Fast Vision Active</div>
</div>
""", unsafe_allow_html=True)

# ५. इनपुट फॉर्म
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("##### 📷 पिकाचे पान, फूल किंवा फळ/बोंडाचा फोटो द्या:")
source_option = st.radio("फोटो कसा निवडायचा?", ("गॅलरीतून निवडा (Upload)", "कॅमेरा वापरा (Camera)"), horizontal=True)

uploaded_file = None
if source_option == "गॅलरीतून निवडा (Upload)":
    uploaded_file = st.file_uploader("छायाचित्र निवडा (JPG / PNG)", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("कॅमेरा समोर धरून फोटो काढा")
st.markdown('</div>', unsafe_allow_html=True)

# ६. १ ते २ सेकंदात जलद स्वयंचलित विश्लेषण
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="विश्लेषणासाठी निवडलेले छायाचित्र", use_container_width=True)

    fast_image = image.copy()
    fast_image.thumbnail((512, 512))

    with st.spinner("⚡ AI सेकंदात सखोल परीक्षण करत आहे..."):
        try:
            analysis_prompt = """
            तुम्ही कृषी शास्त्रज्ञ आहात. या फोटोचे काळजीपूर्वक निरीक्षण करून खालील फॉरमॅटमध्ये थेट व संक्षिप्त मराठीत उत्तर द्या:

            ### 🌾 १. पीक व अवयव ओळख:
            * **ओळखलेले पीक:** [सोयाबीन / कापूस / बटाटा किंवा इतर पिकाचे अचूक नाव]
            * **ओळखलेला अवयव:** [पान / फूल / फळ / बोंड / शेंग]
            * **अचूकता (Confidence):** [उदा. ९५%]

            ---
            ### 🔍 २. रोग निदान:
            * **आरोग्य स्थिती:** [निरोगी / रोगग्रस्त / कीड प्रादुर्भाव]
            * **आढळलेला रोग किंवा कीड:** [रोगाचे नाव किंवा 'पीक पूर्णपणे निरोगी आहे']
            * **दिसून येणारी लक्षणे:** [१-२ ओळीत संक्षिप्त वर्णन]

            ---
            ### 💊 ३. तात्काळ कृषी उपाय:
            * **औषध व फवारणी प्रमाण:** [नेमके नाव व लिटर पाण्याचे प्रमाण]

            ---
            ### 🛡️ ४. शेतकरी प्रतिबंधात्मक सल्ला:
            * [१-२ ओळीत शेत व्यवस्थापनाची खबरदारी]

            (टीप: छायाचित्र कोणत्याही पिकाचे नसल्यास कृपया 'हा फोटो कोणत्याही शेती पिकाचा दिसत नाही' असा स्पष्ट संदेश द्यावा.)
            """

            response = gemini_model.generate_content([analysis_prompt, fast_image])

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(response.text)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"विश्लेषण करताना त्रुटी आली: {e}")
            
