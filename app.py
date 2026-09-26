import streamlit as st
import json
import io
import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit.components.v1 as components
from google import genai

# ==========================================
# 1. PAGE CONFIG & GEMINI SETUP
# ==========================================
st.set_page_config(page_title="कृषी-AI : Smart Agro Diagnostics", page_icon="🌿", layout="wide")

gemini_key = st.secrets.get("GEMINI_API_KEY", None)
gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None

FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

# ==========================================
# 2. SESSION STATE
# ==========================================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def reset_sample():
    st.session_state.uploader_key += 1

# ==========================================
# 3. GLOBAL STYLING
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&family=Mukta:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', 'Mukta', sans-serif; }
.stApp { background: #F8FAFC; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; max-width: 1100px; }
.k-hero { background: linear-gradient(135deg, #064E3B, #059669); border-radius: 18px; padding: 1.4rem; color: #fff; margin-bottom: 1rem; }
.k-card { background: #fff; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.04); }
.res-card { background: #fff; border-radius: 14px; padding: 1.2rem; border: 1px solid #E2E8F0; margin-top: 0.5rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="k-hero">
    <div style="font-size:11px;font-weight:800;color:#A7F3D0;letter-spacing:1px;">AVISHKAR 2026</div>
    <h2 style="margin:2px 0 0 0;font-size:1.6rem;font-weight:800;">🌿 कृषी-AI : सर्वसमावेशक पीक व रोग निदान प्रणाली</h2>
    <div style="font-size:0.88rem;color:#D1FAE5;margin-top:4px;">Cotton · Soybean · Potato | All Diseases Engine with Botanical Anatomy</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. WORKSPACE LAYOUT
# ==========================================
col_l, col_r = st.columns([1, 1.2], gap="large")

with col_l:
    st.markdown('<div class="k-card"><b>⚙️ नियंत्रण पॅनेल (Control Panel)</b>', unsafe_allow_html=True)
    crop_mode = st.selectbox("🌾 पीक निवडा:", ("🤖 ऑटो-डिटेक्ट (सर्व पिके व रोग)", "☁️ कापूस (Cotton)", "🌱 सोयाबीन (Soybean)", "🥔 बटाटा (Potato)"))
    input_mode = st.radio("माध्यम:", ("गॅलरी (Upload)", "कॅमेरा (Camera)"), horizontal=True)
    
    up_key = f"up_{st.session_state.uploader_key}"
    if input_mode == "गॅलरी (Upload)":
        uploaded_file = st.file_uploader("पानाचा फोटो निवडा:", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed", key="g_" + up_key)
    else:
        uploaded_file = st.camera_input("फोटो काढा:", label_visibility="collapsed", key="c_" + up_key)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<div class="k-card"><b>🍃 पान पूर्वावलोकन (Leaf Preview)</b>', unsafe_allow_html=True)
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    if uploaded_file is None:
        st.info("📡 **निदान टर्मिनल सज्ज आहे.**\n\nडाव्या पॅनेलमधून पानाचा फोटो अपलोड करा किंवा कॅमेऱ्याने काढा.")

# ==========================================
# 5. ALL DISEASES DIAGNOSTIC ENGINE
# ==========================================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    # इमेज बाइट्स फॉरमॅट तयार करणे
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    expert_prompt = f"""
    तुम्ही भारतीय कृषी संशोधन परिषद (ICAR) आणि कृषी विद्यापीठाचे वरिष्ठ वनस्पती रोग शास्त्रज्ञ (Plant Pathologist) आहात.
    दिलेल्या पानाचे, देठाचे आणि संरचनेचे (Leaf Anatomy - Lobes, Veins, Color, Lesions) सखोल शास्त्रीय परीक्षण करा.

    वापरकर्त्याने निवडलेला पर्याय: {crop_mode}

    पिकांची शरीररचना आणि रोगांची तपासणी सूची:
    १. कापूस (Cotton):
       - पानाला ३ ते ५ टोकदार हस्तसदृश खंड (Palmate Lobes), बोंड/कळी किंवा कडांवर तांबूसपणा.
       - रोग: दहिया रोग (Grey Mildew), जिवाणू करपा (Bacterial Blight/Black Arm), आल्टरनेरिया करपा, लाल्या रोग (Leaf Reddening), मर रोग (Wilt), किंवा निरोगी पान.

    २. सोयाबीन (Soybean):
       - एका देठावर तीन अंडाकृती उपपाने (Trifoliolate leaflets).
       - रोग: पिवळा मोझॅक (Yellow Mosaic), तांबेरा (Rust), चारकोल रॉट (Charcoal Rot), अळी/भुंगा प्रादुर्भाव (Caterpillar/Beetle), रायझोक्टोनिया ब्लाईट, किंवा निरोगी पान.

    ३. बटाटा (Potato):
       - समोरासमोर लंबगोलाकार सुरकुतलेली पाने, ठळक शिरा जाळे (Pinnate reticulate).
       - रोग: लवकर येणारा करपा (Early Blight), उशिरा येणारा करपा (Late Blight), काळा खवल्या (Black Scurf), रिंग रॉट/मर, मोझॅक, किंवा निरोगी पान.

    खालील रचनेनुसार शुद्ध, स्पष्ट आणि शेतकरी बांधवांना समजेल अशा मराठीत अचूक अहवाल द्या:

    🌾 **पीक नाव:** [कापूस / सोयाबीन / बटाटा]
    🩺 **रोग / कीड निदान:** [रोगाचे नाव व तीव्रता (मध्यम / तीव्र / निरोगी)]
    🔍 **पानावरील मुख्य लक्षणे:** [दिसणारे बदल व डागांचे स्वरूप]
    🧪 **रासायनिक फवारणी उपचार:** [शिफारसीत औषधाचे नाव आणि १५ लिटर पंपासाठी अचूक प्रमाण]
    🌿 **सेंद्रिय / जैविक उपाय:** [जैविक घटक, काढा किंवा प्रतिबंधक उपाय]
    📅 **वेळापत्रक:** दिवस ८ आणि दिवस १५ चा फवारणी सल्ला.
    """

    with col_r:
        with st.spinner("AI सर्व रोगांची आणि पानांच्या संरचनेची तपासणी करत आहे..."):
            diag_output = None
            if gemini_client:
                for model_candidate in FALLBACK_MODELS:
                    try:
                        res = gemini_client.models.generate_content(
                            model=model_candidate,
                            contents=[
                                expert_prompt,
                                genai.types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                            ]
                        )
                        if res and res.text:
                            diag_output = res.text
                            break
                    except Exception:
                        continue

            if diag_output:
                st.markdown('<div class="k-card"><b>🩺 अचूक पीक व रोग निदान अहवाल</b></div>', unsafe_allow_html=True)
                st.markdown(f"<div class='res-card'>{diag_output}</div>", unsafe_allow_html=True)

                # ऑडिओ सल्ला ऐकण्यासाठी
                audio_text = diag_output[:220].replace("*", "").replace("\n", " ")
                a_js = json.dumps(audio_text)
                a_html = f'<script>function spk(){{window.speechSynthesis.cancel();var m=new SpeechSynthesisUtterance({a_js});m.lang="mr-IN";window.speechSynthesis.speak(m);}}</script><button onclick="spk()" style="width:100%;margin-top:12px;background:linear-gradient(135deg,#059669,#10b981);color:#fff;border:none;padding:12px;border-radius:12px;font-weight:700;cursor:pointer;">🔊 ऑडिओ सल्ला ऐका (Listen Audio)</button>'
                components.html(a_html, height=54)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        label="⬇️ अहवाल डाऊनलोड करा (Download)",
                        data=diag_output.encode("utf-8-sig"),
                        file_name="krushi_ai_full_report.txt",
                        mime="text/plain; charset=utf-8",
                        use_container_width=True
                    )
                with d2:
                    st.button("🔄 नवीन फोटो तपासा", on_click=reset_sample, use_container_width=True)
            else:
                st.error("⚠️ AI सर्व्हर व्यस्त आहे. कृपया १ मिनिटाने पुन्हा प्रयत्न करा किंवा Secrets मधील API Key तपासा.")
                
