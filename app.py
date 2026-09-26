import streamlit as st
from PIL import Image
import google.generativeai as genai
import json
import streamlit.components.v1 as components

# ==========================================
# 1. PAGE SETUP & STYLING
# ==========================================
st.set_page_config(page_title="कृषी-AI : स्मार्ट पीक डॉक्टर", page_icon="🌿", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Mukta:wght@500;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Mukta', sans-serif; }
.stApp { background: #F8FAFC; }
#MainMenu, footer, header { visibility: hidden; }
.k-hero { background: linear-gradient(135deg, #064E3B, #059669); border-radius: 16px; padding: 1.2rem; color: #fff; text-align: center; margin-bottom: 1.2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.06); }
.res-card { background: #fff; border-radius: 14px; padding: 1.2rem; border: 1px solid #E2E8F0; margin-top: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.03); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="k-hero">
    <div style="font-size:12px;font-weight:700;color:#A7F3D0;letter-spacing:1px;">AVISHKAR 2026</div>
    <h2 style="margin:2px 0 0 0;font-size:1.6rem;font-weight:800;">🌿 कृषी-AI : १००% ऑटोमॅटिक पीक डॉक्टर</h2>
    <p style="margin:4px 0 0 0;font-size:0.9rem;color:#D1FAE5;">कोणतेही पीक निवडावे लागणार नाही · AI थेट फोटोवरून ओळखेल</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. GEMINI API SETUP
# ==========================================
gemini_key = st.secrets.get("GEMINI_API_KEY", None)

if not gemini_key:
    gemini_key = st.sidebar.text_input("Gemini API Key टाका:", type="password")

if not gemini_key:
    st.warning("⚠️ कृपया ॲप चालवण्यासाठी तुमची मोफत Gemini API Key टाका.")
    st.stop()

genai.configure(api_key=gemini_key)

# ==========================================
# 3. PHOTO INPUT
# ==========================================
input_mode = st.radio("माध्यम निवडा:", ("गॅलरीतून निवडा (Upload)", "कॅमेऱ्याने काढा (Camera)"), horizontal=True)

if input_mode == "गॅलरीतून निवडा (Upload)":
    uploaded_file = st.file_uploader("पानाचा किंवा पिकाचा फोटो टाका:", type=["jpg", "jpeg", "png", "webp"])
else:
    uploaded_file = st.camera_input("पानाचा फोटो काढा:")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="अपलोड केलेला फोटो", use_container_width=True)

    if st.button("🔍 AI द्वारे थेट रोग व पीक निदान करा", type="primary", use_container_width=True):
        with st.spinner("AI पानाचे सखोल परीक्षण करत आहे..."):
            prompt = """
            तुम्ही एक उच्चपदस्थ कृषी तज्ज्ञ (Plant Pathologist) आहात.
            या फोटोचे बारकाईने निरीक्षण करून खालील मुद्यांनुसार थेट व स्पष्ट मराठीत उत्तर द्या:

            १. अचूक पीक (Crop Name): (उदा. सोयाबीन, कापूस, बटाटा, तूर इत्यादी. जर वनस्पतीचे पान नसेल तर थेट सांगा.)
            २. सद्यस्थिती (Condition): (निरोगी आहे की रोगट?)
            ३. रोग किंवा कीड (Disease/Pest): (रोगाचे नाव, निरोगी असल्यास 'निरोगी पान')
            ४. अचूक लक्षणे (Symptoms): (पानावर काय बदल दिसत आहेत?)
            ५. रासायनिक उपाय (Chemical Spray): (औषधाचे नाव आणि प्रति १५ लिटर पंपाचे प्रमाण)
            ६. जैविक/सेंद्रिय उपाय (Organic Solution): (घरगुती किंवा जैविक उपाय)
            ७. पुढील फवारणी सल्ला: (८ व्या आणि १५ व्या दिवशी काय करावे?)

            कोणतीही खोटी किंवा संदिग्ध माहिती देऊ नका. जर सोयाबीन असेल तर सोयाबीनचेच निदान करा.
            """

            # Fallback Chain (Quota एरर येऊ नये म्हणून)
            MODELS = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
            output_text = None

            for m in MODELS:
                try:
                    model = genai.GenerativeModel(m)
                    res = model.generate_content([prompt, img])
                    if res and res.text:
                        output_text = res.text
                        break
                except Exception:
                    continue

            if output_text:
                st.markdown("<div class='res-card'>", unsafe_allow_html=True)
                st.markdown(output_text)
                st.markdown("</div>", unsafe_allow_html=True)

                # ऑडिओ सल्ला
                audio_snippet = output_text[:200].replace("*", "").replace("\n", " ")
                a_js = json.dumps(audio_snippet)
                a_html = f'<script>function spk(){{window.speechSynthesis.cancel();var m=new SpeechSynthesisUtterance({a_js});m.lang="mr-IN";window.speechSynthesis.speak(m);}}</script><button onclick="spk()" style="width:100%;margin-top:12px;background:linear-gradient(135deg,#059669,#10b981);color:#fff;border:none;padding:12px;border-radius:12px;font-weight:700;cursor:pointer;">🔊 ऑडिओ सल्ला ऐका (Listen Audio)</button>'
                components.html(a_html, height=54)
            else:
                st.error("⚠️ AI कोटा संपला आहे किंवा सर्व्हर व्यस्त आहे. कृपया २ मिनिटांनी पुन्हा प्रयत्न करा.")
                
