import streamlit as st
import google.generativeai as genai
from PIL import Image

st.set_page_config(
    page_title="Krushi-AI",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;600;700&display=swap');
    * { font-family: 'Mukta', sans-serif; }
    .stApp { background-color: #f4f8f4; }
    .hero-banner {
        background: linear-gradient(135deg, #073b22, #1e874b);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        color: white;
        margin-bottom: 16px;
    }
    .result-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #c8e6c9;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Google AI Studio API Key ethe direct taka:
API_KEY = "TUMCHI_GOOGLE_AI_STUDIO_KEY_ETHE_PASTE_KARA"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

st.markdown("""
<div class="hero-banner">
    <h2>🌱 Krushi-AI : Peek va Rog Nidan</h2>
    <p>Automated Multi-Crop Vision System</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pikacha photo dya (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Nivadlela photo", use_container_width=True)

    fast_img = img.copy()
    fast_img.thumbnail((512, 512))

    with st.spinner("AI nidan karat ahe..."):
        try:
            prompt = """
            Tumhi krushi tajjnya ahat. Ya pikachya photoche nirikshan karun Marathi madhe spashtha uttar dya:
            1. Okhallele Peek konte ahe?
            2. Avayav konta ahe (Paan, Phool, Fal)?
            3. Rog konta ahe kivha peek nirogi ahe ka?
            4. Upay aani aushadhache praman kay asave?
            """
            response = model.generate_content([prompt, fast_img])
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(response.text)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
            
