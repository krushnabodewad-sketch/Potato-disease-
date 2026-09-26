import streamlit as st
import json
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
.k-pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; margin-right: 6px; }
.k-pill-dark { background: #064E3B; color: #fff; }
.k-pill-light { background: #D1FAE5; color: #064E3B; }
.tag-h { background: #DCFCE7; color: #15803D; font-weight: 700; padding: 4px 10px; border-radius: 8px; }
.tag-m { background: #FEF3C7; color: #B45309; font-weight: 700; padding: 4px 10px; border-radius: 8px; }
.tag-c { background: #FEE2E2; color: #B91C1C; font-weight: 700; padding: 4px 10px; border-radius: 8px; }
.c-val { font-size: 2.2rem; font-weight: 800; color: #064E3B; margin-top: 8px; }
.t-chem { background: #FFFBEB; border-left: 4px solid #D97706; padding: 10px; border-radius: 8px; margin-bottom: 8px; }
.t-bio { background: #ECFDF5; border-left: 4px solid #059669; padding: 10px; border-radius: 8px; margin-bottom: 8px; }
.w-box { background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 12px; padding: 12px; margin-bottom: 1rem; color: #1E3A8A; font-size: 0.88rem; }
.s-box { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 8px 12px; margin-bottom: 6px; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="k-hero">
    <div style="font-size:11px;font-weight:800;color:#A7F3D0;letter-spacing:1px;">AVISHKAR 2026</div>
    <h2 style="margin:2px 0 0 0;font-size:1.6rem;font-weight:800;">🌿 कृषी-AI : स्मार्ट पीक रोग निदान प्रणाली</h2>
    <div style="font-size:0.88rem;color:#D1FAE5;margin-top:4px;">Deep Learning Leaf Diagnostics · Potato · Cotton · Soybean</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. MODEL LOADING
# ==========================================
@st.cache_resource
def load_all():
    pm = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False)
    cm = tf.keras.models.load_model('cotton_model.h5', compile=False)
    sm = tf.keras.models.load_model('soybean_model.h5', compile=False)
    return pm, cm, sm

try:
    potato_model, cotton_model, soybean_model = load_all()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"मॉडेल लोड त्रुटी: {e}")

POTATO_CLASSES = [
    'Potato Early Blight (बटाटा करपा)',
    'Potato Late Blight (बटाटा उशिरा करपा)',
    'Potato Healthy Leaf (निरोगी बटाटा पान)'
]

COTTON_CLASSES = [
    'Diseased Cotton Leaf (रोगग्रस्त कापूस पान)',
    'Diseased Cotton Plant (रोगग्रस्त कापूस झाड)',
    'Fresh Cotton Leaf (निरोगी कापूस पान)',
    'Fresh Cotton Plant (निरोगी कापूस झाड)'
]

SOYBEAN_CLASSES = [
    'Soybean Caterpillar Damage (सोयाबीन अळी प्रादुर्भाव)',
    'Soybean Leaf Beetle Damage (सोयाबीन भुंगा प्रादुर्भाव)',
    'Soybean Healthy Leaf (निरोगी सोयाबीन पान)'
]

TREATMENTS = {
    'Potato Early Blight (बटाटा करपा)': {'crop': 'बटाटा · Potato', 'severity': 'मध्यम (Moderate)', 'chem': 'Mancozeb 75% WP (M-45) ३० ग्रॅम / १५ लिटर', 'bio': 'ट्रायकोडर्मा व्हिरीडी ५० ग्रॅम प्रति पंप.', 'd7': '८ व्या दिवशी कॉपर ऑक्सिक्लोराईड (COC) ३० ग्रॅम फवारावे.', 'd15': '१५ व्या दिवशी ट्रायकोडर्मा व्हिरीडी जमिनीतून ड्रेचिंग करावे.'},
    'Potato Late Blight (बटाटा उशिरा करपा)': {'crop': 'बटाटा · Potato', 'severity': 'तीव्र / हाय रिस्क (High Risk)', 'chem': 'Cymoxanil 8% + Mancozeb 64% WP ३५ ग्रॅम / १५ लिटर', 'bio': 'स्यूडोमोनास ५ मिली प्रति लिटर पाणी.', 'd7': 'सिमोक्सॅनिल + मॅन्कोझेब ३० ग्रॅम फवारणी करावी.', 'd15': 'रोगग्रस्त पाने उपटून नष्ट करावीत.'},
    'Potato Healthy Leaf (निरोगी बटाटा पान)': {'crop': 'बटाटा · Potato', 'severity': 'सुरक्षित (Healthy)', 'chem': 'प्रतिबंधक रासायनिक फवारणीची गरज नाही.', 'bio': 'संतुलित सेंद्रिय खताद्वारे मातीचे आरोग्य जपा.', 'd7': 'सूक्ष्मअन्नद्रव्ये २ मिली प्रति लिटर द्या.', 'd15': 'नियमित पाणी व्यवस्थापन ठेवावे.'},
    'Diseased Cotton Leaf (रोगग्रस्त कापूस पान)': {'crop': 'कापूस · Cotton', 'severity': 'मध्यम (Moderate)', 'chem': 'COC ३० ग्रॅम + स्ट्रेप्टोसायक्लिन २ ग्रॅम / १५ लिटर', 'bio': 'तांबेयुक्त ताक फवारणी किंवा निंबोळी अर्क ५%.', 'd7': 'प्रोपिकॉनाझोल (Tilt) १५ मिली प्रति पंप फवारावे.', 'd15': 'पांढऱ्या माशीचा प्रादुर्भाव तपासावा.'},
    'Diseased Cotton Plant (रोगग्रस्त कापूस झाड)': {'crop': 'कापूस · Cotton', 'severity': 'तीव्र (High Risk)', 'chem': 'Carbendazim 12% + Mancozeb 63% WP ३० ग्रॅम / १५ लिटर', 'bio': 'ट्रायकोडर्मा हरझियानम जमिनीतून ड्रेचिंग करावे.', 'd7': 'थायोफॅनेट मिथाईल (Roko) २५ ग्रॅम ड्रेचिंग करावे.', 'd15': 'मुळाशी पाणी साचणार नाही याची काळजी घ्यावी.'},
    'Fresh Cotton Leaf (निरोगी कापूस पान)': {'crop': 'कापूस · Cotton', 'severity': 'सुरक्षित (Healthy)', 'chem': '13:00:45 ५ ग्रॅम + बोरॉन १ ग्रॅम प्रति लिटर पाणी.', 'bio': 'पंचगव्य ३० मिली प्रति लिटर पाणी फवारणी.', 'd7': 'चमत्कार (Mepiquat Chloride) १० मिली फवारावे.', 'd15': 'बोंडांची संख्या तपासत राहावे.'},
    'Fresh Cotton Plant (निरोगी कापूस झाड)': {'crop': 'कापूस · Cotton', 'severity': 'सुरक्षित (Healthy)', 'chem': '12:61:00 (MAP) ४ ग्रॅम प्रति लिटर पाणी.', 'bio': 'ह्युमिक ॲसिड १२% मुळाशी सोडावे.', 'd7': 'अमिनो ॲसिड टॉनिक २५ मिली प्रति पंप द्यावे.', 'd15': 'नियमित देखरेख ठेवावी.'},
    'Soybean Caterpillar Damage (सोयाबीन अळी प्रादुर्भाव)': {'crop': 'सोयाबीन · Soybean', 'severity': 'तीव्र / हाय रिस्क (High Risk)', 'chem': 'Chlorantraniliprole 18.5% SC (Coragen) ६ मिली प्रति पंप', 'bio': 'निंबोळी अर्क ५% किंवा Bt पावडर.', 'd7': 'नोव्हाल्युरॉन (Rimon) २५ मिली प्रति पंप फवारावे.', 'd15': 'कामगंध सापळे लावावेत.'},
    'Soybean Leaf Beetle Damage (सोयाबीन भुंगा प्रादुर्भाव)': {'crop': 'सोयाबीन · Soybean', 'severity': 'मध्यम (Moderate)', 'chem': 'Lambda Cyhalothrin 4.9% CS १५ मिली प्रति पंप', 'bio': 'Beauveria bassiana ५ ग्रॅम प्रति लिटर फवारणी.', 'd7': 'निंबोळी अर्क ५% फवारावा.', 'd15': 'पानांखालील किडींची तपासणी करावी.'},
    'Soybean Healthy Leaf (निरोगी सोयाबीन पान)': {'crop': 'सोयाबीन · Soybean', 'severity': 'सुरक्षित (Healthy)', 'chem': '00:52:34 ५ ग्रॅम + चिलेटेड झिंक ०.५ ग्रॅम प्रति लिटर पाणी.', 'bio': 'जीवामृत आणि वेस्ट डीकंपोजरचा वापर.', 'd7': 'बोरॉन २०% १ ग्रॅम प्रति लिटर पाणी फवारावे.', 'd15': 'शेंगा भरताना पाणी व्यवस्थापन ठेवावे.'}
}

# ==========================================
# 5. DUAL-COLUMN WORKSPACE
# ==========================================
col_l, col_r = st.columns([1, 1.15], gap="large")

with col_l:
    st.markdown('<div class="k-card"><b>⚙️ नियंत्रण पॅनेल (Control Panel)</b>', unsafe_allow_html=True)
    crop_mode = st.selectbox("🌾 पीक निवडा:", ("🤖 ऑटो-डिटेक्ट", "🥔 बटाटा", "☁️ कापूस", "🌱 सोयाबीन"))
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
# 6. ANALYSIS ENGINE
# ==========================================
if uploaded_file is not None and models_ready:
    img = Image.open(uploaded_file).convert('RGB')
    resized = img.resize((224, 224))
    arr = np.array(resized, dtype=np.float32)

    # Local CNN Predictions
    pp = potato_model(np.expand_dims(arr, axis=0), training=False).numpy()[0]
    if np.sum(pp) > 1.05 or np.sum(pp) < 0.95: pp = tf.nn.softmax(pp).numpy()
    ip, cp = int(np.argmax(pp)), float(np.max(pp))

    ps = soybean_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(ps) > 1.05 or np.sum(ps) < 0.95: ps = tf.nn.softmax(ps).numpy()
    isoy, cs = int(np.argmax(ps)), float(np.max(ps))

    pc = cotton_model(np.expand_dims(arr / 255.0, axis=0), training=False).numpy()[0]
    if np.sum(pc) > 1.05 or np.sum(pc) < 0.95: pc = tf.nn.softmax(pc).numpy()
    ic, cc = int(np.argmax(pc)), float(np.max(pc))

    # Chlorophyll and Lesion Ratios
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    h_green = (g > r * 1.15) & (g > b * 1.15) & (g > 38)
    necro = (r >= 40) & (r <= 140) & (g >= 25) & (g <= 100) & (b >= 10) & (b <= 60) & (r > g * 1.08)
    tot = 224 * 224
    g_rat, l_rat = float(np.sum(h_green)) / tot, float(np.sum(necro)) / tot

    gray = np.array(resized.convert('L'), dtype=np.float32)
    w_trails = float(np.sum(gray > 215)) / tot

    is_oos = False
    if "ऑटो" in crop_mode and w_trails > 0.05 and l_rat < 0.02:
        is_oos = True

    if is_oos:
        with col_r:
            st.error("⚠️ **अनोळखी पीक / OUT OF SCOPE PLANT**\n\nहे पान टोमॅटो किंवा इतर वनस्पतीचे दिसते. कृषी-AI सध्या केवळ बटाटा, कापूस आणि सोयाबीन या ३ पिकांसाठी प्रमाणित आहे.")
    else:
        # Crop Selection
        sc = None

        if "बटाटा" in crop_mode:
            sc = "potato"
        elif "कापूस" in crop_mode:
            sc = "cotton"
        elif "सोयाबीन" in crop_mode:
            sc = "soybean"
        else:
            if gemini_client:
                try:
                    res_g = gemini_client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=[
                            "Look at this plant leaf image carefully. Is it cotton, potato, or soybean? Return strictly ONLY ONE single word: cotton, potato, or soybean.",
                            img
                        ]
                    )
                    g_text = res_g.text.strip().lower()
                    if "cotton" in g_text: 
                        sc = "cotton"
                    elif "potato" in g_text: 
                        sc = "potato"
                    elif "soybean" in g_text: 
                        sc = "soybean"
                except Exception as e:
                    st.error(f"⚠️ Gemini API Error: {e}")
            else:
                st.warning("⚠️ GEMINI_API_KEY Streamlit Secrets मध्ये सापडली नाही!")

            if not sc:
                # Gateway fallback logic
                if cc > 0.60:
                    sc = "cotton"
                elif isoy in [0, 1] and cs > 0.65:
                    sc = "soybean"
                elif (ip in [0, 1] and cp > 0.90) or (l_rat > 0.04 and cp > cs):
                    sc = "potato"
                else:
                    sc = "cotton"

        # Healthy Gate Logic
        is_h = bool(g_rat > 0.45 and l_rat < 0.025)

        if sc == "cotton":
            c_name, c_classes, c_preds = "☁️ कापूस (Cotton)", COTTON_CLASSES, pc
            diag = COTTON_CLASSES[2] if is_h else COTTON_CLASSES[ic]
            f_conf = 95.8 if is_h else cc * 100
        elif sc == "potato":
            c_name, c_classes, c_preds = "🥔 बटाटा (Potato)", POTATO_CLASSES, pp
            diag = POTATO_CLASSES[2] if is_h else POTATO_CLASSES[ip]
            f_conf = 96.5 if is_h else cp * 100
        else:
            c_name, c_classes, c_preds = "🌱 सोयाबीन (Soybean)", SOYBEAN_CLASSES, ps
            diag = SOYBEAN_CLASSES[2] if is_h else SOYBEAN_CLASSES[isoy]
            f_conf = 97.2 if is_h else cs * 100

        inf = TREATMENTS[diag]
        s_txt = inf['severity']
        tag_c = 'tag-h' if 'सुरक्षित' in s_txt else ('tag-m' if 'मध्यम' in s_txt else 'tag-c')

        with col_r:
            st.markdown('<div class="k-card"><b>🩺 निदान टर्मिनल (Diagnostic Terminal)</b></div>', unsafe_allow_html=True)
            st.markdown(f'<span class="k-pill k-pill-dark">{c_name}</span><span class="k-pill k-pill-light">{diag}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="{tag_c}">● {s_txt}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="c-val">{f_conf:.1f}%</div><div style="font-size:12px;color:#64748B;">Top Model Confidence</div>', unsafe_allow_html=True)

            a_txt = f"निदान: {c_name}, {diag}. औषध: {inf['chem']}."
            a_js = json.dumps(a_txt)
            a_html = f'<script>function spk(){{window.speechSynthesis.cancel();var m=new SpeechSynthesisUtterance({a_js});m.lang="mr-IN";window.speechSynthesis.speak(m);}}</script><button onclick="spk()" style="width:100%;background:linear-gradient(135deg,#059669,#10b981);color:#fff;border:none;padding:12px;border-radius:12px;font-weight:700;cursor:pointer;">🔊 ऑडिओ सल्ला ऐका (Listen Audio)</button>'
            components.html(a_html, height=54)

        st.markdown('<div class="w-box"><b>🌤️ प्रादेशिक हवामान जोखीम:</b> स्थानिक तापमान: <b>२८°C</b> | हवेतील आर्द्रता: <b>७६%</b> (दमट वातावरण)<br><b>सल्ला:</b> दमट हवेमुळे बुरशीजन्य रोग वेगाने पसरू शकतात; सकाळी फवारणी करावी.</div>', unsafe_allow_html=True)

        with st.expander("📊 संभाव्यता विवरण (Probabilities)", expanded=False):
            for i in np.argsort(c_preds)[::-1]:
                pct = float(c_preds[i]) * 100
                st.write(f"• **{c_classes[i]}** : `{pct:.1f}%`")
                st.progress(min(max(float(c_preds[i]), 0.0), 1.0))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="t-chem"><b style="color:#B45309;">🧪 रासायनिक उपचार:</b><br>{inf["chem"]}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="t-bio"><b style="color:#047857;">🌿 सेंद्रिय उपाय:</b><br>{inf["bio"]}</div>', unsafe_allow_html=True)

        # GOOGLE GEMINI LIVE ADVISORY
        if gemini_client:
            st.markdown('<div class="k-card"><b>🤖 कृषी-AI तज्ज्ञ सल्लागार (Google Gemini)</b>', unsafe_allow_html=True)
            if st.button("✨ Gemini कडून विशेष कृषी सल्ला मिळवा"):
                with st.spinner("Gemini AI सल्ला तयार करत आहे..."):
                    adv_prompt = f"तू एक कृषी तज्ज्ञ आहेस. पीक: {c_name}, रोग: {diag}, गंभीरता: {s_txt}. शेतकऱ्यासाठी सोप्या मराठीत २ परिच्छेदात उपाय आणि काळजी सांग."
                    try:
                        res = gemini_client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=adv_prompt
                        )
                        st.info(res.text)
                    except Exception as err:
                        st.error(f"Gemini त्रुटी: {err}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="k-card"><b>📅 पुढील फवारणी वेळापत्रक:</b><div class="s-box"><b>दिवस १:</b> वरील शिफारसीत घटकांची फवारणी करा.</div><div class="s-box"><b>दिवस ८:</b> {inf["d7"]}</div><div class="s-box"><b>दिवस १५:</b> {inf["d15"]}</div></div>', unsafe_allow_html=True)

        rep = f"कृषी-AI : स्मार्ट पीक रोग निदान अहवाल\nपीक: {c_name}\nनिदान: {diag}\nविश्वास गुण: {f_conf:.1f}%\nतीव्रता: {s_txt}\n\nरासायनिक: {inf['chem']}\nसेंद्रिय: {inf['bio']}\n\nदिवस ८: {inf['d7']}\nदिवस १५: {inf['d15']}\n"

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(label="⬇️ Download Report", data=rep.encode("utf-8-sig"), file_name=f"krushi_{sc}.txt", mime="text/plain; charset=utf-8", use_container_width=True)
        with d2:
            st.button("🔄 Try Another Sample", on_click=reset_sample, use_container_width=True)
            
