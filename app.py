import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =====================================================================
# 1. PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="कृषी-AI: स्वयंचलित पीक व रोग निदान",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# 2. MODELS LOADING  (unchanged)
# =====================================================================
@st.cache_resource
def load_all_models():
    p_m = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False)
    c_m = tf.keras.models.load_model('cotton_model.h5', compile=False)
    s_m = tf.keras.models.load_model('soybean_model.h5', compile=False)
    feat_m = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
    return p_m, c_m, s_m, feat_m

# =====================================================================
# 3. REMEDIES DATABASE  (unchanged)
# =====================================================================
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

# Crop registry (used by the routing + UI layer)
CROPS = {
    'potato': {'label': 'बटाटा', 'emoji': '🥔', 'classes': POTATO_CLASSES, 'remedies': POTATO_REMEDIES},
    'cotton': {'label': 'कापूस', 'emoji': '☁️', 'classes': COTTON_CLASSES, 'remedies': COTTON_REMEDIES},
    'soybean': {'label': 'सोयाबीन', 'emoji': '🫘', 'classes': SOYBEAN_CLASSES, 'remedies': SOYBEAN_REMEDIES},
}

# =====================================================================
# 4. PLANT PART DETECTION  (unchanged)
# =====================================================================
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

# =====================================================================
# 5. ENSEMBLE ROUTING & PREDICTION
#    Every crop model sees the image; the most confident model wins.
#    Input size and scaling are read from each model so shapes never clash.
# =====================================================================
def _has_builtin_rescaling(layer_or_model, depth=0):
    """True if the model already scales pixels itself (Rescaling / Normalization)."""
    if depth > 3:
        return False
    for layer in getattr(layer_or_model, 'layers', [])[:8]:
        if layer.__class__.__name__ in ('Rescaling', 'Normalization'):
            return True
        if hasattr(layer, 'layers') and _has_builtin_rescaling(layer, depth + 1):
            return True
    return False


def _model_input_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    h = shape[1] if len(shape) > 2 and shape[1] else 224
    w = shape[2] if len(shape) > 2 and shape[2] else 224
    return int(h), int(w)


def _prepare_input(img_pil, model):
    h, w = _model_input_size(model)
    arr = np.array(img_pil.convert('RGB').resize((w, h)), dtype=np.float32)
    if not _has_builtin_rescaling(model):
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def _predict_probs(model, img_pil):
    x = _prepare_input(img_pil, model)
    out = model(x, training=False)
    probs = np.array(out.numpy()[0], dtype=np.float64).flatten()
    total = probs.sum()
    if probs.min() < 0 or abs(total - 1.0) > 1e-3:
        e = np.exp(probs - probs.max())
        probs = e / e.sum()
    return probs


def run_ensemble(img_pil, forced_crop=None):
    """Returns the routed crop key, its probabilities and every crop's top confidence."""
    model_map = {'potato': potato_model, 'cotton': cotton_model, 'soybean': soybean_model}
    all_probs, top_conf = {}, {}
    for key, mdl in model_map.items():
        n_classes = len(CROPS[key]['classes'])
        p = _predict_probs(mdl, img_pil)[:n_classes]
        all_probs[key] = p
        top_conf[key] = float(p.max())
    crop = forced_crop if forced_crop else max(top_conf, key=top_conf.get)
    return crop, all_probs[crop], top_conf

# =====================================================================
# 6. UI: STYLES
# =====================================================================
def html(s):
    """Flatten indented HTML so Markdown never turns it into a code block."""
    return "".join(line.strip() for line in s.strip().splitlines())


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;500;600;700;800&family=Poppins:wght@500;600;700&display=swap');

:root{
  --em-900:#0B4F30; --em-700:#167A49; --em-100:#DCF0E3;
  --mint:#F4FAF6; --line:#D8EBD9; --ink:#12281D; --muted:#5B7264;
  --red-700:#B42318; --red-100:#FDECEA; --red-line:#F5C2BD;
  --card:rgba(255,255,255,.82);
}
html, body, [class*="css"], .stApp, .stMarkdown, p, label, button, input{
  font-family:'Mukta','Poppins',sans-serif !important;
}
.stApp{
  background:
    radial-gradient(900px 380px at 100% -5%, #E2F4E8 0%, transparent 60%),
    radial-gradient(700px 320px at -10% 0%, #EAF7EE 0%, transparent 55%),
    var(--mint);
  color:var(--ink);
}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}
.block-container{max-width:760px !important; padding:1.1rem 1rem 3rem !important;}

/* ---------- Hero ---------- */
.hero{
  position:relative; overflow:hidden; border-radius:24px; padding:1.5rem 1.35rem 1.35rem;
  background:linear-gradient(135deg,var(--em-900) 0%,var(--em-700) 100%);
  color:#fff; box-shadow:0 14px 34px rgba(11,79,48,.28);
}
.hero::after{
  content:"🌿"; position:absolute; right:-6px; bottom:-28px; font-size:8.5rem; opacity:.13; transform:rotate(-12deg);
}
.hero-top{display:flex; align-items:center; justify-content:space-between; gap:.6rem; flex-wrap:wrap;}
.brand{font-family:'Poppins','Mukta',sans-serif !important; font-weight:700; font-size:1.05rem; letter-spacing:.2px; opacity:.95;}
.status{
  display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .75rem; border-radius:999px;
  background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35); font-size:.86rem; font-weight:600;
  backdrop-filter:blur(6px);
}
.status .dot{width:8px;height:8px;border-radius:50%;background:#6BFFA6; box-shadow:0 0 0 0 rgba(107,255,166,.7); animation:pulse 2s infinite;}
.status.off .dot{background:#FFB4A9; animation:none;}
@keyframes pulse{70%{box-shadow:0 0 0 8px rgba(107,255,166,0);}100%{box-shadow:0 0 0 0 rgba(107,255,166,0);}}
.hero h1{font-size:1.85rem !important; line-height:1.25; font-weight:800; margin:.9rem 0 .35rem !important; padding:0 !important; color:#fff !important;}
.hero p{font-size:1.02rem; margin:0; max-width:34ch; opacity:.92; line-height:1.5;}
.chips{display:flex; gap:.45rem; flex-wrap:wrap; margin-top:1rem;}
.chip{padding:.22rem .7rem; border-radius:999px; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28); font-size:.88rem; font-weight:500;}

/* ---------- Section titles ---------- */
.sec{display:flex; align-items:center; gap:.55rem; margin:1.6rem 0 .7rem;}
.sec .bar{width:5px; height:22px; border-radius:4px; background:var(--em-700);}
.sec h3{font-size:1.2rem !important; font-weight:700; margin:0 !important; padding:0 !important; color:var(--em-900);}
.hint{color:var(--muted); font-size:.95rem; margin:-.3rem 0 .7rem;}

/* ---------- Radio -> segmented control ---------- */
div[data-testid="stRadio"] > label{display:none !important;}
div[role="radiogroup"]{display:flex !important; gap:.5rem; background:#fff; border:1px solid var(--line); padding:.35rem; border-radius:16px; box-shadow:0 2px 8px rgba(11,79,48,.05);}
div[role="radiogroup"] > label{
  flex:1; justify-content:center; margin:0 !important; padding:.62rem .5rem !important; border-radius:12px; cursor:pointer;
  transition:background .2s, color .2s; text-align:center;
}
div[role="radiogroup"] > label > div:first-child{display:none !important;}
div[role="radiogroup"] > label p{font-weight:600; font-size:1rem; margin:0; color:var(--muted);}
div[role="radiogroup"] > label:has(input:checked){background:var(--em-900);}
div[role="radiogroup"] > label:has(input:checked) p{color:#fff;}

/* ---------- Uploader & camera ---------- */
[data-testid="stFileUploader"] label{display:none !important;}
[data-testid="stFileUploaderDropzone"]{
  background:var(--card) !important; border:2px dashed #9CCFAE !important; border-radius:20px !important; padding:1.6rem 1rem !important;
  transition:border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--em-700) !important; background:#fff !important;}
[data-testid="stFileUploaderDropzone"] button{
  background:var(--em-900) !important; color:#fff !important; border:none !important; border-radius:999px !important; padding:.5rem 1.2rem !important; font-weight:600;
}
[data-testid="stCameraInput"]{border-radius:20px; overflow:hidden; border:1px solid var(--line); background:#fff;}
[data-testid="stCameraInput"] button{background:var(--em-900) !important; color:#fff !important; border-radius:999px !important; border:none !important;}
[data-testid="stCameraInput"] label{display:none !important;}
[data-testid="stImage"] img{border-radius:18px; border:1px solid var(--line); box-shadow:0 8px 22px rgba(11,79,48,.12);}

/* ---------- Expander ---------- */
[data-testid="stExpander"]{background:var(--card); border:1px solid var(--line) !important; border-radius:16px !important;}
[data-testid="stExpander"] summary p{font-weight:600; color:var(--em-900);}

/* ---------- Pills ---------- */
.pills{display:flex; gap:.5rem; flex-wrap:wrap; margin:.9rem 0 .2rem;}
.pill{
  display:inline-flex; align-items:center; gap:.4rem; padding:.4rem .9rem; border-radius:999px;
  background:#fff; border:1px solid var(--line); font-weight:600; font-size:.98rem; color:var(--em-900);
  box-shadow:0 2px 6px rgba(11,79,48,.06);
}
.pill small{font-weight:500; color:var(--muted); font-size:.82rem;}
.pill.crop{background:var(--em-100); border-color:#B9DDC5;}

/* ---------- Result card ---------- */
.result{
  border-radius:22px; padding:1.25rem 1.2rem; margin-top:.9rem; border:1px solid var(--line);
  box-shadow:0 12px 30px rgba(11,79,48,.12); position:relative; overflow:hidden;
}
.result.ok{background:linear-gradient(160deg,#FFFFFF 0%,#E4F5EA 100%); border-left:8px solid var(--em-700);}
.result.bad{background:linear-gradient(160deg,#FFFFFF 0%,var(--red-100) 100%); border-color:var(--red-line); border-left:8px solid var(--red-700);}
.result .row{display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;}
.status-tag{display:inline-flex; align-items:center; gap:.4rem; font-weight:700; font-size:.95rem; padding:.28rem .75rem; border-radius:999px;}
.ok .status-tag{background:var(--em-100); color:var(--em-900);}
.bad .status-tag{background:#fff; color:var(--red-700); border:1px solid var(--red-line);}
.result h2{font-size:1.45rem !important; line-height:1.3; font-weight:800; margin:.75rem 0 .2rem !important; padding:0 !important;}
.ok h2{color:var(--em-900) !important;} .bad h2{color:var(--red-700) !important;}
.result .sub{color:var(--muted); font-size:.98rem;}
.metric{text-align:right; flex-shrink:0;}
.metric .num{font-family:'Poppins',sans-serif !important; font-weight:700; font-size:2rem; line-height:1;}
.ok .num{color:var(--em-700);} .bad .num{color:var(--red-700);}
.metric .cap{font-size:.82rem; color:var(--muted); margin-top:.25rem;}

/* ---------- Probability bars ---------- */
.probs{background:var(--card); border:1px solid var(--line); border-radius:20px; padding:1rem 1.1rem; box-shadow:0 6px 18px rgba(11,79,48,.07);}
.prob{margin:.55rem 0;}
.prob .lbl{display:flex; justify-content:space-between; gap:.6rem; font-size:.96rem; font-weight:500; margin-bottom:.28rem;}
.prob .lbl b{font-family:'Poppins',sans-serif !important; font-weight:600; font-size:.9rem;}
.prob .track{height:10px; border-radius:999px; background:#E6F1E9; overflow:hidden;}
.prob .fill{height:100%; border-radius:999px; background:linear-gradient(90deg,#8FD0A6,var(--em-700)); transition:width .8s ease;}
.prob.top .fill{background:linear-gradient(90deg,var(--em-700),var(--em-900));}
.prob.top.bad .fill{background:linear-gradient(90deg,#EF7B6E,var(--red-700));}
.prob.top .lbl{font-weight:700;}
.crop-scan{display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.9rem; padding-top:.8rem; border-top:1px dashed var(--line); font-size:.88rem; color:var(--muted);}
.crop-scan span{background:#fff; border:1px solid var(--line); padding:.15rem .6rem; border-radius:999px;}
.crop-scan span.win{background:var(--em-100); color:var(--em-900); font-weight:600;}

/* ---------- Advisory ---------- */
.adv{border-radius:20px; padding:1.1rem 1.15rem; margin-top:.8rem; background:var(--card); border:1px solid var(--line); box-shadow:0 6px 18px rgba(11,79,48,.07);}
.adv.cure{border-top:5px solid var(--red-700);}
.adv.care{border-top:5px solid var(--em-700);}
.adv.cure.okc{border-top-color:var(--em-700);}
.adv .head{display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem;}
.adv .ico{width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.2rem; background:var(--red-100);}
.adv.care .ico, .adv.cure.okc .ico{background:var(--em-100);}
.adv h4{margin:0 !important; padding:0 !important; font-size:1.08rem !important; font-weight:700; color:var(--ink);}
.adv p{margin:0; font-size:1.05rem; line-height:1.65; color:#26392E;}

.note{margin-top:.9rem; padding:.7rem .9rem; border-radius:14px; background:#FFF7E6; border:1px solid #F3DDA6; color:#7A5A00; font-size:.95rem;}
.foot{text-align:center; color:var(--muted); font-size:.85rem; margin-top:2rem;}

@media (max-width:480px){
  .hero{padding:1.25rem 1.05rem;} .hero h1{font-size:1.55rem !important;}
  .result .row{flex-direction:column; gap:.6rem;} .metric{text-align:left;}
}
@media (prefers-reduced-motion:reduce){ *{animation:none !important; transition:none !important;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =====================================================================
# 7. UI: LOAD MODELS + HERO
# =====================================================================
try:
    potato_model, cotton_model, soybean_model, feature_model = load_all_models()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"मॉडेल लोड करताना त्रुटी आली: {e}")

status_html = (
    '<span class="status"><span class="dot"></span>⚡ AI Active</span>'
    if models_ready else
    '<span class="status off"><span class="dot"></span>⚠️ AI Offline</span>'
)

st.markdown(html(f"""
<div class="hero">
  <div class="hero-top">
    <span class="brand">🌿 कृषी-AI</span>
    {status_html}
  </div>
  <h1>पिकाचा फोटो द्या, रोगाचे निदान मिळवा</h1>
  <p>बटाटा, कापूस आणि सोयाबीन पिकांचे रोग व किडी ओळखून मराठीत उपाय सुचवतो.</p>
  <div class="chips">
    <span class="chip">🥔 बटाटा</span><span class="chip">☁️ कापूस</span><span class="chip">🫘 सोयाबीन</span>
    <span class="chip">🍃 पान</span><span class="chip">🌸 फूल</span><span class="chip">🍏 फळ / बोंड</span>
  </div>
</div>
"""), unsafe_allow_html=True)

# =====================================================================
# 8. UI: INPUT
# =====================================================================
st.markdown(html("""
<div class="sec"><div class="bar"></div><h3>१. पिकाचा फोटो द्या</h3></div>
<div class="hint">फोटो स्पष्ट व दिवसाच्या प्रकाशात घ्या. पान, फूल किंवा बोंड जवळून दिसू द्या.</div>
"""), unsafe_allow_html=True)

input_method = st.radio(
    "फोटो पद्धत",
    ["📁 फोटो अपलोड करा", "📷 कॅमेऱ्याने काढा"],
    horizontal=True,
    label_visibility="collapsed",
)

def safe_widget(fn, *args, **kwargs):
    """Call a Streamlit widget; on older Streamlit ver
