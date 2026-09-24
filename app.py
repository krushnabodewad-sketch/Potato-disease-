cd /home/claude/w && cp /mnt/user-data/outputs/app.py gemini_app.py && cat > top.txt <<'EOF'
# कृषी-AI : स्मार्ट पीक व रोग निदान प्रणाली  (No API key needed)
# Runs fully on your own trained models: potato / cotton / soybean (.h5) + MobileNetV2 plant-part detection.
#
# Files that must sit next to this app.py in the repo:
#   potato_disease_model (1).h5 , cotton_model.h5 , soybean_model.h5
# requirements.txt: use the same one your earlier TensorFlow app used (tensorflow, streamlit, Pillow, numpy).

import re
from html import escape

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
    initial_sidebar_state="collapsed",
)

# =====================================================================
# 2. MODELS LOADING
# =====================================================================
@st.cache_resource
def load_all_models():
    p_m = tf.keras.models.load_model('potato_disease_model (1).h5', compile=False)
    c_m = tf.keras.models.load_model('cotton_model.h5', compile=False)
    s_m = tf.keras.models.load_model('soybean_model.h5', compile=False)
    feat_m = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
    return p_m, c_m, s_m, feat_m

# =====================================================================
# 3. REMEDIES DATABASE
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

CROPS = {
    'potato': {'label': 'बटाटा', 'en': 'Potato', 'emoji': '🥔', 'classes': POTATO_CLASSES, 'remedies': POTATO_REMEDIES},
    'cotton': {'label': 'कापूस', 'en': 'Cotton', 'emoji': '☁', 'classes': COTTON_CLASSES, 'remedies': COTTON_REMEDIES},
    'soybean': {'label': 'सोयाबीन', 'en': 'Soybean', 'emoji': '🫘', 'classes': SOYBEAN_CLASSES, 'remedies': SOYBEAN_REMEDIES},
}

# =====================================================================
# 4. PLANT PART DETECTION
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
# 6. HELPERS
# =====================================================================
def flat(s):
    # Flatten indented HTML so Markdown never renders it as a code block.
    return "".join(line.strip() for line in s.strip().splitlines())


QTY_RE = re.compile(
    r"([०-९0-9]+(?:[.,/][०-९0-9]+)?(?:\s*(?:ते|-|–)\s*[०-९0-9]+(?:[.,][०-९0-9]+)?)?"
    r"\s*(?:ग्रॅम|ग्राम|मि\.ली\.|मिली|मिलि|ml|mL|ML|gm|kg|किलो|लिटर|g|%|टक्के)(?![A-Za-z]))"
)


def bold_dose(text):
    # Escape first, then bold dosages such as २ ग्रॅम, ३ मिली, ५%.
    return QTY_RE.sub(r"<b>\1</b>", escape(text))


EOF
python3 - <<'EOF'
s = open('gemini_app.py', encoding='utf-8').read()
a = s.index('CSS = """'); b = s.index('st.markdown(CSS, unsafe_allow_html=True)')
css = s[a:b]
extra = """
/* ---------- Probability bars ---------- */
.probs{background:#fff; border:1px solid var(--line); border-radius:20px; padding:.9rem 1rem; box-shadow:0 6px 18px rgba(11,79,48,.07); margin-top:.5rem;}
.prob{margin:.55rem 0;}
.prob .lbl{display:flex; justify-content:space-between; gap:.6rem; font-size:.94rem; font-weight:500; margin-bottom:.28rem;}
.prob .lbl b{font-family:'Poppins',sans-serif !important; font-weight:600; font-size:.88rem;}
.prob .pt{height:9px; border-radius:999px; background:#E6F1E9; overflow:hidden;}
.prob .pf{height:100%; border-radius:999px; background:linear-gradient(90deg,#8FD0A6,var(--em-700)); transition:width .8s ease;}
.prob.top .pf{background:linear-gradient(90deg,var(--em-600),var(--em-900));}
.prob.top.bad .pf{background:linear-gradient(90deg,#EF7B6E,var(--red-700));}
.prob.top .lbl{font-weight:700;}
.crop-scan{display:flex; gap:.4rem; flex-wrap:wrap; align-items:center; margin-top:.8rem; padding-top:.7rem; border-top:1px dashed var(--line); font-size:.86rem; color:var(--muted);}
.crop-scan span.c{background:#fff; border:1px solid var(--line); padding:.12rem .55rem; border-radius:999px;}
.crop-scan span.win{background:var(--em-100); color:var(--em-900); font-weight:600;}
[data-testid="stExpander"]{background:#F6FBF7; border:1px solid var(--line) !important; border-radius:14px !important;}
[data-testid="stExpander"] summary p{font-weight:600; color:var(--em-900); font-size:.95rem;}
"""
css = css.replace("</style>", extra + "</style>")
open('css2.txt', 'w', encoding='utf-8').write(css + 'st.markdown(CSS, unsafe_allow_html=True)\n')
a = s.index('LOGO_SVG = """'); b = s.index('api_key = get_api_key()')
open('logo2.txt', 'w', encoding='utf-8').write(s[a:b])
EOF
cat > rest.txt <<'EOF'

# =====================================================================
# 8. HERO + MODEL LOADING
# =====================================================================
hero_slot = st.empty()


def render_hero(state):
    if state is None:
        badge = '<span class="status"><span class="dot"></span>⏳ मॉडेल लोड होत आहेत</span>'
    elif state:
        badge = '<span class="status"><span class="dot"></span>⚡ AI Active</span>'
    else:
        badge = '<span class="status off"><span class="dot"></span>⚠ AI Offline</span>'
    hero_slot.markdown(flat(f"""
    <div class="hero">
      <div class="hero-row">
        <div class="brand">{LOGO_SVG}<div class="bt"><b>कृषी-<i>AI</i></b><small>स्मार्ट पीक व रोग निदान प्रणाली</small></div></div>
        {badge}
      </div>
    </div>
    """), unsafe_allow_html=True)


render_hero(None)

models_ready = False
try:
    potato_model, cotton_model, soybean_model, feature_model = load_all_models()
    models_ready = True
except Exception as e:
    st.markdown(flat("""
    <div class="setup"><b>⚠ AI मॉडेल लोड होऊ शकले नाहीत.</b><br>
    GitHub मध्ये <code>potato_disease_model (1).h5</code>, <code>cotton_model.h5</code> आणि <code>soybean_model.h5</code>
    या फाइल्स <code>app.py</code> सोबत आहेत का ते तपासा आणि <code>requirements.txt</code> मध्ये <code>tensorflow</code> असल्याची खात्री करा.</div>
    """), unsafe_allow_html=True)
    with st.expander("तांत्रिक तपशील"):
        st.code(str(e))

render_hero(models_ready)

# =====================================================================
# 9. INPUT
# =====================================================================
def safe_widget(fn, *args, **kwargs):
    # Call a Streamlit widget; on older Streamlit versions retry without newer kwargs.
    try:
        return fn(*args, **kwargs)
    except (TypeError, AttributeError):
        kwargs.pop("label_visibility", None)
        kwargs.pop("horizontal", None)
        return fn(*args, **kwargs)


def input_card():
    try:
        return st.container(border=True, key="inputcard")
    except TypeError:
        try:
            return st.container(border=True)
        except TypeError:
            return st.container()


source = None
forced_crop = None
with input_card():
    st.markdown(flat("""
    <div class="card-h"><div class="ico">📸</div><b>पिकाचा फोटो द्या</b></div>
    <div class="hint">फोटो स्पष्ट, दिवसाच्या प्रकाशात व जवळून घ्या. पान, फूल किंवा बोंड ठळक दिसू द्या.</div>
    """), unsafe_allow_html=True)

    method = safe_widget(
        st.radio, "फोटो पद्धत",
        ["🖼 गॅलरीतून निवडा (Upload)", "📷 कॅमेऱ्याने फोटो काढा (Camera)"],
        horizontal=True, label_visibility="collapsed",
    )

    if method.startswith("🖼"):
        source = safe_widget(st.file_uploader, "फोटो निवडा", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        source = safe_widget(st.camera_input, "फोटो काढा", label_visibility="collapsed")

    with st.expander("⚙ पीक स्वतः निवडा (ऐच्छिक)"):
        crop_choice = safe_widget(
            st.selectbox, "पीक",
            ["🤖 स्वयंचलित ओळख (शिफारस)", "🥔 बटाटा", "☁ कापूस", "🫘 सोयाबीन"],
            label_visibility="collapsed",
        )
    forced_crop = {"🥔 बटाटा": "potato", "☁ कापूस": "cotton", "🫘 सोयाबीन": "soybean"}.get(crop_choice)

# =====================================================================
# 10. ANALYSIS + RESULTS
# =====================================================================
if source is not None and models_ready:
    try:
        img = Image.open(source).convert("RGB")
    except Exception:
        st.error("हा फोटो उघडता आला नाही. कृपया JPG / PNG फोटो वापरा.")
        st.stop()

    st.markdown('<div class="sec"><div class="bar"></div><h3>२. तुमचा फोटो</h3></div>', unsafe_allow_html=True)
    try:
        st.image(img, use_container_width=True)
    except TypeError:
        st.image(img, use_column_width=True)

    with st.spinner("🔍 AI फोटोचे विश्लेषण करत आहे…"):
        part = get_part_and_features(img)
        crop_key, probs, top_conf = run_ensemble(img, forced_crop)

    crop = CROPS[crop_key]
    classes = crop['classes']
    remedies = crop['remedies']
    idx = int(np.argmax(probs))
    confidence = float(probs[idx]) * 100
    info = remedies[classes[idx]]
    healthy = info['type'] == 'निरोगी'
    tone = "ok" if healthy else "bad"
    icon = "✅" if healthy else "⚠"

    st.markdown('<div class="sec"><div class="bar"></div><h3>३. निदान</h3></div>', unsafe_allow_html=True)
    st.markdown(flat(f"""
    <div class="pills">
      <span class="pill crop">{crop['emoji']} {crop['label']} <small>{crop['en']}</small></span>
      <span class="pill">{escape(part)}</span>
    </div>
    <div class="result {tone}">
      <div class="row">
        <div class="ico-lg">{icon}</div>
        <div class="grow">
          <span class="tag">{icon} {escape(info['type'])} · {escape(info['badge'])}</span>
          <h2>{escape(info['title'])}</h2>
          <div class="en">{'तुमचे पीक सुरक्षित दिसत आहे.' if healthy else 'त्वरित लक्ष देण्याची गरज आहे.'}</div>
        </div>
        <div class="metric"><div class="num">{confidence:.1f}%</div><div class="cap">अचूकता (Confidence)</div></div>
      </div>
      <div class="track"><div class="fill" style="width:{max(confidence, 3):.1f}%"></div></div>
    </div>
    """), unsafe_allow_html=True)

    if confidence < 60:
        st.markdown(
            "<div class='note'>💡 खात्री कमी आहे. जवळून, स्पष्ट व चांगल्या प्रकाशात दुसरा फोटो घेऊन पुन्हा तपासा, "
            "किंवा वर \"पीक स्वतः निवडा\" वापरा.</div>",
            unsafe_allow_html=True,
        )

    # ---- Probability breakdown ----
    st.markdown('<div class="sec"><div class="bar"></div><h3>संभाव्यता तपशील</h3></div>', unsafe_allow_html=True)
    bars = ""
    for i in np.argsort(probs)[::-1]:
        pct = float(probs[i]) * 100
        top = " top" if i == idx else ""
        bad = " bad" if (i == idx and not healthy) else ""
        bars += (
            f'<div class="prob{top}{bad}"><div class="lbl"><span>{escape(remedies[classes[i]]["title"])}</span>'
            f'<b>{pct:.1f}%</b></div><div class="pt"><div class="pf" style="width:{max(pct, 1):.1f}%"></div></div></div>'
        )
    scan = "".join(
        f'<span class="c{" win" if k == crop_key else ""}">{CROPS[k]["emoji"]} {CROPS[k]["label"]} {v * 100:.0f}%</span>'
        for k, v in top_conf.items()
    )
    scan_title = "पीक जुळणी:" if not forced_crop else "पीक (हाताने निवडलेले):"
    st.markdown(flat(f"""
    <div class="probs">{bars}<div class="crop-scan"><span>{scan_title}</span>{scan}</div></div>
    """), unsafe_allow_html=True)

    # ---- Advisory ----
    st.markdown('<div class="sec"><div class="bar"></div><h3>४. सल्ला व उपाययोजना</h3></div>', unsafe_allow_html=True)
    cure_cls = "adv cure okc" if healthy else "adv cure"
    st.markdown(flat(f"""
    <div class="{cure_cls}">
      <div class="head"><div class="ico">{'🌱' if healthy else '💊'}</div><h4>तात्काळ रासायनिक / जैविक उपाय</h4></div>
      <ul class="lst"><li>{bold_dose(info['cure'])}</li></ul>
    </div>
    <div class="adv care">
      <div class="head"><div class="ico">🛡</div><h4>शेतकरी प्रतिबंधात्मक सल्ला व काळजी</h4></div>
      <ul class="lst"><li>{bold_dose(info['prevention'])}</li></ul>
    </div>
    <div class="note">📌 औषधांचे प्रमाण व फवारणी करण्यापूर्वी स्थानिक कृषी अधिकारी / कृषी विज्ञान केंद्राचा सल्ला घ्या.</div>
    """), unsafe_allow_html=True)

elif source is None and models_ready:
    st.markdown(flat("""
    <div class="note" style="text-align:center;">📸 सुरुवात करण्यासाठी वर फोटो निवडा किंवा कॅमेऱ्याने काढा.</div>
    """), unsafe_allow_html=True)

st.markdown('<div class="foot">🌾 कृषी-AI · स्वयंचलित पीक निदान · Avishkar Research Convention</div>', unsafe_allow_html=True)
EOF
python3 - <<'EOF'
def rd(n): return open(n, encoding='utf-8').read()
out = rd('top.txt') + "# =====================================================================\n# 7. STYLES + LOGO\n# =====================================================================\n" + rd('css2.txt') + "\n" + rd('logo2.txt') + rd('rest.txt')
open('/mnt/user-data/outputs/app.py', 'w', encoding='utf-8').write(out)
import ast; ast.parse(out); print('syntax ok', out.count('"""'), 'triple quotes', len(out.splitlines()), 'lines')
EOF
grep -n "api_key\|genai\|API_KEY" /mnt/user-data/outputs/app.py | head
