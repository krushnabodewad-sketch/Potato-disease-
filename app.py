"""
कृषी-AI : स्मार्ट पीक व रोग निदान प्रणाली
Gemini Vision powered crop / plant-part / disease detection with Marathi advisory.

Run:      streamlit run app_gemini.py
Install:  pip install streamlit google-generativeai Pillow

API KEY (choose one):
  1) Paste it into API_KEY below (quick local testing only), or
  2) Streamlit secrets (recommended). Create .streamlit/secrets.toml:
         GEMINI_API_KEY = "your-key-here"
     On Streamlit Cloud: App -> Settings -> Secrets -> paste the same line.
  3) Environment variable GEMINI_API_KEY.
"""

import os
import re
import hashlib
from io import BytesIO
from html import escape

import streamlit as st
from PIL import Image, ImageOps

try:
    import google.generativeai as genai
    GENAI_OK = True
    GENAI_IMPORT_ERROR = ""
except Exception as _e:  # library missing / broken
    genai = None
    GENAI_OK = False
    GENAI_IMPORT_ERROR = str(_e)

# =====================================================================
# 1. CONFIG
# =====================================================================
API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"

# Tried in this order. "gemini-1.5-flash" has been retired by Google, so newer
# Flash models are tried first and it is kept only as the last fallback.
MODEL_CANDIDATES = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

MAX_IMAGE_SIDE = 1280

st.set_page_config(
    page_title="कृषी-AI: स्मार्ट पीक व रोग निदान प्रणाली",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def get_api_key():
    try:
        v = st.secrets.get("GEMINI_API_KEY", "")
        if v:
            return str(v).strip()
    except Exception:
        pass
    v = os.environ.get("GEMINI_API_KEY", "").strip()
    if v:
        return v
    if API_KEY and not API_KEY.startswith("PASTE_YOUR"):
        return API_KEY.strip()
    return ""


# =====================================================================
# 2. GEMINI PROMPT (forces a strict, parseable Markdown template)
# =====================================================================
PROMPT = """
You are a senior Indian agronomist and plant pathologist. Analyse the attached photo
for a farmer in Maharashtra, India. Detect the crop (Cotton, Soybean, Potato or any other crop),
the plant part, the health status, and any disease / pest / nutrient problem.

RULES
- If the photo does not clearly show a plant or crop, set IS_PLANT: NO.
- If the photo is blurry or you are unsure, use STATUS: UNCERTAIN, give a low CONFIDENCE and say so. Never guess.
- Write every *_MR field and every bullet in simple, farmer-friendly MARATHI (Devanagari). Product names and units may stay in English.
- CURE section: 3 to 5 bullets. Give products/active ingredients commonly registered in India with EXACT dosages
  "per litre of water" AND "per 15-litre pump". Include at least one biological / organic option
  (e.g. Trichoderma, Neem seed kernel extract 5%, Beauveria, Pseudomonas). If the crop is healthy, say no spraying is needed
  and give 1 to 2 optional routine tips.
- PREVENTION section: 4 to 6 bullets on cultural practices, field hygiene, irrigation, crop rotation and monitoring.
- Do not invent product names or dosages. End the CURE section with one bullet telling the farmer to read the label
  and confirm with the local Krishi Vigyan Kendra / agriculture officer.
- Reply with ONLY the template below. No introduction, no code fences, no extra text.

TEMPLATE
IS_PLANT: YES or NO
CROP_MR: crop name in Marathi
CROP_EN: crop name in English
PART: one of LEAF | FLOWER | FRUIT | STEM | WHOLE_PLANT | OTHER
STATUS: one of HEALTHY | DISEASED | PEST | NUTRIENT | UNCERTAIN
DIAGNOSIS_MR: disease / pest / condition name in Marathi (add the English name in brackets)
DIAGNOSIS_EN: disease / pest / condition name in English
CONFIDENCE: integer from 0 to 100
SEVERITY: one of NONE | LOW | MEDIUM | HIGH
SYMPTOMS_MR: one or two Marathi sentences describing what is visible in the photo
NOT_PLANT_MESSAGE: Marathi message (fill only when IS_PLANT is NO, otherwise write -)

## CURE
- bullet
## PREVENTION
- bullet
""".strip()

FIELDS = ["IS_PLANT", "CROP_MR", "CROP_EN", "PART", "STATUS", "DIAGNOSIS_MR",
          "DIAGNOSIS_EN", "CONFIDENCE", "SEVERITY", "SYMPTOMS_MR", "NOT_PLANT_MESSAGE"]


# =====================================================================
# 3. IMAGE PREP + GEMINI CALL + PARSING
# =====================================================================
def prepare_image(raw_bytes):
    img = Image.open(BytesIO(raw_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return img, buf.getvalue()


@st.cache_data(show_spinner=False, ttl=3600)
def analyze_image(img_bytes, api_key):
    """Returns (markdown_text, model_used). Cached per image so reruns cost no API calls."""
    genai.configure(api_key=api_key)
    img = Image.open(BytesIO(img_bytes))
    last_err = None
    for name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(
                name,
                generation_config={"temperature": 0.2, "max_output_tokens": 2048},
            )
            resp = model.generate_content([PROMPT, img])
            try:
                text = resp.text
            except ValueError:
                raise RuntimeError("BLOCKED")
            return text, name
        except Exception as e:
            last_err = e
            m = str(e).lower()
            if "404" in m or "not found" in m or "not supported" in m or "no longer available" in m:
                continue  # try the next model name
            raise
    raise last_err if last_err else RuntimeError("No model available")


def friendly_error(e):
    m = str(e).lower()
    if "blocked" in m:
        return "हा फोटो सुरक्षितता कारणास्तव तपासता आला नाही. कृपया दुसरा स्पष्ट फोटो वापरा."
    if "api key" in m or "api_key" in m or "permission" in m or "401" in m or "403" in m:
        return "API Key चुकीची किंवा अवैध आहे. कृपया Key तपासा."
    if "429" in m or "quota" in m or "rate" in m or "exhausted" in m:
        return "आजची मोफत मर्यादा (quota) संपली आहे. थोड्या वेळाने पुन्हा प्रयत्न करा."
    if "timeout" in m or "deadline" in m or "connect" in m or "unavailable" in m or "503" in m:
        return "इंटरनेट किंवा सर्व्हरची अडचण आहे. कृपया पुन्हा प्रयत्न करा."
    if "404" in m or "not found" in m:
        return "निवडलेले Gemini मॉडेल उपलब्ध नाही. कोडमधील MODEL_CANDIDATES अद्ययावत करा."
    return "विश्लेषण करताना अडचण आली. कृपया पुन्हा प्रयत्न करा."


def parse_response(text):
    text = re.sub(r"```[a-zA-Z]*", "", text).strip()
    data = {}
    for f in FIELDS:
        m = re.search(rf"^[\s>*_\-]*{f}[\s*_]*:[\s*_]*(.+?)\s*$", text, re.M | re.I)
        data[f] = m.group(1).strip().strip("*_ ") if m else ""

    def section(name):
        m = re.search(rf"##\s*{name}\s*\n(.*?)(?=\n##\s|\Z)", text, re.S | re.I)
        if not m:
            return []
        items = []
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^([-*•]|\d+[.)])\s*", "", line).strip()
            line = line.replace("**", "").strip()
            if line:
                items.append(line)
        return items

    data["CURE"] = section("CURE")
    data["PREVENTION"] = section("PREVENTION")
    return data


# =====================================================================
# 4. LABEL MAPS
# =====================================================================
PART_LABELS = {
    "LEAF": "🍃 पान (Leaf)",
    "FLOWER": "🌸 फूल (Flower)",
    "FRUIT": "🍏 फळ / बोंड / शेंग (Fruit / Boll / Pod)",
    "STEM": "🌿 खोड / फांदी (Stem)",
    "WHOLE_PLANT": "🌱 संपूर्ण झाड (Whole plant)",
    "OTHER": "🌾 इतर भाग (Other)",
}
STATUS_LABELS = {
    "HEALTHY": "निरोगी", "DISEASED": "रोगग्रस्त", "PEST": "किडीचा प्रादुर्भाव",
    "NUTRIENT": "अन्नद्रव्य कमतरता", "UNCERTAIN": "निदान अनिश्चित",
}
SEVERITY_LABELS = {"NONE": "धोका नाही", "LOW": "सौम्य तीव्रता", "MEDIUM": "मध्यम तीव्रता", "HIGH": "तीव्र प्रादुर्भाव"}


def crop_emoji(crop_en):
    c = crop_en.lower()
    if "cotton" in c:
        return "☁️"
    if "soy" in c:
        return "🫘"
    if "potato" in c:
        return "🥔"
    return "🌾"


def flat(s):
    """Flatten indented HTML so Markdown never renders it as a code block."""
    return "".join(line.strip() for line in s.strip().splitlines())


QTY_RE = re.compile(
    r"([०-९0-9]+(?:[.,/][०-९0-9]+)?(?:\s*(?:ते|-|–)\s*[०-९0-9]+(?:[.,][०-९0-9]+)?)?"
    r"\s*(?:ग्रॅम|ग्राम|मि\.ली\.|मिली|मिलि|ml|mL|ML|gm|kg|किलो|लिटर|g|%|टक्के)(?![A-Za-z]))"
)


def li(items):
    """Bullet list; dosages (e.g. २ ग्रॅम, 3 ml, ५%) are auto-bolded."""
    out = []
    for i in items:
        out.append("<li>" + QTY_RE.sub(r"<b>\1</b>", escape(i)) + "</li>")
    return "<ul class='lst'>" + "".join(out) + "</ul>"


# =====================================================================
# 5. STYLES
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Mukta:wght@400;500;600;700;800&family=Poppins:wght@500;600;700&display=swap');
:root{
  --em-950:#073B22; --em-900:#0B4F30; --em-800:#0F6639; --em-700:#167A49; --em-600:#1E874B; --em-500:#22A061; --em-100:#DFF2E5;
  --teal:#00D2B4; --gold:#F5C453;
  --mint:#F4F8F4; --line:#E2EDE3; --ink:#12281D; --muted:#5B7264;
  --red-700:#B42318; --red-100:#FDECEA; --red-line:#F5C2BD;
  --amb-700:#8A5A00; --amb-100:#FFF6E0; --amb-line:#F0D9A0;
}
html, body, [class*="css"], .stApp, .stMarkdown, p, label, button, input{font-family:'Mukta','Poppins',sans-serif !important;}
.stApp{
  background:radial-gradient(800px 320px at 100% -5%, #E3F4E9 0%, transparent 60%),
             radial-gradient(600px 280px at -10% 0%, #EAF6EE 0%, transparent 55%), var(--mint);
  color:var(--ink);
}
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{display:none !important;}
.block-container{max-width:720px !important; padding:.9rem .9rem 2.5rem !important;}
div[data-testid="stVerticalBlock"]{gap:.6rem !important;}
div[data-testid="stElementContainer"]:empty, div.element-container:empty{display:none !important;}
div[data-testid="stMarkdownContainer"] p:empty{display:none !important;}

/* ---------- Compact hero ---------- */
.hero{position:relative; overflow:hidden; border-radius:20px; padding:.8rem .95rem;
  background:linear-gradient(135deg,#073B22 0%,#0F6639 55%,#1E874B 100%); color:#fff;
  border:1px solid rgba(255,255,255,.22);
  box-shadow:0 10px 26px rgba(7,59,34,.28), inset 0 1px 0 rgba(255,255,255,.18);}
.hero::before{content:""; position:absolute; right:-40px; top:-60px; width:190px; height:190px; border-radius:50%;
  background:radial-gradient(circle,rgba(0,210,180,.28) 0%,transparent 68%); pointer-events:none;}
.hero-row{position:relative; display:flex; align-items:center; justify-content:space-between; gap:.6rem; flex-wrap:wrap;}
.brand{display:flex; align-items:center; gap:.65rem;}
.brand svg{width:46px; height:46px; flex-shrink:0;}
.bt b{display:block; font-size:1.42rem; font-weight:800; line-height:1.05; letter-spacing:.2px;}
.bt b i{font-style:normal; color:#7DF3E0; font-family:'Poppins',sans-serif !important; font-weight:700;}
.bt small{display:block; font-size:.8rem; opacity:.88; margin-top:.2rem; font-weight:500;}
.status{display:inline-flex; align-items:center; gap:.4rem; padding:.24rem .7rem; border-radius:999px; background:rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.32); font-size:.76rem; font-weight:600; white-space:nowrap; backdrop-filter:blur(6px);}
.status .dot{width:7px;height:7px;border-radius:50%;background:#6BFFA6; animation:pulse 2s infinite;}
.status.off .dot{background:#FFB4A9; animation:none;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(107,255,166,.7);}70%{box-shadow:0 0 0 7px rgba(107,255,166,0);}100%{box-shadow:0 0 0 0 rgba(107,255,166,0);}}

/* ---------- Section titles ---------- */
.sec{display:flex; align-items:center; gap:.5rem; margin:1.1rem 0 .25rem;}
.sec .bar{width:4px; height:18px; border-radius:4px; background:linear-gradient(180deg,var(--em-500),var(--em-800));}
.sec h3{font-size:1.1rem !important; font-weight:700; margin:0 !important; padding:0 !important; color:var(--em-900);}
.hint{color:var(--muted); font-size:.9rem; margin:0 0 .2rem; line-height:1.5;}

/* ---------- Input card ---------- */
.st-key-inputcard, div[data-testid="stVerticalBlockBorderWrapper"]{
  background:#fff !important; border:1px solid var(--line) !important; border-radius:20px !important;
  padding:.95rem !important; box-shadow:0 8px 24px rgba(11,79,48,.09);}
.card-h{display:flex; align-items:center; gap:.5rem; margin:0 0 .15rem;}
.card-h .ico{width:32px; height:32px; border-radius:10px; display:flex; align-items:center; justify-content:center; background:var(--em-100); font-size:1rem;}
.card-h b{font-size:1.08rem; color:var(--em-900); font-weight:700;}

div[data-testid="stRadio"] > label{display:none !important;}
div[role="radiogroup"]{display:flex !important; gap:.3rem; background:#EEF5EF; border:1px solid var(--line); padding:.28rem; border-radius:14px;}
div[role="radiogroup"] > label{flex:1; justify-content:center; margin:0 !important; padding:.55rem .3rem !important; border-radius:11px; cursor:pointer; transition:background .2s, box-shadow .2s; text-align:center;}
div[role="radiogroup"] > label > div:first-child{display:none !important;}
div[role="radiogroup"] > label p{font-weight:600; font-size:.9rem; margin:0; color:var(--muted); line-height:1.3;}
div[role="radiogroup"] > label:has(input:checked){background:linear-gradient(135deg,#0F6639,#1E874B); box-shadow:0 4px 12px rgba(15,102,57,.35);}
div[role="radiogroup"] > label:has(input:checked) p{color:#fff;}

[data-testid="stFileUploader"] label{display:none !important;}
[data-testid="stFileUploaderDropzone"]{background:#F6FBF7 !important; border:2px dashed #9CCFAE !important; border-radius:16px !important; padding:1.1rem .8rem !important; transition:border-color .2s, background .2s;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--em-600) !important; background:#fff !important;}
[data-testid="stFileUploaderDropzone"] button{background:linear-gradient(135deg,#0F6639,#1E874B) !important; color:#fff !important; border:none !important; border-radius:999px !important; padding:.45rem 1.1rem !important; font-weight:600;}
[data-testid="stCameraInput"]{border-radius:16px; overflow:hidden; border:1px solid var(--line); background:#F6FBF7;}
[data-testid="stCameraInput"] button{background:linear-gradient(135deg,#0F6639,#1E874B) !important; color:#fff !important; border-radius:999px !important; border:none !important;}
[data-testid="stCameraInput"] label{display:none !important;}
[data-testid="stImage"] img{border-radius:18px; border:1px solid var(--line); box-shadow:0 8px 22px rgba(11,79,48,.14);}

/* ---------- Pills ---------- */
.pills{display:flex; gap:.45rem; flex-wrap:wrap; margin:.5rem 0 .1rem;}
.pill{display:inline-flex; align-items:center; gap:.4rem; padding:.38rem .85rem; border-radius:999px; background:#fff; border:1px solid var(--line);
  font-weight:600; font-size:.95rem; color:var(--em-900); box-shadow:0 2px 6px rgba(11,79,48,.06);}
.pill small{font-weight:500; color:var(--muted); font-size:.8rem;}
.pill.crop{background:var(--em-100); border-color:#B9DDC5;}

/* ---------- Status card ---------- */
.result{border-radius:20px; padding:1.05rem 1rem; margin-top:.6rem; border:1px solid var(--line); box-shadow:0 10px 26px rgba(11,79,48,.12);}
.result.ok{background:linear-gradient(160deg,#FFFFFF 0%,#E1F4E8 100%); border-left:7px solid var(--em-500);}
.result.bad{background:linear-gradient(160deg,#FFFFFF 0%,var(--red-100) 100%); border-color:var(--red-line); border-left:7px solid var(--red-700);}
.result.warn{background:linear-gradient(160deg,#FFFFFF 0%,var(--amb-100) 100%); border-color:var(--amb-line); border-left:7px solid #E0A93B;}
.result .row{display:flex; align-items:flex-start; gap:.75rem;}
.ico-lg{width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:1.45rem; flex-shrink:0; background:#fff; box-shadow:0 4px 12px rgba(0,0,0,.08);}
.grow{flex:1; min-width:0;}
.tag{display:inline-flex; align-items:center; font-weight:700; font-size:.86rem; padding:.2rem .65rem; border-radius:999px;}
.ok .tag{background:var(--em-100); color:var(--em-900);}
.bad .tag{background:#fff; color:var(--red-700); border:1px solid var(--red-line);}
.warn .tag{background:#fff; color:var(--amb-700); border:1px solid var(--amb-line);}
.result h2{font-size:1.28rem !important; line-height:1.35; font-weight:800; margin:.5rem 0 .1rem !important; padding:0 !important;}
.ok h2{color:var(--em-900) !important;} .bad h2{color:var(--red-700) !important;} .warn h2{color:var(--amb-700) !important;}
.result .en{color:var(--muted); font-size:.9rem; font-family:'Poppins',sans-serif !important;}
.result .sym{margin-top:.75rem; font-size:1rem; line-height:1.6; color:#26392E;}
.metric{text-align:right; flex-shrink:0;}
.metric .num{font-family:'Poppins',sans-serif !important; font-weight:700; font-size:1.7rem; line-height:1;}
.ok .num{color:var(--em-700);} .bad .num{color:var(--red-700);} .warn .num{color:var(--amb-700);}
.metric .cap{font-size:.72rem; color:var(--muted); margin-top:.2rem; max-width:92px; line-height:1.25;}
.track{height:8px; border-radius:999px; background:rgba(0,0,0,.07); overflow:hidden; margin-top:.85rem;}
.fill{height:100%; border-radius:999px; transition:width .8s ease;}
.ok .fill{background:linear-gradient(90deg,var(--em-500),var(--em-900));}
.bad .fill{background:linear-gradient(90deg,#EF7B6E,var(--red-700));}
.warn .fill{background:linear-gradient(90deg,#F3C766,#C48A12);}

/* ---------- Advice cards ---------- */
.adv{border-radius:20px; padding:1rem 1rem .8rem; margin-top:.7rem; background:#fff; border:1px solid var(--line); box-shadow:0 6px 18px rgba(11,79,48,.07);}
.adv.cure{border-top:5px solid var(--red-700);}
.adv.cure.okc{border-top-color:var(--em-500);}
.adv.care{border-top:5px solid var(--em-500);}
.adv .head{display:flex; align-items:center; gap:.55rem; margin-bottom:.3rem;}
.adv .ico{width:36px; height:36px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.15rem; background:var(--red-100);}
.adv.care .ico, .adv.cure.okc .ico{background:var(--em-100);}
.adv h4{margin:0 !important; padding:0 !important; font-size:1.02rem !important; font-weight:700; color:var(--ink);}
.lst{margin:.25rem 0 0; padding-left:1.1rem;}
.lst li{font-size:1rem; line-height:1.65; color:#26392E; margin:.4rem 0;}
.lst li::marker{color:var(--em-500);}
.lst li b{color:var(--em-900); background:linear-gradient(transparent 62%, #CBEFD6 62%); padding:0 .1rem; font-weight:700;}
.adv.cure:not(.okc) .lst li b{color:#7A1A12; background:linear-gradient(transparent 62%, #FAD3CF 62%);}

.notplant{border-radius:20px; padding:1.3rem 1.1rem; margin-top:.7rem; background:var(--amb-100); border:1px solid var(--amb-line); text-align:center;}
.notplant .big{font-size:2.2rem;}
.notplant h3{margin:.3rem 0 .4rem !important; padding:0 !important; color:var(--amb-700); font-size:1.2rem !important;}
.notplant p{margin:0 0 .5rem; color:#5f4700; font-size:1rem; line-height:1.6;}
.note{margin-top:.7rem; padding:.65rem .85rem; border-radius:14px; background:#fff; border:1px solid var(--line); color:var(--muted); font-size:.9rem; line-height:1.55;}
.setup{border-radius:18px; padding:1rem; margin-top:.7rem; background:#fff; border:1px solid var(--amb-line); border-left:6px solid #E0A93B; color:#3d3200; line-height:1.65; font-size:.95rem;}
.setup code{background:#F4F1E4; padding:.1rem .4rem; border-radius:6px; font-size:.88rem;}
.foot{text-align:center; color:var(--muted); font-size:.82rem; margin-top:1.6rem;}

@media (max-width:
