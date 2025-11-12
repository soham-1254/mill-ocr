# ======================================================
# pages/Cop_Winding.py — Cop Winding (Weft) OCR
# (Gemini 2.5 Flash + 2-Step Tally Verification, no PDF)
# ======================================================
import os, io, json, re, datetime as dt
import pandas as pd
import streamlit as st
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Cop Winding (Weft) OCR", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")
COLL_NAME = os.getenv("COLLECTION_NAME", "cop_winding_entries")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY missing in .env (Google AI Studio)")

# ------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------
COP_COLUMNS = [
    "Sl_No","Frame_No","Worker_Name","Labour_No",
    "Quality","Frames","Winders","Marka","Spindle",
    "Tally_Marks","Tally_From_Marks","Tally_LastCol",
    "Final_Tally","Verified","Remarks",
    "Lot_Total_Tally","Lot_Footer_Total","Lot_Verified"
]

# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def parse_frames_winders(text):
    m = re.search(r"(\d+)\s*[-/xX]\s*(\d+)", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def parse_marka_spindle(text):
    m = re.search(r"([A-Za-z/]+)\s*[-/]\s*(\d+)", str(text or ""))
    return (m.group(1).upper(), int(m.group(2))) if m else (None, None)

def fallback_tally_count(text):
    if not text:
        return None
    s = str(text).strip().replace("\\", "/")
    groups = re.findall(r'[|/]+', s)
    total = 0
    for g in groups:
        if '/' in g:
            total += 5
        else:
            total += g.count('|')
    return total if total > 0 else None

def json_safe_load(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except:
        m = re.search(r"\{.*\}", str(s), flags=re.S)
        return json.loads(m.group(0)) if m else {}

def to_int(x):
    try:
        if x in [None, "", "null"]:
            return None
        return int(str(x).strip())
    except:
        return None

# ------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------
def normalize_and_verify(rows):
    rows = rows or []
    norm = []
    for r in rows:
        row = {c: r.get(c) for c in COP_COLUMNS if c in r}

        f, w = parse_frames_winders(r.get("Quality", "") or "")
        mk, sp = parse_marka_spindle(r.get("Quality", "") or "")
        row["Frames"] = to_int(row.get("Frames")) or f
        row["Winders"] = to_int(row.get("Winders")) or w
        row["Marka"] = row.get("Marka") or mk
        row["Spindle"] = to_int(row.get("Spindle")) or sp

        marks_count = to_int(row.get("Tally_From_Marks")) or fallback_tally_count(r.get("Tally_Marks"))
        last_col = to_int(row.get("Tally_LastCol"))

        if marks_count is not None and last_col is not None:
            verified = "Yes" if marks_count == last_col else "No"
            final_tally = last_col
        elif last_col is not None:
            verified = "No"; final_tally = last_col
        elif marks_count is not None:
            verified = "No"; final_tally = marks_count
        else:
            verified = "No"; final_tally = None

        row.update({
            "Tally_From_Marks": marks_count,
            "Tally_LastCol": last_col,
            "Final_Tally": final_tally,
            "Verified": verified,
            "Sl_No": to_int(row.get("Sl_No")),
            "Labour_No": to_int(row.get("Labour_No")),
            "Lot_Footer_Total": to_int(row.get("Lot_Footer_Total")),
        })
        norm.append(row)

    df = pd.DataFrame(norm)

    if df.empty:
        for c in COP_COLUMNS:
            df[c] = None
        return df

    for c in COP_COLUMNS:
        if c not in df.columns:
            df[c] = None

    df["Final_Tally"] = pd.to_numeric(df["Final_Tally"], errors="coerce").fillna(0).astype(int)
    lot_sum = df.groupby("Quality", dropna=False)["Final_Tally"].sum().reset_index()
    lot_sum.rename(columns={"Final_Tally": "Lot_Total_Tally"}, inplace=True)
    df = df.merge(lot_sum, on="Quality", how="left")

    footer = df.groupby("Quality", dropna=False)["Lot_Footer_Total"].max().reset_index()
    df = df.drop(columns=["Lot_Footer_Total"]).merge(footer, on="Quality", how="left")

    def lot_verify(row):
        if row.get("Lot_Footer_Total") is None:
            return "Unknown"
        try:
            return "Yes" if int(row["Lot_Total_Tally"]) == int(row["Lot_Footer_Total"]) else "No"
        except:
            return "Unknown"

    df["Lot_Verified"] = df.apply(lot_verify, axis=1)
    df = df.reindex(columns=COP_COLUMNS, fill_value=None)
    return df

# ------------------------------------------------------
# GEMINI OCR
# ------------------------------------------------------
def call_gemini_vision_for_cop(image_bytes, mime_type):
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {"header": {}, "rows": []}

    prompt = """You are extracting data from a Cop Winding (Weft) register.
Return JSON only with fields header and rows."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content([prompt, {"mime_type": mime_type, "data": image_bytes}],
                                      generation_config=cfg)
        data = json_safe_load(resp.text)
        data.setdefault("header", {})
        data.setdefault("rows", [])
        return data
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header": {}, "rows": []}

# ------------------------------------------------------
# MONGO SAVE
# ------------------------------------------------------
def upsert_mongo(header, df, img_name, img_bytes):
    doc = {
        "register_type": "Cop Winding (Weft)",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": img_name,
        "extracted_data": df.to_dict(orient="records"),
        "validated": False,
    }
    key = {"original_image_name": img_name, "header.Date": header.get("Date")}
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True,
                                    return_document=ReturnDocument.AFTER)

# ------------------------------------------------------
# UI
# ------------------------------------------------------
st.title("🧵 Cop Winding (Weft) OCR — 2-Step Verified Tallies")

with st.sidebar:
    st.subheader("Input")
    cam = st.camera_input("📸 Capture Image")
    up = st.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg"])

img_bytes = img_name = mime = None
if cam:
    img_bytes = cam.getvalue(); img_name = f"cam_{dt.datetime.utcnow().isoformat()}.jpg"; mime = "image/jpeg"
elif up:
    img_bytes = up.getvalue(); img_name = up.name; mime = up.type

if not img_bytes:
    st.info("📷 Upload or capture an image to start.")
    st.stop()

st.image(img_bytes, caption="Input Image", use_container_width=True)
st.markdown("**Step 1:** Extracting with Gemini…")
data = call_gemini_vision_for_cop(img_bytes, mime)
rows = data.get("rows", []) or []
header = data.get("header", {}) or {}

df = normalize_and_verify(rows)

# ------------------- Editor -------------------
st.markdown("**Step 2:** Review & Edit**")
edited = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Verified": st.column_config.SelectboxColumn(options=["Yes", "No"], width="small"),
        "Lot_Verified": st.column_config.SelectboxColumn(options=["Yes", "No", "Unknown"], width="small")
    },
)

# ------------------- Header -------------------
st.markdown("**Step 3:** Header Details**")
c1, c2, c3 = st.columns(3)
regn = c1.text_input("Register Name", header.get("Register_Name") or "S/Weft Wind")
shift = c2.text_input("Shift", header.get("Shift") or "")
date = c3.text_input("Date", header.get("Date") or "")
header_edit = {"Register_Name": regn, "Shift": shift, "Date": date}

# ------------------- Export -------------------
st.markdown("**Step 4:** Save & Export**")
c1, c2 = st.columns(2)

if c1.button("💾 Save to MongoDB", type="primary"):
    s = upsert_mongo(header_edit, edited, img_name, img_bytes)
    st.success("✅ Saved to MongoDB")
    st.json({"_id": str(s.get('_id')), "Date": date})

csv_bytes = edited.to_csv(index=False).encode()
c2.download_button("⬇️ CSV", csv_bytes, "cop_winding_data.csv", "text/csv")

st.markdown("---")
st.caption("Notes: PDF export removed for reliability. Use CSV for reporting.")
