# ======================================================
# pages/Drawing_Meter_Reading.py — Drawing Meter OCR Page
# Gemini 2.5 Flash + Mongo + CSV/XLSX/JSON
# ======================================================
import os, io, json, re, datetime as dt
import pandas as pd
import streamlit as st
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Drawing Meter Reading OCR", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")

# ✅ Dedicated collection for Drawing Meter
COLL_NAME = os.getenv("DRAWING_COLLECTION", "drawing_meter_entries")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY is missing in environment variables.")

# ------------------ CONSTANTS ------------------
ROW_COLUMNS = [
    "Drawing_Stage",
    "Sl_No",
    "Mc_No",
    "Efficiency_at_100%",
    "Opening_Meter_Reading",
    "Closing_Meter_Reading",
    "Difference",
    "Efficiency",
    "Worker_Name"
]

# ------------------ HELPERS ------------------
def json_safe_load(s: str) -> dict:
    """Safely load JSON response from Gemini."""
    try:
        return json.loads(s)
    except:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return {}
        return {}

def to_int(x):
    try:
        if x in [None, "", "null", "-", "—"]:
            return None
        return int(str(x).strip())
    except:
        return None

def to_float(x):
    try:
        if x in [None, "", "null", "-", "—"]:
            return None
        return float(str(x).strip())
    except:
        return None

def normalize_rows(rows: list) -> pd.DataFrame:
    """Normalize OCR rows — fix M/C No, repeat values, compute diff."""
    norm = []

    prev_mc = prev_eff100 = prev_open = prev_close = None

    def repeat_prev(val, prev):
        if str(val).strip() in ["", ",,", "''", "\"\"", None]:
            return prev
        return val

    for r in rows or []:
        mc_raw = repeat_prev(r.get("Mc_No"), prev_mc)
        eff100_raw = repeat_prev(r.get("Efficiency_at_100%"), prev_eff100)
        op_raw = repeat_prev(r.get("Opening_Meter_Reading"), prev_open)
        cl_raw = repeat_prev(r.get("Closing_Meter_Reading"), prev_close)

        prev_mc, prev_eff100, prev_open, prev_close = mc_raw, eff100_raw, op_raw, cl_raw

        mc_no = to_int(mc_raw)
        eff100 = to_float(eff100_raw)

        # FIX 1 — MC No must be 1–30, Efficiency must be 900–1200
        if mc_no and mc_no > 30:
            mc_no, eff100 = None, mc_no

        if eff100 and eff100 < 900:
            eff100 = None

        opening = to_int(op_raw)
        closing = to_int(cl_raw)

        diff = closing - opening if opening and closing else None

        row = {
            "Drawing_Stage": (r.get("Drawing_Stage") or "").strip(),
            "Sl_No": to_int(r.get("Sl_No")),
            "Mc_No": mc_no,
            "Efficiency_at_100%": eff100,
            "Opening_Meter_Reading": opening,
            "Closing_Meter_Reading": closing,
            "Difference": diff,
            "Efficiency": to_float(r.get("Efficiency")),
            "Worker_Name": str(r.get("Worker_Name") or "").strip(),
        }
        norm.append(row)

    return pd.DataFrame(norm, columns=ROW_COLUMNS)

def segregate_drawings(df: pd.DataFrame):
    """Separate 1st, 2nd, 3rd drawing tables."""
    df1 = df[df["Drawing_Stage"].str.contains("1", case=False, na=False)].copy()
    df2 = df[df["Drawing_Stage"].str.contains("2", case=False, na=False)].copy()
    df3 = df[df["Drawing_Stage"].str.contains("3", case=False, na=False)].copy()
    return df1, df2, df3

# ------------------ GEMINI OCR ------------------
def call_gemini_for_drawing(image_bytes: bytes, mime_type: str) -> dict:
    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY missing")
        return {"header": {}, "rows": []}

    prompt = """
You are reading a DRAWING METER READING REGISTER.

Rules:
- Column 1 = M/C No. (1–30 only)
- Column 2 = Efficiency at 100% (typically 900–1200)
- If a value is written as ,, it means SAME AS PREVIOUS ROW.
- Drawing_Stage should be one of: "1st Drawing", "2nd Drawing", "3rd Drawing".

Return STRICT JSON:
{
  "header": {
    "Date": "DD/MM/YYYY",
    "Shift": "A/B/C",
    "Supervisor_Signature": "Name"
  },
  "rows": [
    {
      "Drawing_Stage": "...",
      "Sl_No": int,
      "Mc_No": int,
      "Efficiency_at_100%": float,
      "Opening_Meter_Reading": int,
      "Closing_Meter_Reading": int,
      "Efficiency": float,
      "Worker_Name": str
    }
  ]
}
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}],
            generation_config=cfg,
        )
        return json_safe_load(resp.text)
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return {"header": {}, "rows": []}

# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: dict, df: pd.DataFrame, img_name: str):
    if df.empty:
        st.error("❌ No data to save.")
        return None

    doc = {
        "register_type": "Drawing Meter Reading",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": img_name,
        "extracted_data": df.to_dict(orient="records"),
        "validated": False,
    }
    key = {"original_image_name": img_name}
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True, return_document=ReturnDocument.AFTER)

# ------------------ UI ------------------
st.title("🧵 Drawing Meter Reading — OCR Automation")

with st.sidebar:
    st.subheader("Upload Input")
    cam = st.camera_input("📸 Capture Image")
    up = st.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg"])

img_bytes = img_name = mime = None
if cam:
    img_bytes = cam.getvalue()
    img_name = f"cam_{dt.datetime.utcnow().isoformat()}.jpg"
    mime = "image/jpeg"
elif up:
    img_bytes = up.getvalue()
    img_name = up.name
    mime = up.type

if not img_bytes:
    st.info("Upload or capture a Drawing Meter Reading sheet to begin.")
    st.stop()

st.image(img_bytes, caption="Uploaded Image", use_container_width=True)

st.markdown("### 🔍 Step 1 — Extracting with Gemini…")
data = call_gemini_for_drawing(img_bytes, mime)

header = data.get("header", {})
rows = data.get("rows", [])

df = normalize_rows(rows)

# ------------------ DISPLAY SEPARATE TABLES ------------------
st.markdown("### ✏️ Step 2 — Review by Drawing Stage")

df1, df2, df3 = segregate_drawings(df)

st.subheader("1️⃣ 1st Drawing")
df1_edit = st.data_editor(df1, use_container_width=True, num_rows="dynamic")

st.subheader("2️⃣ 2nd Drawing")
df2_edit = st.data_editor(df2, use_container_width=True, num_rows="dynamic")

st.subheader("3️⃣ 3rd Drawing")
df3_edit = st.data_editor(df3, use_container_width=True, num_rows="dynamic")

edited_df = pd.concat([df1_edit, df2_edit, df3_edit], ignore_index=True)

# ------------------ HEADER ------------------
st.markdown("### 📝 Step 3 — Header")
c1, c2, c3 = st.columns(3)
date_val = c1.text_input("Date", value=header.get("Date") or "")
shift_val = c2.text_input("Shift", value=header.get("Shift") or "")
sup_val = c3.text_input("Supervisor Signature", value=header.get("Supervisor_Signature") or "")

header_edit = {"Date": date_val, "Shift": shift_val, "Supervisor_Signature": sup_val}

# ------------------ AVERAGE EFFICIENCY ------------------
st.markdown("### 📊 Step 4 — Efficiency Summary")

if not edited_df.empty:
    avg_eff = edited_df["Efficiency"].mean()
    st.metric("📈 Average Efficiency (%)", f"{avg_eff:.2f}" if avg_eff else "N/A")

# ------------------ EXPORT ------------------
st.markdown("### 💾 Step 5 — Save & Export")
cA, cB, cC = st.columns(3)

if cA.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(header_edit, edited_df, img_name)
    if saved:
        st.success("✔ Saved to MongoDB successfully!")
        st.json({"_id": str(saved.get("_id")), "Date": header_edit["Date"]})

# Export CSV
csv_bytes = edited_df.to_csv(index=False).encode()
cB.download_button(
    "⬇️ Download CSV",
    data=csv_bytes,
    file_name="drawing_meter_data.csv",
    mime="text/csv"
)

# Export JSON
json_bytes = edited_df.to_json(orient="records", indent=2).encode()
cC.download_button(
    "⬇️ Download JSON",
    data=json_bytes,
    file_name="drawing_meter_data.json",
    mime="application/json"
)

st.markdown("---")
st.caption("Automated OCR • Drawing Meter Reading • Hastings Jute Mill")
