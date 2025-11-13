# ======================================================
# pages/Drawing_Meter_Reading.py — Drawing Meter OCR Page
# (Gemini 2.5 Flash + Mongo + CSV/XLSX/JSON)
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
COLL_NAME = os.getenv("DRAWING_COLLECTION", "drawing_meter_entries")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ------------------ CONSTANTS ------------------
ROW_COLUMNS = [
    "Drawing_Stage",
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
        if x in ["", None, "null", "-", "—", ",", ",,"]:
            return None
        return int(str(x).strip())
    except:
        return None

def to_float(x):
    try:
        if x in ["", None, "null", "-", "—", ",", ",,"]:
            return None
        return float(str(x).strip())
    except:
        return None

# ------------------ NORMALIZATION ------------------
def normalize_rows(rows: list) -> pd.DataFrame:
    """Fix: treat column1 as M/C No., column2 as 100%, handle ',,' repeat."""
    norm = []

    prev_mc = None
    prev_eff100 = None
    prev_opening = None
    prev_closing = None

    for r in rows or []:

        mc_raw = r.get("Mc_No")
        eff_raw = r.get("Efficiency_at_100%")
        op_raw = r.get("Opening_Meter_Reading")
        cl_raw = r.get("Closing_Meter_Reading")

        # ------ REPEAT PREVIOUS LOGIC ------
        if str(mc_raw).strip() in ["", ",", ",,"]:
            mc_no = prev_mc
        else:
            mc_no = to_int(mc_raw)
            prev_mc = mc_no

        if str(eff_raw).strip() in ["", ",", ",,"]:
            eff100 = prev_eff100
        else:
            eff100 = to_float(eff_raw)
            prev_eff100 = eff100

        if str(op_raw).strip() in ["", ",", ",,"]:
            opening = prev_opening
        else:
            opening = to_int(op_raw)
            prev_opening = opening

        if str(cl_raw).strip() in ["", ",", ",,"]:
            closing = prev_closing
        else:
            closing = to_int(cl_raw)
            prev_closing = closing

        # -------- DIFFERENCE --------
        diff = closing - opening if opening is not None and closing is not None else None

        row = {
            "Drawing_Stage": (r.get("Drawing_Stage") or "").strip(),
            "Mc_No": mc_no,
            "Efficiency_at_100%": eff100,
            "Opening_Meter_Reading": opening,
            "Closing_Meter_Reading": closing,
            "Difference": diff,
            "Efficiency": to_float(r.get("Efficiency")),
            "Worker_Name": str(r.get("Worker_Name") or "").strip()
        }

        norm.append(row)

    return pd.DataFrame(norm, columns=ROW_COLUMNS)

# ------------------ DRAWING SPLIT ------------------
def segregate_drawings(df: pd.DataFrame):
    df1 = df[df["Drawing_Stage"].str.contains("1", na=False)].copy()
    df2 = df[df["Drawing_Stage"].str.contains("2", na=False)].copy()
    df3 = df[df["Drawing_Stage"].str.contains("3", na=False)].copy()
    return df1, df2, df3

# ------------------ GEMINI OCR ------------------
def call_gemini_for_drawing(image_bytes: bytes, mime_type: str) -> dict:

    prompt = """
Extract from a Drawing Meter Reading register.

Rules:
- First column = Mc_No (1,2,3,4...)
- Second column = Efficiency_at_100% (values like 1030, 1050, 900)
- ,, means SAME AS PREVIOUS
- Return strict JSON

{
  "header": {"Date": "...", "Shift": "...", "Supervisor_Signature": "..."},
  "rows": [
    {
      "Drawing_Stage": "1st Drawing" | "2nd Drawing" | "3rd Drawing",
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
    except:
        return {"header": {}, "rows": []}

# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: dict, df: pd.DataFrame, img_name: str):
    doc = {
        "register_type": "Drawing Meter Reading",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": img_name,
        "extracted_data": df.to_dict(orient="records"),
        "validated": False,
    }
    return coll.find_one_and_update(
        {"original_image_name": img_name},
        {"$set": doc},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )

# ------------------ UI ------------------
st.title("🧵 Drawing Meter Reading OCR")

with st.sidebar:
    st.subheader("Input")
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
    st.info("Upload image to start.")
    st.stop()

st.image(img_bytes, caption="Input Image", use_column_width=True)

data = call_gemini_for_drawing(img_bytes, mime)
header = data.get("header", {})
rows = data.get("rows", [])

df = normalize_rows(rows)

df1, df2, df3 = segregate_drawings(df)

st.subheader("1️⃣ 1st Drawing")
df1_edit = st.data_editor(df1, use_container_width=True, key="draw1")

st.subheader("2️⃣ 2nd Drawing")
df2_edit = st.data_editor(df2, use_container_width=True, key="draw2")

st.subheader("3️⃣ 3rd Drawing")
df3_edit = st.data_editor(df3, use_container_width=True, key="draw3")

edited = pd.concat([df1_edit, df2_edit, df3_edit], ignore_index=True)

# -------- HEADER ----------
date_val = st.text_input("Date", value=header.get("Date") or "")
shift_val = st.text_input("Shift", value=header.get("Shift") or "")
sup_val = st.text_input("Supervisor Signature", value=header.get("Supervisor_Signature") or "")

header_edit = {"Date": date_val, "Shift": shift_val, "Supervisor_Signature": sup_val}

# -------- SAVE ----------
if st.button("💾 Save to MongoDB"):
    upsert_mongo(header_edit, edited, img_name)
    st.success("Saved successfully!")

# Export
st.download_button("⬇ CSV", edited.to_csv(index=False), "drawing_meter.csv")
st.download_button("⬇ JSON", edited.to_json(orient="records", indent=2), "drawing_meter.json")
