# ================================================
# pages/Spool_Winding.py — Spool Winding Production OCR
# Gemini 2.5 Flash | Mongo | CSV/JSON/XLSX
# ================================================
import os, io, json, re, datetime as dt
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------ CONFIG ------------------
st.set_page_config(page_title="🧵 Spool Winding Production OCR", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")
COLL_NAME = os.getenv("SPOOL_COLLECTION", "spool_winding_entries")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY missing in .env")

# ------------------ CONSTANTS ------------------
ROW_COLUMNS = [
    "Sl_No", "Quality_Mc_Alloc", "Spinning_Frame_No", "Winding_Frame_No",
    "Labour_No", "Production_Per_Fera", "Production_Sum", "Net_Weight", "Remarks"
]
HEADER_FIELDS = ["Date", "Shift", "Hands", "Number_of_Frames", "Unit", "Title"]

# ------------------ HELPERS ------------------
def json_safe_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

def to_int(x):
    try:
        if x in [None, "", "null"]:
            return None
        return int(str(x).strip())
    except Exception:
        return None

def to_int_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        return [int(str(v)) for v in x if str(v).isdigit()]
    return [int(n) for n in re.findall(r"\d+", str(x))]

def normalize_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    norm = []
    for r in rows:
        row = {
            "Sl_No": to_int(r.get("Sl_No")),
            "Quality_Mc_Alloc": r.get("Quality_Mc_Alloc") or r.get("Quality"),
            "Spinning_Frame_No": str(r.get("Spinning_Frame_No") or ""),
            "Winding_Frame_No": str(r.get("Winding_Frame_No") or ""),
            "Labour_No": to_int(r.get("Labour_No")),
            "Production_Per_Fera": to_int_list(r.get("Production_Per_Fera")),
            "Net_Weight": to_int(r.get("Net_Weight")),
            "Remarks": r.get("Remarks"),
        }
        row["Production_Sum"] = sum(row["Production_Per_Fera"]) if row["Production_Per_Fera"] else None
        norm.append(row)
    return pd.DataFrame(norm, columns=ROW_COLUMNS)

# ------------------ GEMINI OCR ------------------
def call_gemini_for_spool(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {"header": {}, "rows": []}

    prompt = """
You are parsing a Spool Winding Production register page.

Header fields: Date, Shift, Hands, Number_of_Frames, Unit, Title
Table columns:
Sl_No, Quality_Mc_Alloc, Spinning_Frame_No, Winding_Frame_No, Labour_No,
Production_Per_Fera, Production_Sum, Net_Weight, Remarks

Return valid JSON ONLY:
{
  "header": {
    "Date": "DD/MM/YY",
    "Shift": "A/B/C",
    "Hands": int,
    "Number_of_Frames": int,
    "Unit": str,
    "Title": str
  },
  "rows": [...]
}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content([prompt, {"mime_type": mime_type, "data": image_bytes}],
                                      generation_config=cfg)
        return json_safe_load(resp.text) or {"header": {}, "rows": []}
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header": {}, "rows": []}

# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: Dict[str, Any], df: pd.DataFrame, image_name: str, raw_bytes: bytes):
    doc = {
        "register_type": "Spool Winding Production",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": image_name,
        "extracted_data": df.to_dict(orient="records"),
        "validated": False,
    }
    key = {"original_image_name": image_name, "header.Date": header.get("Date")}
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True,
                                    return_document=ReturnDocument.AFTER)

# ------------------ STREAMLIT UI ------------------
st.title("🧶 Spool Winding Production — OCR")
with st.sidebar:
    st.subheader("📤 Upload Input")
    cam = st.camera_input("📸 Capture Image (optional)")
    up = st.file_uploader("📁 Upload Register Image", type=["png", "jpg", "jpeg"])

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
    st.info("📸 Please upload or capture a register image to begin OCR extraction.")
    st.stop()

st.image(img_bytes, caption="Input Image Preview", use_container_width=True)
st.markdown("**Step 1:** Extracting data with Gemini…")
data = call_gemini_for_spool(img_bytes, mime)

header = data.get("header", {}) or {}
rows = data.get("rows", []) or []
df = normalize_rows(rows)

# ------------------ Editor ------------------
st.markdown("**Step 2:** Review & Edit Extracted Data")
df["Spinning_Frame_No"] = df["Spinning_Frame_No"].astype(str)
df["Winding_Frame_No"] = df["Winding_Frame_No"].astype(str)
df_preview = df.copy()
df_preview["Production_Per_Fera"] = df_preview["Production_Per_Fera"].apply(
    lambda x: ", ".join(map(str, x)) if x else ""
)

edited = st.data_editor(
    df_preview,
    use_container_width=True,
    num_rows="dynamic",
    key="spool_editor",
)

edited_df = edited.copy()
edited_df["Production_Per_Fera"] = edited_df["Production_Per_Fera"].apply(to_int_list)
edited_df["Production_Sum"] = edited_df["Production_Per_Fera"].apply(lambda lst: sum(lst) if lst else None)

# ------------------ Header ------------------
st.markdown("**Step 3:** Header Information")
c1, c2, c3, c4 = st.columns(4)
date_val = c1.text_input("📅 Date", value=header.get("Date") or "")
shift_val = c2.text_input("🕒 Shift", value=header.get("Shift") or "")
hands_val = c3.text_input("✋ Hands", value=str(header.get("Hands") or ""))
frames_val = c4.text_input("🧾 Frames", value=str(header.get("Number_of_Frames") or ""))

header_edit = {
    "Date": date_val,
    "Shift": shift_val,
    "Hands": hands_val,
    "Number_of_Frames": frames_val,
    "Unit": header.get("Unit"),
    "Title": header.get("Title"),
}

# ------------------ Export ------------------
st.markdown("**Step 4:** Save / Export Data")
cA, cB, cC = st.columns(3)

if cA.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(header_edit, edited_df, img_name, img_bytes)
    st.success("✅ Data Saved to MongoDB Successfully!")
    st.json({"_id": str(saved.get("_id")), "Date": header_edit["Date"]})

# CSV
csv_bytes = edited_df.to_csv(index=False).encode()
cB.download_button("⬇️ CSV", csv_bytes, "spool_winding.csv", "text/csv")

# JSON
json_bytes = edited_df.to_json(orient="records", indent=2).encode()
cC.download_button("⬇️ JSON", json_bytes, "spool_winding.json", "application/json")

# XLSX
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    edited_df.to_excel(writer, index=False, sheet_name="SpoolWinding")
st.download_button(
    "⬇️ XLSX",
    data=xlsx_buf.getvalue(),
    file_name="spool_winding.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")
st.caption("💡 PDF export removed for stability. Use CSV or XLSX for reports.")
