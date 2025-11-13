# ================================================
# pages/Spinning_Production.py
# Gemini 2.5 Flash OCR • Mongo • CSV/JSON/XLSX
# ================================================
import os, io, json, re, datetime as dt
import pandas as pd
import streamlit as st
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Spinning Production OCR", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")
COLL_NAME = "spinning_production_entries"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY missing in .env (Google AI Studio)")

# ------------------ CONSTANTS ------------------
ROW_COLUMNS = [
    "Sl_No", "Quality", "Frame_A", "Frame_B", "Frame_C",
    "Production_A", "Production_B", "Production_C"
]
NUMERIC_COLS = ["Frame_A", "Frame_B", "Frame_C", "Production_A", "Production_B", "Production_C"]

# ------------------ HELPERS ------------------
def json_safe_load(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", str(s), flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

def to_int(x):
    try:
        if x in [None, "", "null", "-"]:
            return None
        return int(str(x).strip())
    except Exception:
        return None

def normalize_rows(rows: list) -> pd.DataFrame:
    """Normalize OCR rows and append a safe Total row."""
    norm = []
    for r in rows or []:
        row = {
            "Sl_No": to_int(r.get("Sl_No")),
            "Quality": (r.get("Quality") or "").strip(),
            "Frame_A": to_int(r.get("Frame_A")),
            "Frame_B": to_int(r.get("Frame_B")),
            "Frame_C": to_int(r.get("Frame_C")),
            "Production_A": to_int(r.get("Production_A")),
            "Production_B": to_int(r.get("Production_B")),
            "Production_C": to_int(r.get("Production_C")),
        }
        norm.append(row)

    df = pd.DataFrame(norm, columns=ROW_COLUMNS)

    # Ensure numeric columns are numeric for stable sums
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not df.empty:
        total_row = {
            "Sl_No": "Total", "Quality": "",
            "Frame_A": int(df["Frame_A"].sum(skipna=True)),
            "Frame_B": int(df["Frame_B"].sum(skipna=True)),
            "Frame_C": int(df["Frame_C"].sum(skipna=True)),
            "Production_A": int(df["Production_A"].sum(skipna=True)),
            "Production_B": int(df["Production_B"].sum(skipna=True)),
            "Production_C": int(df["Production_C"].sum(skipna=True)),
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    return df

# ------------------ GEMINI OCR ------------------
def call_gemini_for_spinning(image_bytes: bytes, mime_type: str) -> dict:
    """Gemini 2.5 Flash OCR for Spinning Production."""
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {"header": {}, "rows": []}

    prompt = """
You are extracting from a Spinning Production register.

Return strict JSON ONLY:
{
  "header": {"Date": "DD/MM/YY or DD/MM/YYYY", "Supervisor_Signature": "Name"},
  "rows": [
    {"Sl_No": int, "Quality": str,
     "Frame_A": int, "Frame_B": int, "Frame_C": int,
     "Production_A": int, "Production_B": int, "Production_C": int}
  ]
}
- Each Frame/Production column is already shift-wise, so do NOT add Shift.
- Keep integers as integers; if blank write null.
- Do not include any text outside JSON.
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}],
            generation_config=cfg,
        )
        return json_safe_load(resp.text) or {"header": {}, "rows": []}
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header": {}, "rows": []}

# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: dict, df: pd.DataFrame, img_name: str, raw_bytes: bytes):
    # Optional: avoid storing the "Total" row in Mongo
    df_to_store = df.copy()
    if not df_to_store.empty:
        df_to_store = df_to_store[df_to_store["Sl_No"].astype(str).str.lower() != "total"]

    doc = {
        "register_type": "Spinning Production",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": img_name,
        "extracted_data": df_to_store.to_dict(orient="records"),
        "validated": False,
    }
    key = {"original_image_name": img_name, "header.Date": header.get("Date")}
    return coll.find_one_and_update(
        key, {"$set": doc}, upsert=True, return_document=ReturnDocument.AFTER
    )

# ------------------ STREAMLIT UI ------------------
st.title("🧵 Spinning Production OCR")

with st.sidebar:
    st.subheader("Input")
    cam = st.camera_input("📸 Capture Image (optional)")
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
    st.info("Upload or capture a Spinning Production sheet image to start.")
    st.stop()

st.image(img_bytes, caption="Input Image", use_column_width=True)
st.markdown("**Step 1:** Extracting with Gemini…")

data = call_gemini_for_spinning(img_bytes, mime)
header = data.get("header", {}) or {}
rows = data.get("rows", []) or []
df = normalize_rows(rows)

# ------------------ DISPLAY ------------------
st.markdown("**Step 2: Preview & Edit**")
edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")

# Header inputs
c1, c2 = st.columns(2)
date_val = c1.text_input("Date", value=header.get("Date") or "")
sup_val = c2.text_input("Supervisor Signature", value=header.get("Supervisor_Signature") or "")
header_edit = {"Date": date_val, "Supervisor_Signature": sup_val}

# ------------------ SAVE / EXPORT ------------------
st.markdown("**Step 3: Save & Export**")
cA, cB, cC = st.columns(3)

if cA.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(header_edit, edited, img_name, img_bytes)
    st.success("✅ Saved to MongoDB")
    st.json({"_id": str(saved.get("_id")), "Date": header_edit["Date"]})

# CSV
csv_bytes = edited.to_csv(index=False).encode()
cB.download_button("⬇️ CSV", data=csv_bytes, file_name="spinning_production.csv", mime="text/csv")

# JSON
json_bytes = edited.to_json(orient="records", indent=2).encode()
cC.download_button("⬇️ JSON", data=json_bytes, file_name="spinning_production.json", mime="application/json")

# XLSX
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    edited.to_excel(writer, index=False, sheet_name="SpinningProduction")
st.download_button(
    "⬇️ XLSX", data=xlsx_buf.getvalue(),
    file_name="spinning_production.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")
st.caption("PDF export removed for reliability — use CSV or XLSX instead.")
