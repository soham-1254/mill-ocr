# ======================================================
# pages/Roll_Stock_Carding.py — Roll Stock Carding OCR Page
# (Gemini 2.5 Flash + Mongo + /tmp Font-safe PDF)
# ======================================================
import os, io, json, re, datetime as dt
import pandas as pd
import streamlit as st
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai
from utils.pdf_utils import get_pdf_base   # ✅ centralized /tmp font base

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Roll Stock Carding OCR", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")
COLL_NAME = os.getenv("COLLECTION_NAME", "roll_stock_entries")
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
    "Sl_No", "Qty", "6AM", "11AM", "2PM", "5PM", "10PM", "Remarks", "K_Cutting", "L_Cutting"
]
HEADER_FIELDS = ["Date", "Shift", "Supervisor_Signature"]

# ------------------ HELPERS ------------------
def json_safe_load(s: str) -> dict:
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

def normalize_rows(rows: list) -> pd.DataFrame:
    """Normalize rows data and compute necessary fields."""
    norm = []
    for r in rows:
        row = {
            "Sl_No": to_int(r.get("Sl_No")),
            "Qty": r.get("Qty"),
            "6AM": to_int(r.get("6AM")),
            "11AM": to_int(r.get("11AM")),
            "2PM": to_int(r.get("2PM")),
            "5PM": to_int(r.get("5PM")),
            "10PM": to_int(r.get("10PM")),
            "Remarks": r.get("Remarks"),
            "K_Cutting": r.get("K_Cutting", ""),
            "L_Cutting": r.get("L_Cutting", ""),
        }
        norm.append(row)

    df = pd.DataFrame(norm, columns=ROW_COLUMNS)
    return df

# ------------------ GEMINI OCR ------------------
def call_gemini_for_roll_stock(image_bytes: bytes, mime_type: str) -> dict:
    """Call Gemini API for Roll Stock Carding OCR extraction"""
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {"header": {}, "rows": []}

    prompt = """
You are extracting rows from a Roll Stock Carding register page.

Return STRICT JSON ONLY:
{
  "header": {
    "Date": "DD/MM/YY or DD/MM/YYYY",
    "Shift": "A/B/C",
    "Supervisor_Signature": "Name"
  },
  "rows": [
    {
      "Sl_No": int,
      "Qty": str,
      "6AM": int,
      "11AM": int,
      "2PM": int,
      "5PM": int,
      "10PM": int,
      "Remarks": str,
      "K_Cutting": str,
      "L_Cutting": str
    }
  ]
}
Rules:
- Read each column exactly as printed.
- “K_Cutting” and “L_Cutting” are separate fields often at bottom/right of the sheet.
- Return only JSON (no text explanations).
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

# ------------------ ✅ PDF EXPORT (Unified Font via /tmp) ------------------
def export_pdf(df: pd.DataFrame, header: dict) -> bytes:
    """✅ Fully Streamlit-safe PDF export (always returns bytes)."""
    import io
    from fpdf import FPDF

    pdf = get_pdf_base("Roll Stock Carding — OCR Extract", header)
    pdf.set_font("NotoSans", "", 8)

    show_cols = ["Sl_No", "Qty", "6AM", "11AM", "2PM", "5PM", "10PM", "Remarks", "K_Cutting", "L_Cutting"]
    col_w = [10, 20, 20, 20, 20, 20, 20, 35, 25, 25]

    # ---- Table Header ----
    for i, c in enumerate(show_cols):
        pdf.cell(col_w[i], 6, c, border=1, align="C")
    pdf.ln()

    # ---- Table Rows ----
    for _, r in df.iterrows():
        row = [
            str(r.get("Sl_No") or ""),
            str(r.get("Qty") or ""),
            str(r.get("6AM") or ""),
            str(r.get("11AM") or ""),
            str(r.get("2PM") or ""),
            str(r.get("5PM") or ""),
            str(r.get("10PM") or ""),
            str(r.get("Remarks") or ""),
            str(r.get("K_Cutting") or ""),
            str(r.get("L_Cutting") or ""),
        ]
        for i, val in enumerate(row):
            pdf.cell(col_w[i], 6, val, border=1)
        pdf.ln()

    # ✅ Safe memory buffer output (instead of dest="S")
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)

    pdf_bytes = buf.getvalue()

    # ✅ Guarantee correct binary type
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1", errors="ignore")
    elif not isinstance(pdf_bytes, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_bytes)

    return pdf_bytes


# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: dict, df: pd.DataFrame, img_name: str, raw_bytes: bytes):
    doc = {
        "register_type": "Roll Stock Carding",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": img_name,
        "extracted_data": df.to_dict(orient="records"),
        "validated": False,
    }
    key = {"original_image_name": img_name, "header.Date": header.get("Date")}
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True, return_document=ReturnDocument.AFTER)

# ------------------ STREAMLIT UI ------------------
st.title("🧵 Roll Stock Carding OCR")

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
    st.info("Upload or capture a Roll Stock Carding sheet image to start.")
    st.stop()

st.image(img_bytes, caption="Input Image", use_container_width=True)
st.markdown("**Step 1:** Extracting with Gemini…")

data = call_gemini_for_roll_stock(img_bytes, mime)
header = data.get("header", {}) or {}
rows = data.get("rows", []) or []

df = normalize_rows(rows)

# ------------------ DISPLAYING AND EXPORTING ------------------
st.markdown("**Step 2: Review & Edit**")
edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")

# Step 3: Header
c1, c2, c3 = st.columns(3)
date_val = c1.text_input("Date", value=header.get("Date") or "")
shift_val = c2.text_input("Shift", value=header.get("Shift") or "")
supervisor_val = c3.text_input("Supervisor Signature", value=header.get("Supervisor_Signature") or "")
header_edit = {"Date": date_val, "Shift": shift_val, "Supervisor_Signature": supervisor_val}

# Step 4: Save and Export
if st.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(header_edit, edited, img_name, img_bytes)
    st.success("✅ Saved to MongoDB")
    st.json({"_id": str(saved.get("_id")), "Date": header_edit["Date"]})

c1, c2, c3, c4 = st.columns(4)

# CSV
csv_bytes = edited.to_csv(index=False).encode()
c1.download_button("⬇️ CSV", data=csv_bytes, file_name="roll_stock_data.csv", mime="text/csv")

# JSON
json_bytes = edited.to_json(orient="records", indent=2).encode()
c2.download_button("⬇️ JSON", data=json_bytes, file_name="roll_stock_data.json", mime="application/json")

# XLSX
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    edited.to_excel(writer, index=False, sheet_name="RollStock")
c3.download_button("⬇️ XLSX", data=xlsx_buf.getvalue(),
                   file_name="roll_stock_data.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# PDF
pdf_bytes = export_pdf(edited, header_edit)
c4.download_button("⬇️ PDF", data=pdf_bytes,
                   file_name="roll_stock_data.pdf", mime="application/pdf")

st.markdown("---")
st.caption("💡 Tip: You can refine values before saving or exporting.")
