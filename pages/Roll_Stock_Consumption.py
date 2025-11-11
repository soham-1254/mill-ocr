# ================================================
# pages/Roll_Stock_Consumption.py
# Gemini 2.5 Flash OCR • Mongo • CSV/JSON/XLSX/PDF
# with safe /tmp font via utils/pdf_utils.get_pdf_base
# ================================================
import os, io, json, re, datetime as dt
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai
from utils.pdf_utils import get_pdf_base

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Roll Stock Consumption OCR", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")
COLL_NAME = os.getenv("ROLL_STOCK_COLLECTION", "roll_stock_entries")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY missing in .env (Google AI Studio)")

# ------------------ CONSTANTS ------------------
HEADER_FIELDS = ["Date", "Unit", "Title"]
TABLE_COLS = ["Qty", "Maturity_Hrs", "AM_6", "AM_11", "PM_2", "PM_5", "PM_10"]

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
        if x in [None, "", "null", "-"]:
            return None
        return int(str(x).strip())
    except Exception:
        return None

def to_int_or_zero(x):
    v = to_int(x)
    return 0 if v is None else v

def normalize_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Coerce to schema with numeric columns. Keep Qty as text."""
    norm = []
    for r in rows:
        row = {
            "Qty": str(r.get("Qty") or r.get("Quality") or "").strip(),
            "Maturity_Hrs": to_int(r.get("Maturity_Hrs") or r.get("Maturity Hrs")),
            "AM_6": to_int_or_zero(r.get("AM_6") or r.get("6AM")),
            "AM_11": to_int_or_zero(r.get("AM_11") or r.get("11AM")),
            "PM_2": to_int_or_zero(r.get("PM_2") or r.get("2PM")),
            "PM_5": to_int_or_zero(r.get("PM_5") or r.get("5PM")),
            "PM_10": to_int_or_zero(r.get("PM_10") or r.get("10PM")),
        }
        norm.append(row)
    df = pd.DataFrame(norm, columns=TABLE_COLS)
    return df

# ------------------ GEMINI OCR ------------------
def call_gemini_for_roll_stock(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """Extract header + table rows + footer totals from Roll Stock Consumption sheet."""
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {"header": {}, "rows": [], "footer_totals": {}}

    prompt = f"""
You are parsing a Roll Stock Consumption register.

Extract strictly JSON:
{{
  "header": {{
    "Date": "DD/MM/YY or DD/MM/YYYY" or null,
    "Unit": str or null,
    "Title": str or null
  }},
  "rows": [
    {{
      "Qty": str,
      "Maturity_Hrs": int,
      "AM_6": int,
      "AM_11": int,
      "PM_2": int,
      "PM_5": int,
      "PM_10": int
    }}
  ],
  "footer_totals": {{
    "AM_6": int, "AM_11": int, "PM_2": int, "PM_5": int, "PM_10": int
  }}
}}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content([prompt, {"mime_type": mime_type, "data": image_bytes}], generation_config=cfg)
        data = json_safe_load(resp.text) or {"header": {}, "rows": [], "footer_totals": {}}
        data.setdefault("header", {})
        data.setdefault("rows", [])
        data.setdefault("footer_totals", {})
        return data
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header": {}, "rows": [], "footer_totals": {}}

# ------------------ ✅ PDF EXPORT (safe font via /tmp) ------------------
def export_pdf(df: pd.DataFrame, header: Dict[str, Any], footer: Dict[str, Any]) -> bytes:
    """✅ Fully Streamlit-safe PDF export (always returns bytes)."""
    import io
    from fpdf import FPDF

    pdf = get_pdf_base("Roll Stock Consumption — OCR Extract", header)
    pdf.set_font("NotoSans", "", 8)

    col_w = [18, 24, 22, 22, 22, 22, 22]

    # ---- Table Header ----
    for c in TABLE_COLS:
        pdf.cell(col_w[TABLE_COLS.index(c)], 6, c, border=1, align="C")
    pdf.ln()

    # ---- Table Rows ----
    for _, r in df.iterrows():
        values = [
            str(r.get("Qty") or ""),
            str(r.get("Maturity_Hrs") or ""),
            str(int(r.get("AM_6") or 0)),
            str(int(r.get("AM_11") or 0)),
            str(int(r.get("PM_2") or 0)),
            str(int(r.get("PM_5") or 0)),
            str(int(r.get("PM_10") or 0)),
        ]
        for i, v in enumerate(values):
            pdf.cell(col_w[i], 6, v, border=1)
        pdf.ln()

    # ---- Footer Totals ----
    pdf.ln(2)
    pdf.set_font("NotoSans", "", 9)
    ft = footer or {}
    pdf.cell(
        0,
        6,
        f"Footer Totals → 6AM: {ft.get('AM_6')} | 11AM: {ft.get('AM_11')} | 2PM: {ft.get('PM_2')} | 5PM: {ft.get('PM_5')} | 10PM: {ft.get('PM_10')}",
        ln=1,
    )

    # ✅ Safe PDF generation using BytesIO
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)

    pdf_bytes = buf.getvalue()

    # ✅ Guarantee pure bytes format
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1", errors="ignore")
    elif not isinstance(pdf_bytes, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_bytes)

    return pdf_bytes


# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: Dict[str, Any], df: pd.DataFrame, footer: Dict[str, Any], image_name: str, raw_bytes: bytes):
    doc = {
        "register_type": "Roll Stock Consumption",
        "header": header,
        "footer_totals": footer,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": image_name,
        "extracted_data": df.to_dict(orient="records"),
        "validated": False,
    }
    key = {"original_image_name": image_name, "header.Date": header.get("Date")}
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True, return_document=ReturnDocument.AFTER)

# ------------------ STREAMLIT UI ------------------
st.title("📊 Roll Stock Consumption — OCR")

with st.sidebar:
    st.subheader("Input")
    cam = st.camera_input("📸 Capture Image (optional)")
    up = st.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg"])

img_bytes = img_name = mime = None
if cam:
    img_bytes = cam.getvalue(); img_name = f"cam_{dt.datetime.utcnow().isoformat()}.jpg"; mime = "image/jpeg"
elif up:
    img_bytes = up.getvalue(); img_name = up.name; mime = up.type

if not img_bytes:
    st.info("Upload or capture a Roll Stock Consumption sheet to start.")
    st.stop()

st.image(img_bytes, caption="Input Image", use_column_width=True)
st.markdown("**Step 1:** Extracting with Gemini…")

data = call_gemini_for_roll_stock(img_bytes, mime)
header = data.get("header", {}) or {}
rows = data.get("rows", []) or []
footer = data.get("footer_totals", {}) or {}
df = normalize_rows(rows)

# ------------------ Display and Compute ------------------
st.markdown("**Step 2: Preview & Edit**")
edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")

# Derived column totals
col_sums = {c: int(edited[c].sum()) if c in edited else 0 for c in ["AM_6", "AM_11", "PM_2", "PM_5", "PM_10"]}

def cmp_footer(k):
    f = footer.get(k)
    return "Match ✅" if (f is not None and to_int(f) == col_sums[k]) else (f"⚠️ Mismatch ({col_sums[k]} vs {f})" if f else "—")

st.markdown("**Column Totals (After Edits)**")
cols = st.columns(5)
for i, k in enumerate(["AM_6", "AM_11", "PM_2", "PM_5", "PM_10"]):
    cols[i].metric(k.replace("_", " "), col_sums[k], cmp_footer(k))

# Header edit
st.markdown("**Step 3: Header Info**")
c1, c2, c3 = st.columns(3)
date_val = c1.text_input("Date", value=header.get("Date") or "")
unit_val = c2.text_input("Unit", value=header.get("Unit") or "HASTINGS JUTE MILL")
title_val = c3.text_input("Title", value=header.get("Title") or "ROLL STOCK CONSUMPTION")
header_edit = {"Date": date_val, "Unit": unit_val, "Title": title_val}

# Save / Export
st.markdown("**Step 4: Save / Export**")
b1, b2, b3, b4, b5 = st.columns(5)

if b1.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(header_edit, edited, footer, img_name, img_bytes)
    st.success("✅ Saved to MongoDB")
    st.json({"_id": str(saved.get("_id")), "Date": header_edit["Date"]})

# CSV
csv_bytes = edited.to_csv(index=False).encode()
b2.download_button("⬇️ CSV", data=csv_bytes, file_name="roll_stock_consumption.csv", mime="text/csv")

# JSON
json_bytes = edited.to_json(orient="records", indent=2).encode()
b3.download_button("⬇️ JSON", data=json_bytes, file_name="roll_stock_consumption.json", mime="application/json")

# XLSX
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    edited.to_excel(writer, index=False, sheet_name="RollStock")
b4.download_button(
    "⬇️ XLSX",
    data=xlsx_buf.getvalue(),
    file_name="roll_stock_consumption.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# PDF
pdf_bytes = export_pdf(edited, header_edit, footer)
b5.download_button("⬇️ PDF", data=pdf_bytes, file_name="roll_stock_consumption.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Tip: Adjust any numbers in the grid before saving or exporting.")
