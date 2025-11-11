# ================================================
# pages/Roll_Stock_Consumption.py
# (Gemini 2.5 Flash OCR • Mongo • CSV/JSON/XLSX/PDF)
# ================================================
import os, io, json, re, datetime as dt, requests
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai
from fpdf import FPDF

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
            "AM_6": to_int_or_zero(r.get("AM_6") or r.get("6AM") or r.get("6 A.M")),
            "AM_11": to_int_or_zero(r.get("AM_11") or r.get("11AM") or r.get("11 A.M")),
            "PM_2": to_int_or_zero(r.get("PM_2") or r.get("2PM") or r.get("2 P.M")),
            "PM_5": to_int_or_zero(r.get("PM_5") or r.get("5PM") or r.get("5 P.M")),
            "PM_10": to_int_or_zero(r.get("PM_10") or r.get("10PM") or r.get("10 P.M")),
        }
        # Keep "11" as string quality, not integer
        row["Qty"] = row["Qty"] if row["Qty"] != "None" else ""
        norm.append(row)
    df = pd.DataFrame(norm, columns=TABLE_COLS)
    return df

# ------------------ GEMINI OCR ------------------
def call_gemini_for_roll_stock(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Extract header + table rows + footer totals from Roll Stock Consumption sheet.
    """
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {"header": {}, "rows": [], "footer_totals": {}}

    prompt = f"""
You are parsing a **Roll Stock Consumption** register page.

Header fields to extract if visible:
- Date (e.g., 07/11/25)
- Unit (e.g., Hastings Jute Mill)
- Title (e.g., ROLL STOCK CONSUMPTION)

Table columns EXACTLY (match keys):
{TABLE_COLS}

Notes:
- "Qty" is the row key like P, O, T, X, B, K, J, I, 11, A, S, R — keep as string.
- Blank cells should be null (for header) and 0 for table numeric entries.
- If a total row exists at the bottom (e.g., "Total  | 1928 | 1910 | 1899 | 1880"), return it under:
  "footer_totals": {{"AM_6": int or null, "AM_11": int or null, "PM_2": int or null, "PM_5": int or null, "PM_10": int or null}}

Return strictly valid JSON:
{{
  "header": {{
    "Date": "DD/MM/YY or DD/MM/YYYY" or null,
    "Unit": str or null,
    "Title": str or null
  }},
  "rows": [
    {{
      "Qty": str or null,
      "Maturity_Hrs": int or null,
      "AM_6": int or null,
      "AM_11": int or null,
      "PM_2": int or null,
      "PM_5": int or null,
      "PM_10": int or null
    }}
  ],
  "footer_totals": {{
    "AM_6": int or null, "AM_11": int or null, "PM_2": int or null, "PM_5": int or null, "PM_10": int or null
  }}
}}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}],
            generation_config=cfg
        )
        data = json_safe_load(resp.text)
        if not data:
            data = {"header": {}, "rows": [], "footer_totals": {}}
        # ensure keys exist
        data.setdefault("header", {})
        data.setdefault("rows", [])
        data.setdefault("footer_totals", {})
        return data
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header": {}, "rows": [], "footer_totals": {}}

# ------------------ PDF (Unicode-safe) ------------------
def export_pdf(df: pd.DataFrame, header: Dict[str, Any], footer: Dict[str, Any]) -> bytes:
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    # Download NotoSans if missing
    root_dir = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
    font_path = os.path.join(root_dir, "NotoSans-Regular.ttf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
        r = requests.get(url); open(font_path, "wb").write(r.content)
    pdf.add_font("NotoSans", "", font_path, uni=True)
    pdf.set_font("NotoSans", "", 14)

    def safe(t): return re.sub(r"[—–−]", "-", str(t or ""))

    title = safe(header.get("Title") or "ROLL STOCK CONSUMPTION")
    pdf.cell(0, 8, f"{title} — OCR Extract", ln=1)
    pdf.set_font("NotoSans", "", 11)
    line = f"Date: {safe(header.get('Date'))}    Unit: {safe(header.get('Unit'))}"
    pdf.cell(0, 7, line, ln=1)
    pdf.ln(2)

    # table
    pdf.set_font("NotoSans", "", 8)
    col_w = [18, 24, 22, 22, 22, 22, 22]
    for i, c in enumerate(TABLE_COLS):
        pdf.cell(col_w[i], 6, c, border=1, align="C")
    pdf.ln()

    for _, r in df.iterrows():
        values = [
            str(r.get("Qty") or ""),
            str(r.get("Maturity_Hrs") if pd.notna(r.get("Maturity_Hrs")) else ""),
            str(int(r.get("AM_6") or 0)),
            str(int(r.get("AM_11") or 0)),
            str(int(r.get("PM_2") or 0)),
            str(int(r.get("PM_5") or 0)),
            str(int(r.get("PM_10") or 0)),
        ]
        for i, v in enumerate(values):
            pdf.cell(col_w[i], 6, v, border=1)
        pdf.ln()

    # footer totals
    pdf.ln(2)
    pdf.set_font("NotoSans", "", 9)
    ft = {
        "AM_6": footer.get("AM_6"),
        "AM_11": footer.get("AM_11"),
        "PM_2": footer.get("PM_2"),
        "PM_5": footer.get("PM_5"),
        "PM_10": footer.get("PM_10"),
    }
    pdf.cell(0, 6, f"Footer Totals → 6AM: {ft['AM_6']} | 11AM: {ft['AM_11']} | 2PM: {ft['PM_2']} | 5PM: {ft['PM_5']} | 10PM: {ft['PM_10']}", ln=1)

    return pdf.output(dest="S").encode("latin-1", errors="ignore")

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

# ------------------ UI ------------------
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

st.image(img_bytes, caption="Input Image", use_container_width=True)
st.markdown("**Step 1:** Extracting with Gemini…")

data = call_gemini_for_roll_stock(img_bytes, mime)
header = data.get("header", {}) or {}
rows = data.get("rows", []) or []
footer = data.get("footer_totals", {}) or {}

df = normalize_rows(rows)

# Derived column totals
col_sums = {
    "AM_6": int(df["AM_6"].sum()) if not df.empty else 0,
    "AM_11": int(df["AM_11"].sum()) if not df.empty else 0,
    "PM_2": int(df["PM_2"].sum()) if not df.empty else 0,
    "PM_5": int(df["PM_5"].sum()) if not df.empty else 0,
    "PM_10": int(df["PM_10"].sum()) if not df.empty else 0,
}

# Comparisons vs footer
def cmp_footer(key):
    f = footer.get(key)
    return "Match ✅" if (f is not None and to_int(f) == col_sums[key]) else ("No footer" if f in [None, ""] else f"Mismatch ⚠️ (sum={col_sums[key]}, footer={f})")

st.markdown("**Step 2: Preview & Edit**")
edited = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Qty": st.column_config.TextColumn("Qty"),
        "Maturity_Hrs": st.column_config.NumberColumn("Maturity Hrs"),
        "AM_6": st.column_config.NumberColumn("6 A.M"),
        "AM_11": st.column_config.NumberColumn("11 A.M"),
        "PM_2": st.column_config.NumberColumn("2 P.M"),
        "PM_5": st.column_config.NumberColumn("5 P.M"),
        "PM_10": st.column_config.NumberColumn("10 P.M"),
    },
    key="roll_stock_editor"
)

# Recompute sums after edits
edited_sums = {
    "AM_6": int(edited["AM_6"].fillna(0).sum()) if not edited.empty else 0,
    "AM_11": int(edited["AM_11"].fillna(0).sum()) if not edited.empty else 0,
    "PM_2": int(edited["PM_2"].fillna(0).sum()) if not edited.empty else 0,
    "PM_5": int(edited["PM_5"].fillna(0).sum()) if not edited.empty else 0,
    "PM_10": int(edited["PM_10"].fillna(0).sum()) if not edited.empty else 0,
}

st.markdown("**Column Totals** (after edits)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("6 A.M", edited_sums["AM_6"], cmp_footer("AM_6"))
c2.metric("11 A.M", edited_sums["AM_11"], cmp_footer("AM_11"))
c3.metric("2 P.M", edited_sums["PM_2"], cmp_footer("PM_2"))
c4.metric("5 P.M", edited_sums["PM_5"], cmp_footer("PM_5"))
c5.metric("10 P.M", edited_sums["PM_10"], cmp_footer("PM_10"))

st.markdown("**Step 3: Header**")
h1, h2, h3 = st.columns(3)
date_val = h1.text_input("Date (DD/MM/YY)", value=header.get("Date") or "")
unit_val = h2.text_input("Unit", value=header.get("Unit") or "HASTINGS JUTE MILL")
title_val = h3.text_input("Title", value=header.get("Title") or "ROLL STOCK CONSUMPTION")

header_edit = {"Date": date_val, "Unit": unit_val, "Title": title_val}

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

# XLSX (xlsxwriter fallback -> openpyxl)
xlsx_buf = io.BytesIO()
engine = "xlsxwriter"
try:
    with pd.ExcelWriter(xlsx_buf, engine=engine) as writer:
        edited.to_excel(writer, index=False, sheet_name="RollStock")
except Exception:
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
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
