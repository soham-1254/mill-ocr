# ======================================================
# pages/Drawing_Meter_Reading.py — Drawing Meter OCR Page
# (Gemini 2.5 Flash + Mongo + PDF with /tmp Font via utils.pdf_utils)
# ======================================================
import os, io, json, re, datetime as dt
import pandas as pd
import streamlit as st
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai
from utils.pdf_utils import get_pdf_base   # ✅ centralized, cloud-safe PDF base

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Drawing Meter Reading OCR", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")
COLL_NAME = os.getenv("COLLECTION_NAME", "drawing_meter_entries")
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
    "Sl_No",
    "Mc_No",
    "Efficiency_at_100%",
    "Opening_Meter_Reading",
    "Closing_Meter_Reading",
    "Difference",
    "Efficiency",
    "Worker_Name"
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

def to_float(x):
    try:
        if x in [None, "", "null"]:
            return None
        return float(str(x).strip())
    except Exception:
        return None

def normalize_rows(rows: list) -> pd.DataFrame:
    """Normalize rows data and compute necessary fields."""
    norm = []
    for r in rows:
        row = {
            "Sl_No": to_int(r.get("Sl_No")),
            "Mc_No": r.get("Mc_No"),
            "Efficiency_at_100%": to_float(r.get("Efficiency_at_100%")),
            "Opening_Meter_Reading": to_int(r.get("Opening_Meter_Reading")),
            "Closing_Meter_Reading": to_int(r.get("Closing_Meter_Reading")),
            "Worker_Name": r.get("Worker_Name"),
        }
        if row["Opening_Meter_Reading"] is not None and row["Closing_Meter_Reading"] is not None:
            row["Difference"] = row["Closing_Meter_Reading"] - row["Opening_Meter_Reading"]
        else:
            row["Difference"] = None
        row["Efficiency"] = to_float(r.get("Efficiency"))
        norm.append(row)

    df = pd.DataFrame(norm, columns=ROW_COLUMNS)
    return df

# ------------------ GEMINI OCR ------------------
def call_gemini_for_drawing(image_bytes: bytes, mime_type: str) -> dict:
    """Call Gemini API for Drawing Meter OCR extraction"""
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {"header": {}, "rows": []}

    prompt = """
You are reading a Drawing Meter Reading register page.

Return JSON:
{
  "header": {
    "Date": "DD/MM/YY or DD/MM/YYYY",
    "Shift": "A/B/C",
    "Supervisor_Signature": "Name"
  },
  "rows": [
    {
      "Sl_No": int,
      "Mc_No": str,
      "Efficiency_at_100%": float,
      "Opening_Meter_Reading": int,
      "Closing_Meter_Reading": int,
      "Efficiency": float,
      "Worker_Name": str
    }
  ]
}
- Compute Difference = Closing - Opening.
- Do not invent missing values.
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

# ------------------ ✅ PDF EXPORT (Centralized Font) ------------------
def export_pdf(df: pd.DataFrame, header: dict) -> bytes:
    """Cloud-safe PDF export using shared utils."""
    pdf = get_pdf_base("Drawing Meter Reading — OCR Extract", header)
    pdf.set_font("NotoSans", "", 8)

    show_cols = ["Sl_No", "Mc_No", "Efficiency_at_100%", "Opening_Meter_Reading",
                 "Closing_Meter_Reading", "Efficiency", "Worker_Name"]
    col_w = [12, 18, 28, 28, 28, 22, 40]

    # table header
    for c, w in zip(show_cols, col_w):
        pdf.cell(w, 6, c, border=1, align="C")
    pdf.ln()

    # table rows
    for _, r in df.iterrows():
        row = [
            str(r.get("Sl_No") or ""),
            str(r.get("Mc_No") or ""),
            str(r.get("Efficiency_at_100%") or ""),
            str(r.get("Opening_Meter_Reading") or ""),
            str(r.get("Closing_Meter_Reading") or ""),
            str(r.get("Efficiency") or ""),
            str(r.get("Worker_Name") or ""),
        ]
        for val, w in zip(row, col_w):
            pdf.cell(w, 6, val, border=1)
        pdf.ln()

    # ✅ FIX: handle None-returning pdf.output() safely
    out = pdf.output(dest="S")
    if out is None:
        import io
        buf = io.BytesIO()
        pdf.output(buf)
        out = buf.getvalue()

    # 🧩 FINAL FIX: ensure output is always bytes
    if isinstance(out, str):
        out = out.encode("latin-1", errors="ignore")
    elif not isinstance(out, (bytes, bytearray)):
        out = bytes(out)
    return out


# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: dict, df: pd.DataFrame, img_name: str, raw_bytes: bytes):
    doc = {
        "register_type": "Drawing Meter Reading",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": img_name,
        "extracted_data": df.to_dict(orient="records"),
        "validated": False,
    }
    key = {"original_image_name": img_name, "header.Date": header.get("Date")}
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True, return_document=ReturnDocument.AFTER)

# ------------------ UI ------------------
st.title("🧵 Drawing Meter Reading OCR")

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
    st.info("Upload or capture a Drawing Meter Reading sheet to start.")
    st.stop()

st.image(img_bytes, caption="Input Image", use_container_width=True)
st.markdown("**Step 1:** Extracting with Gemini…")

data = call_gemini_for_drawing(img_bytes, mime)
header = data.get("header", {}) or {}
rows = data.get("rows", []) or []
df = normalize_rows(rows)

# ------------------ DISPLAYING & EXPORT ------------------
st.markdown("**Step 2: Review & Edit**")
edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")

st.markdown("**Step 3: Header Details**")
c1, c2, c3 = st.columns(3)
date_val = c1.text_input("Date", value=header.get("Date") or "")
shift_val = c2.text_input("Shift", value=header.get("Shift") or "")
supervisor_val = c3.text_input("Supervisor Signature", value=header.get("Supervisor_Signature") or "")
header_edit = {"Date": date_val, "Shift": shift_val, "Supervisor_Signature": supervisor_val}

st.markdown("**Step 4: Save & Export**")

if st.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(header_edit, edited, img_name, img_bytes)
    st.success("✅ Saved to MongoDB")
    st.json({"_id": str(saved.get("_id")), "Date": header_edit["Date"]})

c1, c2, c3, c4 = st.columns(4)

csv_bytes = edited.to_csv(index=False).encode()
c1.download_button("⬇️ CSV", data=csv_bytes, file_name="drawing_meter_data.csv", mime="text/csv")

json_bytes = edited.to_json(orient="records", indent=2).encode()
c2.download_button("⬇️ JSON", data=json_bytes, file_name="drawing_meter_data.json", mime="application/json")

xlsx_buf = io.BytesIO()
try:
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        edited.to_excel(writer, index=False, sheet_name="DrawingMeter")
except Exception:
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        edited.to_excel(writer, index=False, sheet_name="DrawingMeter")
c3.download_button("⬇️ XLSX", data=xlsx_buf.getvalue(),
                   file_name="drawing_meter_data.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

pdf_bytes = export_pdf(edited, header_edit)
c4.download_button("⬇️ PDF", data=pdf_bytes,
                   file_name="drawing_meter_data.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Notes: You can review and adjust readings before saving or exporting.")
