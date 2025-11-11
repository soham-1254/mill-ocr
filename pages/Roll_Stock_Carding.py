import os
import io
import json
import datetime as dt
import pandas as pd
import streamlit as st
from pymongo import MongoClient, ReturnDocument
from fpdf import FPDF
from dotenv import load_dotenv
import google.generativeai as genai
import re


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
            "Qty": r.get("Qty"),
            "6AM": to_int(r.get("6AM")),
            "11AM": to_int(r.get("11AM")),
            "2PM": to_int(r.get("2PM")),
            "5PM": to_int(r.get("5PM")),
            "10PM": to_int(r.get("10PM")),
            "Remarks": r.get("Remarks"),
            "K_Cutting": r.get("K_Cutting", ""),  # Extracted separately
            "L_Cutting": r.get("L_Cutting", ""),  # Extracted separately
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

    prompt = f"""
    You are extracting rows from a **Roll Stock Carding** register page.

    Extract the following fields:
    Header: Date, Shift, Supervisor Signature, and other metadata if available.

    Extract rows as follows:
    Sl_No, Qty, 6AM, 11AM, 2PM, 5PM, 10PM, Remarks

    K_Cutting and L_Cutting are independent and should be extracted separately from the regular columns.

    Return valid JSON as follows:
    {{

        "header": {{
            "Date": "DD/MM/YY or DD/MM/YYYY",
            "Shift": "A/B/C",
            "Supervisor_Signature": "Name"
        }},
        "rows": [
            {{
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
            }}
        ]
    }}
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


# ------------------ PDF EXPORT ------------------
def export_pdf(df: pd.DataFrame, header: dict) -> bytes:
    """Export the DataFrame as a PDF"""
    pdf = FPDF()
    pdf.add_page()
    font_path = os.path.join(os.path.dirname(__file__), "..", "NotoSans-Regular.ttf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
        r = requests.get(url)
        open(font_path, "wb").write(r.content)
    pdf.add_font("NotoSans", "", font_path, uni=True)
    pdf.set_font("NotoSans", "", 14)

    def safe(t): return re.sub(r"[—–−]", "-", str(t or ""))

    pdf.cell(0, 8, safe("Roll Stock Carding - OCR Extract"), ln=1)
    pdf.set_font("NotoSans", "", 11)
    pdf.cell(0, 7, f"Date: {safe(header.get('Date'))}   Shift: {safe(header.get('Shift'))}   Supervisor: {safe(header.get('Supervisor_Signature'))}", ln=1)
    pdf.ln(2)

    show_cols = ["Sl_No", "Qty", "6AM", "11AM", "2PM", "5PM", "10PM", "Remarks", "K_Cutting", "L_Cutting"]
    col_w = [10, 20, 25, 25, 25, 30, 35, 40, 30, 30]  # Make sure this length matches the show_cols length

    pdf.set_font("NotoSans", "", 8)
    for i, c in enumerate(show_cols):
        pdf.cell(col_w[i], 6, c, border=1, align="C")
    pdf.ln()

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

    return pdf.output(dest="S").encode("latin-1", errors="ignore")


# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: dict, df: pd.DataFrame, img_name: str, raw_bytes: bytes):
    """Upsert data to MongoDB"""
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
st.markdown("**Step 2: Preview & Edit**")

# Show data preview
edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")

# Step 3: Header
date_val = st.text_input("Date (DD/MM/YY)", value=header.get("Date") or "")
shift_val = st.text_input("Shift", value=header.get("Shift") or "")
supervisor_val = st.text_input("Supervisor Signature", value=header.get("Supervisor_Signature") or "")
header_edit = {
    "Date": date_val,
    "Shift": shift_val,
    "Supervisor_Signature": supervisor_val,
}

# Step 4: Save and Export
if st.button("💾 Save to MongoDB"):
    saved = upsert_mongo(header_edit, edited, img_name, img_bytes)
    st.success("✅ Saved to MongoDB")
    st.json({"_id": str(saved.get("_id")), "Date": header_edit["Date"]})

cA, cB, cC, cD = st.columns(4)

# CSV
csv_bytes = edited.to_csv(index=False).encode()
cB.download_button("⬇️ CSV", data=csv_bytes, file_name="roll_stock_data.csv", mime="text/csv")

# JSON
json_bytes = edited.to_json(orient="records", indent=2).encode()
cC.download_button("⬇️ JSON", data=json_bytes, file_name="roll_stock_data.json", mime="application/json")

# XLSX
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    edited.to_excel(writer, index=False, sheet_name="RollStock")
cD.download_button("⬇️ XLSX", data=xlsx_buf.getvalue(), file_name="roll_stock_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# PDF
pdf_bytes = export_pdf(edited, header_edit)
st.download_button("⬇️ PDF", data=pdf_bytes, file_name="roll_stock_data.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Tip: You can refine values in the grid before saving or exporting.")
