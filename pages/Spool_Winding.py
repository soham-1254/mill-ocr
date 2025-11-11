# ================================================
# pages/Spool_Winding.py — Spool Winding Production OCR
# (Gemini 2.5 Flash, Mongo save, CSV/JSON/XLSX/PDF export)
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
    st.warning("⚠️ GOOGLE_API_KEY missing in .env (Google AI Studio)")

# ------------------ CONSTANTS ------------------
ROW_COLUMNS = [
    "Sl_No",
    "Quality_Mc_Alloc",
    "Spinning_Frame_No",
    "Winding_Frame_No",
    "Labour_No",
    "Production_Per_Fera",
    "Production_Sum",
    "Net_Weight",
    "Remarks",
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
        out = []
        for v in x:
            try:
                out.append(int(str(v).strip()))
            except:
                continue
        return out
    nums = re.findall(r"\d+", str(x))
    return [int(n) for n in nums]

def normalize_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    norm = []
    for r in rows:
        row = {
            "Sl_No": to_int(r.get("Sl_No")),
            "Quality_Mc_Alloc": r.get("Quality_Mc_Alloc") or r.get("Quality") or r.get("Quality/Mc_Alloc"),
            "Spinning_Frame_No": str(r.get("Spinning_Frame_No") or r.get("Spinning Frame No.") or r.get("SpinningNo") or ""),
            "Winding_Frame_No": str(r.get("Winding_Frame_No") or r.get("Winding Frame No.") or r.get("WindingNo") or ""),
            "Labour_No": to_int(r.get("Labour_No") or r.get("Labour Number") or r.get("Labour")),
            "Production_Per_Fera": to_int_list(r.get("Production_Per_Fera") or r.get("Production per feras") or r.get("Production")),
            "Net_Weight": to_int(r.get("Net_Weight") or r.get("Net Weight")),
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

    prompt = f"""
You are parsing a **Spool Winding Production** register page.

Top header fields:
- Date, Shift, Hands, Number_of_Frames, Unit, Title

Then read the table with columns:
{ROW_COLUMNS}

Return JSON:
{{
  "header": {{
    "Date": "...", "Shift": "...", "Hands": int, "Number_of_Frames": int,
    "Unit": str, "Title": str
  }},
  "rows": [...]
}}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}],
            generation_config=cfg
        )
        return json_safe_load(resp.text) or {"header": {}, "rows": []}
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header": {}, "rows": []}

# ------------------ PDF EXPORT ------------------
def export_pdf(df: pd.DataFrame, header: Dict[str, Any]) -> bytes:
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    font_path = os.path.join(os.path.dirname(__file__), "..", "NotoSans-Regular.ttf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
        r = requests.get(url)
        open(font_path, "wb").write(r.content)

    pdf.add_font("NotoSans", "", font_path, uni=True)
    pdf.set_font("NotoSans", "", 14)
    pdf.cell(0, 8, "Spool Winding Production — OCR Extract", ln=1)
    pdf.set_font("NotoSans", "", 11)
    pdf.cell(0, 7, f"Date: {header.get('Date')}   Shift: {header.get('Shift')}   Hands: {header.get('Hands')}   Frames: {header.get('Number_of_Frames')}", ln=1)
    pdf.ln(3)

    show_cols = ["Sl_No","Quality_Mc_Alloc","Spinning_Frame_No","Winding_Frame_No","Labour_No","Production_Sum","Net_Weight","Remarks"]
    col_w = [10, 28, 24, 24, 20, 26, 22, 60]

    pdf.set_font("NotoSans","",8)
    for i,c in enumerate(show_cols):
        pdf.cell(col_w[i], 6, c, border=1, align="C")
    pdf.ln()

    for _, r in df.iterrows():
        row = [str(r.get(c,""))[:18] for c in show_cols]
        for i,val in enumerate(row):
            pdf.cell(col_w[i], 6, val, border=1)
        pdf.ln()
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

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
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True, return_document=ReturnDocument.AFTER)

# ------------------ UI ------------------
st.markdown("<h2 style='color:#00B4D8;'>🧶 Spool Winding Production — OCR</h2>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.subheader("📤 Upload Input")
    cam = st.camera_input("📸 Capture Image (optional)")
    up = st.file_uploader("📁 Upload Register Image", type=["png","jpg","jpeg"])

img_bytes = img_name = mime = None
if cam:
    img_bytes = cam.getvalue(); img_name = f"cam_{dt.datetime.utcnow().isoformat()}.jpg"; mime = "image/jpeg"
elif up:
    img_bytes = up.getvalue(); img_name = up.name; mime = up.type

if not img_bytes:
    st.info("📸 Please upload or capture a register image to begin OCR extraction.")
    st.stop()

st.image(img_bytes, caption="Input Image Preview", use_container_width=True)
st.markdown("**Step 1:** Extracting data from image using Gemini…")
data = call_gemini_for_spool(img_bytes, mime)

header = data.get("header", {}) or {}
rows = data.get("rows", []) or []
df = normalize_rows(rows)

# ------------------ Editor ------------------
st.markdown("**Step 2:** Review & Edit Extracted Data")
df["Spinning_Frame_No"] = df["Spinning_Frame_No"].astype(str)
df["Winding_Frame_No"] = df["Winding_Frame_No"].astype(str)
df_preview = df.copy()
df_preview["Production_Per_Fera"] = df_preview["Production_Per_Fera"].apply(lambda x: ", ".join(map(str, x)) if x else "")

edited = st.data_editor(
    df_preview,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Sl_No": st.column_config.NumberColumn("Sl No."),
        "Quality_Mc_Alloc": st.column_config.TextColumn("Quality / Mc Alloc"),
        "Spinning_Frame_No": st.column_config.TextColumn("Spinning Frame No."),
        "Winding_Frame_No": st.column_config.TextColumn("Winding Frame No."),
        "Labour_No": st.column_config.NumberColumn("Labour No."),
        "Production_Per_Fera": st.column_config.TextColumn("Production per Fera"),
        "Production_Sum": st.column_config.NumberColumn("Production Sum"),
        "Net_Weight": st.column_config.NumberColumn("Net Weight"),
        "Remarks": st.column_config.TextColumn("Remarks"),
    },
    key="spool_editor"
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
cA, cB, cC, cD = st.columns(4)

if cA.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(header_edit, edited_df, img_name, img_bytes)
    st.success("✅ Data Saved to MongoDB Successfully!")
    st.json({"_id": str(saved.get("_id")), "Date": header_edit["Date"]})

csv_bytes = edited_df.to_csv(index=False).encode()
cB.download_button("⬇️ CSV", csv_bytes, "spool_winding.csv", "text/csv")

json_bytes = edited_df.to_json(orient="records", indent=2).encode()
cC.download_button("⬇️ JSON", json_bytes, "spool_winding.json", "application/json")

xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    edited_df.to_excel(writer, index=False, sheet_name="SpoolWinding")
cD.download_button("⬇️ XLSX", xlsx_buf.getvalue(), "spool_winding.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

pdf_bytes = export_pdf(edited_df, header_edit)
st.download_button("⬇️ PDF", pdf_bytes, "spool_winding.pdf", "application/pdf")

st.markdown("---")
st.caption("💡 Tip: Edit data in the grid before saving or exporting for best accuracy.")
