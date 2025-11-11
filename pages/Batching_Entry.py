# ================================================
# pages/Batching_Entry.py — Batching Entry OCR page
# (Gemini 2.5 Flash • Mongo • CSV/JSON/XLSX/PDF)
# ================================================
import os, io, re, json, datetime as dt, requests
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai
from fpdf import FPDF

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Batching Entry OCR", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")
COLL_NAME = os.getenv("BATCHING_COLLECTION", "batching_entries")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY missing in .env (Google AI Studio)")

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

def parse_num(x):
    """
    Clean cell values:
      - 'x' or 'X' -> 0
      - '2:1' -> 2.1
      - '4:50' -> 4.50
      - keep integers/floats
    Returns float or None.
    """
    if x is None:
        return None
    s = str(x).strip()
    if s.lower() == "x":
        return 0.0
    # replace colon with dot for decimals
    s = s.replace(":", ".")
    # keep digits, dot, minus
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in ("", ".", "-"):
        return None
    try:
        return float(s)
    except Exception:
        return None

def to_int(x):
    try:
        if x in [None, "", "null"]:
            return None
        return int(float(str(x)))
    except Exception:
        return None

# ------------------ NORMALIZERS ------------------
def normalize_machine(df_like) -> pd.DataFrame:
    """
    Expect rows like: Mc, A, B, C
    Machines seen in your sheet:
      Spreader, Softner, Inter Spreader, Cutting, Ropes & Habijab
    """
    rows = []
    for r in df_like:
        rows.append({
            "Mc": r.get("Mc"),
            "A": parse_num(r.get("A")),
            "B": parse_num(r.get("B")),
            "C": parse_num(r.get("C")),
        })
    df = pd.DataFrame(rows, columns=["Mc","A","B","C"])
    # totals per row & column
    for col in ["A","B","C"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Row_Total"] = df[["A","B","C"]].sum(axis=1, skipna=True)
    totals = pd.DataFrame([{
        "Mc": "TOTAL",
        "A": df["A"].sum(skipna=True),
        "B": df["B"].sum(skipna=True),
        "C": df["C"].sum(skipna=True),
        "Row_Total": df["Row_Total"].sum(skipna=True),
    }])
    return pd.concat([df, totals], ignore_index=True)

def normalize_production(rows_like) -> pd.DataFrame:
    # Expect: Item, MT
    rows = []
    for r in rows_like:
        rows.append({
            "Item": r.get("Item"),
            "MT": parse_num(r.get("MT"))
        })
    df = pd.DataFrame(rows, columns=["Item","MT"])
    df.loc[df["Item"].str.strip().str.lower().eq("total"), "MT"] = df["MT"].sum(skipna=True)
    return df

def normalize_abc_table(rows_like, key_name: str) -> pd.DataFrame:
    """
    For Pile Made (TON) and Roll Made (EA)
    Expect rows: {key_name: 'BTR' ...}, A, B, C
    Adds Row_Total and footer totals.
    """
    rows = []
    for r in rows_like:
        rows.append({
            key_name: r.get(key_name),
            "A": parse_num(r.get("A")),
            "B": parse_num(r.get("B")),
            "C": parse_num(r.get("C")),
        })
    df = pd.DataFrame(rows, columns=[key_name,"A","B","C"])
    for col in ["A","B","C"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Row_Total"] = df[["A","B","C"]].sum(axis=1, skipna=True)
    totals = pd.DataFrame([{
        key_name: "TOTAL",
        "A": df["A"].sum(skipna=True),
        "B": df["B"].sum(skipna=True),
        "C": df["C"].sum(skipna=True),
        "Row_Total": df["Row_Total"].sum(skipna=True),
    }])
    return pd.concat([df, totals], ignore_index=True)

# ------------------ GEMINI OCR ------------------
def call_gemini_for_batching(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Extract this structure from the Batching Entry sheet:
    - header: Date, Shift, Unit, Title
    - machine_allocation: list of {Mc, A, B, C}
    - production_mt: list of {Item, MT}  (Items: "Ropes & Habijab", "Cutting", "Total")
    - pile_made_ton: list of {Qty, A, B, C}
    - roll_made_ea: list of {Qty, A, B, C}
    Values may be 'x' or contain ':' which should be kept in text; we'll normalize later.
    """
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {"header": {}, "machine_allocation": [], "production_mt": [], "pile_made_ton": [], "roll_made_ea": []}

    prompt = """
You are reading a **BATCHING ENTRY** register page.

Return STRICT JSON ONLY with this schema:
{
  "header": {
    "Date": "DD/MM/YY or DD/MM/YYYY",
    "Shift": "A/B/C or -",
    "Unit": "text or null",
    "Title": "BATCHING ENTRY"
  },
  "machine_allocation": [
    {"Mc": "Spreader", "A": "text", "B": "text", "C": "text"},
    {"Mc": "Softner", "A": "text", "B": "text", "C": "text"},
    {"Mc": "Inter Spreader", "A": "text", "B": "text", "C": "text"},
    {"Mc": "Cutting", "A": "text", "B": "text", "C": "text"},
    {"Mc": "Ropes & Habijab", "A": "text", "B": "text", "C": "text"}
  ],
  "production_mt": [
    {"Item": "Ropes & Habijab", "MT": "text"},
    {"Item": "Cutting", "MT": "text"},
    {"Item": "Total", "MT": "text or empty"}
  ],
  "pile_made_ton": [
    {"Qty": "BTR", "A": "text", "B": "text", "C": "text"},
    {"Qty": "...",  "A": "text", "B": "text", "C": "text"}
  ],
  "roll_made_ea": [
    {"Qty": "P", "A": "text", "B": "text", "C": "text"},
    {"Qty": "O", "A": "text", "B": "text", "C": "text"},
    {"Qty": "T", "A": "text", "B": "text", "C": "text"},
    {"Qty": "J", "A": "text", "B": "text", "C": "text"},
    {"Qty": "B", "A": "text", "B": "text", "C": "text"}
  ]
}

Guidelines:
- Read each small table separately.
- Keep literal cell content as text (e.g., 'x', '4:50', '7:05'); DO NOT compute.
- The Pile Made 'TOTAL' box (bottom right) should be left for client computation; do not invent. 
- If a cell is blank, return "".
Only return the JSON, no explanations.
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}],
            generation_config=cfg
        )
        return json_safe_load(resp.text) or {"header": {}, "machine_allocation": [], "production_mt": [], "pile_made_ton": [], "roll_made_ea": []}
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header": {}, "machine_allocation": [], "production_mt": [], "pile_made_ton": [], "roll_made_ea": []}

# ------------------ PDF (Unicode-safe) ------------------
def export_pdf(header, df_machine, df_prod, df_pile, df_roll) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    font_path = os.path.join(os.path.dirname(__file__), "..", "NotoSans-Regular.ttf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
        r = requests.get(url); open(font_path, "wb").write(r.content)
    pdf.add_font("NotoSans", "", font_path, uni=True)
    pdf.set_font("NotoSans", "", 14)

    def safe(t): return re.sub(r"[—–−]", "-", str(t or ""))

    pdf.cell(0, 8, safe("Batching Entry — OCR Extract"), ln=1)
    pdf.set_font("NotoSans", "", 11)
    pdf.cell(0, 7, f"Date: {safe(header.get('Date'))}   Shift: {safe(header.get('Shift'))}", ln=1)
    pdf.ln(2)

    def table(df: pd.DataFrame, title: str, widths: list):
        pdf.set_font("NotoSans", "", 11); pdf.cell(0, 7, title, ln=1)
        pdf.set_font("NotoSans", "", 8)
        for i, c in enumerate(df.columns):
            pdf.cell(widths[i], 6, str(c), border=1, align="C")
        pdf.ln()
        for _, r in df.iterrows():
            for i, c in enumerate(df.columns):
                pdf.cell(widths[i], 6, str(r[c])[:20], border=1)
            pdf.ln()
        pdf.ln(2)

    table(df_machine, "Machine Allocation", [40,20,20,20,24])
    table(df_prod, "Production (MT)", [60,30])
    table(df_pile, "Pile Made (TON)", [35,20,20,20,24])
    table(df_roll, "Roll Made (EA)", [35,20,20,20,24])

    return pdf.output(dest="S").encode("latin-1", errors="ignore")

# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: Dict[str, Any],
                 df_machine: pd.DataFrame,
                 df_prod: pd.DataFrame,
                 df_pile: pd.DataFrame,
                 df_roll: pd.DataFrame,
                 image_name: str,
                 raw_bytes: bytes):
    doc = {
        "register_type": "Batching Entry",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": image_name,
        "machine_allocation": df_machine.to_dict(orient="records"),
        "production_mt": df_prod.to_dict(orient="records"),
        "pile_made_ton": df_pile.to_dict(orient="records"),
        "roll_made_ea": df_roll.to_dict(orient="records"),
        "validated": False,
    }
    key = {"original_image_name": image_name, "header.Date": header.get("Date")}
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True, return_document=ReturnDocument.AFTER)

# ------------------ UI ------------------
st.title("🧪 Batching Entry — OCR")

with st.sidebar:
    st.subheader("Input")
    cam = st.camera_input("📸 Capture Image (optional)")
    up = st.file_uploader("📁 Upload Image", type=["png","jpg","jpeg"])

img_bytes = img_name = mime = None
if cam:
    img_bytes = cam.getvalue(); img_name = f"cam_{dt.datetime.utcnow().isoformat()}.jpg"; mime = "image/jpeg"
elif up:
    img_bytes = up.getvalue(); img_name = up.name; mime = up.type

if not img_bytes:
    st.info("Upload or capture a Batching Entry sheet image to start.")
    st.stop()

st.image(img_bytes, caption="Input Image", use_container_width=True)

st.markdown("**Step 1:** Extracting with Gemini…")
data = call_gemini_for_batching(img_bytes, mime)

header_raw = data.get("header", {}) or {}
machine_raw = data.get("machine_allocation", []) or []
prod_raw = data.get("production_mt", []) or []
pile_raw = data.get("pile_made_ton", []) or []
roll_raw = data.get("roll_made_ea", []) or []

# Normalize
df_machine = normalize_machine(machine_raw)
df_prod = normalize_production(prod_raw)
df_pile = normalize_abc_table(pile_raw, "Qty")
df_roll = normalize_abc_table(roll_raw, "Qty")

# ------------------ Editors ------------------
st.markdown("**Step 2: Review & Edit**")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Machine Allocation")
    df_machine_edit = st.data_editor(df_machine, use_container_width=True, num_rows="dynamic", key="mach")

with c2:
    st.subheader("Production (MT)")
    df_prod_edit = st.data_editor(df_prod, use_container_width=True, num_rows="dynamic", key="prod")

c3, c4 = st.columns(2)
with c3:
    st.subheader("Pile Made (TON)")
    df_pile_edit = st.data_editor(df_pile, use_container_width=True, num_rows="dynamic", key="pile")

with c4:
    st.subheader("Roll Made (EA)")
    df_roll_edit = st.data_editor(df_roll, use_container_width=True, num_rows="dynamic", key="roll")

# Header
st.markdown("**Step 3: Header**")
h1, h2, h3, h4 = st.columns(4)
date_val = h1.text_input("Date", value=header_raw.get("Date") or "")
shift_val = h2.text_input("Shift", value=header_raw.get("Shift") or "")
unit_val  = h3.text_input("Unit",  value=header_raw.get("Unit") or "HASTINGS JUTE MILL")
title_val = h4.text_input("Title", value=header_raw.get("Title") or "BATCHING ENTRY")

header = {"Date": date_val, "Shift": shift_val, "Unit": unit_val, "Title": title_val}

# ------------------ Save / Export ------------------
st.markdown("**Step 4: Save / Export**")
b1, b2, b3, b4, b5 = st.columns(5)

if b1.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(header, df_machine_edit, df_prod_edit, df_pile_edit, df_roll_edit, img_name, img_bytes)
    st.success("✅ Saved to MongoDB")
    st.json({"_id": str(saved.get("_id")), "Date": header["Date"]})

# CSV (zipped into one CSV per section would be messy; export separate)
csv_zip = {
    "machine_allocation.csv": df_machine_edit.to_csv(index=False),
    "production_mt.csv": df_prod_edit.to_csv(index=False),
    "pile_made_ton.csv": df_pile_edit.to_csv(index=False),
    "roll_made_ea.csv": df_roll_edit.to_csv(index=False),
}
# Single button for CSVs is optional; we keep per-section formats below.

# JSON
json_bytes = json.dumps({
    "header": header,
    "machine_allocation": df_machine_edit.to_dict(orient="records"),
    "production_mt": df_prod_edit.to_dict(orient="records"),
    "pile_made_ton": df_pile_edit.to_dict(orient="records"),
    "roll_made_ea": df_roll_edit.to_dict(orient="records"),
}, indent=2).encode()
b2.download_button("⬇️ JSON", json_bytes, "batching_entry.json", "application/json")

# XLSX (multi-sheet)
xlsx_buf = io.BytesIO()
try:
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        pd.DataFrame([header]).to_excel(writer, index=False, sheet_name="Header")
        df_machine_edit.to_excel(writer, index=False, sheet_name="MachineAllocation")
        df_prod_edit.to_excel(writer, index=False, sheet_name="ProductionMT")
        df_pile_edit.to_excel(writer, index=False, sheet_name="PileMadeTON")
        df_roll_edit.to_excel(writer, index=False, sheet_name="RollMadeEA")
except Exception:
    # Fallback to openpyxl if xlsxwriter is missing
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        pd.DataFrame([header]).to_excel(writer, index=False, sheet_name="Header")
        df_machine_edit.to_excel(writer, index=False, sheet_name="MachineAllocation")
        df_prod_edit.to_excel(writer, index=False, sheet_name="ProductionMT")
        df_pile_edit.to_excel(writer, index=False, sheet_name="PileMadeTON")
        df_roll_edit.to_excel(writer, index=False, sheet_name="RollMadeEA")
b3.download_button(
    "⬇️ XLSX",
    xlsx_buf.getvalue(),
    "batching_entry.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# PDF
pdf_bytes = export_pdf(header, df_machine_edit, df_prod_edit, df_pile_edit, df_roll_edit)
b4.download_button("⬇️ PDF", pdf_bytes, "batching_entry.pdf", "application/pdf")

# Individual CSVs (optional quick exports)
b5.download_button("⬇️ Machine CSV", df_machine_edit.to_csv(index=False).encode(), "machine_allocation.csv", "text/csv")
st.download_button("⬇️ Production CSV", df_prod_edit.to_csv(index=False).encode(), "production_mt.csv", "text/csv")
st.download_button("⬇️ Pile CSV", df_pile_edit.to_csv(index=False).encode(), "pile_made_ton.csv", "text/csv")
st.download_button("⬇️ Roll CSV", df_roll_edit.to_csv(index=False).encode(), "roll_made_ea.csv", "text/csv")

st.markdown("---")
st.caption("Notes: 'x' is treated as 0. Values like 4:50 are parsed as 4.50. Edit any cell before saving/exporting.")
