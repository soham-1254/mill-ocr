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
# keep the collection name that is already working for you
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
NUMERIC_COLS = ["Frame_A", "Frame_B", "Frame_C",
                "Production_A", "Production_B", "Production_C"]

SUMMARY_KEYS = [
    "Total_Frames_A", "Total_Frames_B", "Total_Frames_C", "Total_Frames",
    "Total_Prod_A", "Total_Prod_B", "Total_Prod_C", "Total_Prod",
    "FS", "CS", "Avg_kg_per_frame"
]

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
        if x in [None, "", "null", "-", "–", "—"]:
            return None
        return int(str(x).strip())
    except Exception:
        return None

def to_float(x):
    try:
        if x in [None, "", "null", "-", "–", "—"]:
            return None
        return float(str(x).strip())
    except Exception:
        return None

# ------------------ NORMALIZATION ------------------
def normalize_rows(rows: list) -> pd.DataFrame:
    """
    Normalize OCR rows.
    - Handles ',,' as "same as previous" for Frame_* and Production_* columns.
    - DOES NOT add any extra 'Total' row; only uses what Gemini returns.
    """
    norm = []

    prev = {
        "Frame_A": None,
        "Frame_B": None,
        "Frame_C": None,
        "Production_A": None,
        "Production_B": None,
        "Production_C": None,
    }

    def repeat_prev(value, key):
        # two commas, empty, quotes etc. mean "same as previous"
        if value is None:
            s = ""
        else:
            s = str(value).strip()
        if s in ["", ",,", "''", '"', '""']:
            return prev[key]
        return value

    for r in rows or []:
        # apply "same as previous" logic to numeric columns
        fA_raw = repeat_prev(r.get("Frame_A"), "Frame_A")
        fB_raw = repeat_prev(r.get("Frame_B"), "Frame_B")
        fC_raw = repeat_prev(r.get("Frame_C"), "Frame_C")
        pA_raw = repeat_prev(r.get("Production_A"), "Production_A")
        pB_raw = repeat_prev(r.get("Production_B"), "Production_B")
        pC_raw = repeat_prev(r.get("Production_C"), "Production_C")

        prev["Frame_A"] = fA_raw
        prev["Frame_B"] = fB_raw
        prev["Frame_C"] = fC_raw
        prev["Production_A"] = pA_raw
        prev["Production_B"] = pB_raw
        prev["Production_C"] = pC_raw

        row = {
            "Sl_No": to_int(r.get("Sl_No")),
            "Quality": (r.get("Quality") or "").strip(),
            "Frame_A": to_int(fA_raw),
            "Frame_B": to_int(fB_raw),
            "Frame_C": to_int(fC_raw),
            "Production_A": to_int(pA_raw),
            "Production_B": to_int(pB_raw),
            "Production_C": to_int(pC_raw),
        }
        norm.append(row)

    df = pd.DataFrame(norm, columns=ROW_COLUMNS)

    # ensure numeric dtype (no extra totals added)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def normalize_summary(summary_raw: dict) -> dict:
    """Normalize summary block from Gemini; no calculations, just cleaning."""
    summary = {k: None for k in SUMMARY_KEYS}
    if not isinstance(summary_raw, dict):
        return summary

    summary["Total_Frames_A"] = to_int(summary_raw.get("Total_Frames_A"))
    summary["Total_Frames_B"] = to_int(summary_raw.get("Total_Frames_B"))
    summary["Total_Frames_C"] = to_int(summary_raw.get("Total_Frames_C"))
    summary["Total_Frames"]   = to_int(summary_raw.get("Total_Frames"))

    summary["Total_Prod_A"] = to_int(summary_raw.get("Total_Prod_A"))
    summary["Total_Prod_B"] = to_int(summary_raw.get("Total_Prod_B"))
    summary["Total_Prod_C"] = to_int(summary_raw.get("Total_Prod_C"))
    summary["Total_Prod"]   = to_int(summary_raw.get("Total_Prod"))

    summary["FS"] = to_int(summary_raw.get("FS"))
    summary["CS"] = to_int(summary_raw.get("CS"))
    summary["Avg_kg_per_frame"] = to_float(summary_raw.get("Avg_kg_per_frame"))
    return summary

# ------------------ GEMINI OCR ------------------
def call_gemini_for_spinning(image_bytes: bytes, mime_type: str) -> dict:
    """Gemini 2.5 Flash OCR for Spinning Production."""
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {"header": {}, "rows": [], "summary": {}}

    prompt = """
You are extracting from a Spinning Production register.

TABLE PART:
- Columns (left to right) are:
  Quality, Frame A, Frame B, Frame C, Production A, Production B, Production C.
- "Sl_No" is the running serial number of the rows and may appear as 1,2,3,...; if it is not in the page, you may infer from order.
- Some cells are written as ",," meaning "SAME AS PREVIOUS ROW" for that column. Keep the SAME number, do NOT change it.
- DO NOT add your own "Total" row; if the page already has a total row, just return it like any other row.

SUMMARY PART (BOTTOM OF PAGE):
Extract the numbers for:
- A-shift frames total, B-shift frames total, C-shift frames total, and their grand total.
- A-shift production total, B-shift production total, C-shift production total, and their grand total.
- F/S (frames spun) value.
- C/S (cop spun or conversion) value.
- Average production per frame in kg/frame.

Return STRICT JSON ONLY in this shape:

{
  "header": {
    "Date": "DD/MM/YY or DD/MM/YYYY",
    "Supervisor_Signature": "Name"
  },
  "rows": [
    {
      "Sl_No": int,
      "Quality": str,
      "Frame_A": int,
      "Frame_B": int,
      "Frame_C": int,
      "Production_A": int,
      "Production_B": int,
      "Production_C": int
    }
  ],
  "summary": {
    "Total_Frames_A": int,
    "Total_Frames_B": int,
    "Total_Frames_C": int,
    "Total_Frames": int,
    "Total_Prod_A": int,
    "Total_Prod_B": int,
    "Total_Prod_C": int,
    "Total_Prod": int,
    "FS": int,
    "CS": int,
    "Avg_kg_per_frame": float
  }
}

IMPORTANT:
- Copy the totals/F-S/C-S/Avg exactly from the book. Do NOT recompute anything.
- Do not include any text outside this JSON.
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}],
            generation_config=cfg,
        )
        return json_safe_load(resp.text) or {"header": {}, "rows": [], "summary": {}}
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header": {}, "rows": [], "summary": {}}

# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: dict, df: pd.DataFrame, summary: dict,
                 img_name: str, raw_bytes: bytes):
    # store rows exactly as edited (including any Total row from book)
    doc = {
        "register_type": "Spinning Production",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": img_name,
        "extracted_data": df.to_dict(orient="records"),
        "summary": summary,
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
header_raw = data.get("header", {}) or {}
rows = data.get("rows", []) or []
summary_raw = data.get("summary", {}) or {}

df = normalize_rows(rows)
summary = normalize_summary(summary_raw)

# ------------------ DISPLAY TABLE ------------------
st.markdown("**Step 2: Preview & Edit Table**")
edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="spg_table")

# ------------------ SUMMARY (EDITABLE) ------------------
st.markdown("**Step 2B: Totals & Summary (Editable)**")

s1, s2, s3, s4 = st.columns(4)
total_frames_a = s1.number_input(
    "Frames A total", value=summary["Total_Frames_A"] or 0, step=1
)
total_frames_b = s2.number_input(
    "Frames B total", value=summary["Total_Frames_B"] or 0, step=1
)
total_frames_c = s3.number_input(
    "Frames C total", value=summary["Total_Frames_C"] or 0, step=1
)
total_frames = s4.number_input(
    "Total Frames (A+B+C)", value=summary["Total_Frames"] or 0, step=1
)

p1, p2, p3, p4 = st.columns(4)
total_prod_a = p1.number_input(
    "Production A total", value=summary["Total_Prod_A"] or 0, step=1
)
total_prod_b = p2.number_input(
    "Production B total", value=summary["Total_Prod_B"] or 0, step=1
)
total_prod_c = p3.number_input(
    "Production C total", value=summary["Total_Prod_C"] or 0, step=1
)
total_prod = p4.number_input(
    "Total Production (A+B+C)", value=summary["Total_Prod"] or 0, step=1
)

q1, q2, q3 = st.columns(3)
fs_val = q1.number_input("F/S", value=summary["FS"] or 0, step=1)
cs_val = q2.number_input("C/S", value=summary["CS"] or 0, step=1)
avg_val = q3.number_input(
    "Avg kg/frame", value=summary["Avg_kg_per_frame"] or 0.0, step=0.1, format="%.2f"
)

summary_edit = {
    "Total_Frames_A": int(total_frames_a),
    "Total_Frames_B": int(total_frames_b),
    "Total_Frames_C": int(total_frames_c),
    "Total_Frames": int(total_frames),
    "Total_Prod_A": int(total_prod_a),
    "Total_Prod_B": int(total_prod_b),
    "Total_Prod_C": int(total_prod_c),
    "Total_Prod": int(total_prod),
    "FS": int(fs_val),
    "CS": int(cs_val),
    "Avg_kg_per_frame": float(avg_val),
}

# ------------------ HEADER ------------------
st.markdown("**Step 3: Header Details**")
c1, c2 = st.columns(2)
date_val = c1.text_input("Date", value=header_raw.get("Date") or "")
sup_val = c2.text_input("Supervisor Signature", value=header_raw.get("Supervisor_Signature") or "")
header_edit = {"Date": date_val, "Supervisor_Signature": sup_val}

# ------------------ SAVE / EXPORT ------------------
st.markdown("**Step 4: Save & Export**")
cA, cB, cC = st.columns(3)

if cA.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(header_edit, edited, summary_edit, img_name, img_bytes)
    st.success("✅ Saved to MongoDB")
    st.json({
        "_id": str(saved.get("_id")),
        "Date": header_edit["Date"],
        "Total_Prod": summary_edit["Total_Prod"],
        "Total_Frames": summary_edit["Total_Frames"],
    })

# CSV – only row data (no header/summary)
csv_bytes = edited.to_csv(index=False).encode()
cB.download_button("⬇️ CSV (rows only)", data=csv_bytes,
                   file_name="spinning_production_rows.csv", mime="text/csv")

# JSON – header + rows + summary in one object
full_json = json.dumps(
    {
        "header": header_edit,
        "rows": edited.to_dict(orient="records"),
        "summary": summary_edit,
    },
    indent=2,
).encode()
cC.download_button("⬇️ JSON (full)", data=full_json,
                   file_name="spinning_production_full.json",
                   mime="application/json")

# XLSX – Header, Data, Summary sheets
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    pd.DataFrame([header_edit]).to_excel(writer, index=False, sheet_name="Header")
    edited.to_excel(writer, index=False, sheet_name="Data")
    pd.DataFrame([summary_edit]).to_excel(writer, index=False, sheet_name="Summary")

st.download_button(
    "⬇️ XLSX (Header+Data+Summary)",
    data=xlsx_buf.getvalue(),
    file_name="spinning_production.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")
st.caption("Summary values are taken from the book (no auto-calculation) but remain fully editable.")
