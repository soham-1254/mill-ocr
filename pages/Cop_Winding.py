# ======================================================
# app.py — Mill Register OCR Dashboard
# (Gemini 2.5 Flash + 2-Step Tally Verification)
# Robust JSON handling + robust PDF bytes export
# ======================================================
import os, io, json, re, datetime as dt
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai
from utils.pdf_utils import get_pdf_base   # ✅ centralized PDF base

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Mill Register OCR Dashboard", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")
COLL_NAME = os.getenv("COLLECTION_NAME", "cop_winding_entries")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY missing in .env (get one from Google AI Studio)")

# ------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------
REGISTER_TYPES = [
    "Pile Khata with Roll Stock",
    "Spreader Roll Pile Stock Consumption",
    "Entry Khata",
    "Roll Stock Carding",
    "Drawing Meter Book",
    "Winding Production Khata",
    "Cop Winding (Weft)",
    "Warp Spool Winding Khata",
]

COP_COLUMNS = [
    "Sl_No","Frame_No","Worker_Name","Labour_No",
    "Quality","Frames","Winders","Marka","Spindle",
    "Tally_Marks","Tally_From_Marks","Tally_LastCol","Final_Tally","Verified",
    "Remarks","Lot_Total_Tally","Lot_Footer_Total","Lot_Verified"
]

# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def parse_frames_winders(text: str) -> Tuple[Any, Any]:
    m = re.search(r"(\d+)\s*[-/xX]\s*(\d+)", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def parse_marka_spindle(text: str) -> Tuple[Any, Any]:
    m = re.search(r"([A-Za-z/]+)\s*[-/]\s*(\d+)", str(text or ""))
    return (m.group(1).upper(), int(m.group(2))) if m else (None, None)

def fallback_tally_count(text: str) -> int | None:
    if not text:
        return None
    s = str(text).strip().replace("\\", "/")
    groups = re.findall(r'[|/]+', s)
    total = 0
    for g in groups:
        if '/' in g:
            total += 5
        else:
            cnt = g.count('|')
            total += (cnt // 5) * 5 + (cnt % 5)
    return total if total > 0 else None

def json_safe_load(s: str) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        try:
            m = re.search(r"\{.*\}", str(s), flags=re.S)
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}

def to_int(x):
    try:
        if x in [None, "", "null"]:
            return None
        return int(str(x).strip())
    except Exception:
        return None

def normalize_and_verify(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = rows or []
    norm = []
    for r in rows:
        row = {c: r.get(c) for c in COP_COLUMNS if c in r}

        f, w = parse_frames_winders(r.get("Quality", "") or "")
        mk, sp = parse_marka_spindle(r.get("Quality", "") or "")
        row["Frames"] = to_int(row.get("Frames")) or f
        row["Winders"] = to_int(row.get("Winders")) or w
        row["Marka"] = row.get("Marka") or mk
        row["Spindle"] = to_int(row.get("Spindle")) or sp

        marks_count = to_int(row.get("Tally_From_Marks")) or fallback_tally_count(r.get("Tally_Marks"))
        last_col = to_int(row.get("Tally_LastCol"))

        if marks_count is not None and last_col is not None:
            verified = "Yes" if marks_count == last_col else "No"
            final_tally = last_col
        elif last_col is not None:
            verified = "No"; final_tally = last_col
        elif marks_count is not None:
            verified = "No"; final_tally = marks_count
        else:
            verified = "No"; final_tally = None

        row.update({
            "Tally_From_Marks": marks_count,
            "Tally_LastCol": last_col,
            "Final_Tally": final_tally,
            "Verified": verified,
            "Sl_No": to_int(row.get("Sl_No")),
            "Labour_No": to_int(row.get("Labour_No")),
            "Lot_Footer_Total": to_int(row.get("Lot_Footer_Total")),
        })
        norm.append(row)

    df = pd.DataFrame(norm)
    if df.empty:
        # Ensure all columns exist even if there are no rows
        for c in COP_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[COP_COLUMNS]

    # Ensure missing columns exist
    for c in COP_COLUMNS:
        if c not in df.columns:
            df[c] = None

    # Compute lot totals
    df["Final_Tally"] = pd.to_numeric(df["Final_Tally"], errors="coerce").fillna(0).astype(int)
    lot_sum = df.groupby("Quality", dropna=False)["Final_Tally"].sum().reset_index()
    lot_sum.rename(columns={"Final_Tally": "Lot_Total_Tally"}, inplace=True)
    df = df.merge(lot_sum, on="Quality", how="left")

    footer = df.groupby("Quality", dropna=False)["Lot_Footer_Total"].max().reset_index()
    df = df.drop(columns=["Lot_Footer_Total"]).merge(footer, on="Quality", how="left")

    def lot_verify(row):
        if row.get("Lot_Footer_Total") is None:
            return "Unknown"
        try:
            return "Yes" if int(row["Lot_Total_Tally"]) == int(row["Lot_Footer_Total"]) else "No"
        except Exception:
            return "Unknown"

    df["Lot_Verified"] = df.apply(lot_verify, axis=1)
    return df[COP_COLUMNS]

# ------------------------------------------------------
# GEMINI OCR
# ------------------------------------------------------
def call_gemini_vision_for_cop(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing"); return {"header": {}, "rows": []}

    prompt = """You are extracting rows from a *Cop Winding (Weft)* register.
Return STRICT JSON ONLY:
{
  "header": {"Register_Name": "S/Weft Wind", "Shift": "A/B/C", "Date": "DD/MM/YY"},
  "rows": [
    {
      "Sl_No": int,
      "Frame_No": str,
      "Worker_Name": str,
      "Labour_No": int,
      "Quality": str,
      "Frames": int,
      "Winders": int,
      "Marka": str,
      "Spindle": int,
      "Tally_Marks": str,
      "Tally_From_Marks": int,
      "Tally_LastCol": int,
      "Remarks": str,
      "Lot_Footer_Total": int
    }
  ]
}"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content([prompt, {"mime_type": mime_type, "data": image_bytes}],
                                      generation_config=cfg)
        data = json_safe_load(resp.text)
        if not isinstance(data, dict):
            return {"header": {}, "rows": []}
        data.setdefault("header", {})
        data.setdefault("rows", [])
        return data
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header": {}, "rows": []}

# ------------------------------------------------------
# ✅ PDF (robust bytes export)
# ------------------------------------------------------
def save_pdf(df: pd.DataFrame, header: dict) -> bytes:
    """
    Always return bytes. Prefer dest='S'. If that fails, write to /tmp and read back.
    """
    pdf = get_pdf_base("Cop Winding Production — OCR Extract", header)
    pdf.set_font("NotoSans", "", 8)

    show_cols = df.columns.tolist()
    # simple width calc; avoid zero division
    col_w = [max(20, 240 // max(1, len(show_cols))) for _ in show_cols]

    # Header row
    for i, c in enumerate(show_cols):
        pdf.cell(col_w[i], 6, str(c), border=1, align="C")
    pdf.ln()

    # Data rows
    for _, r in df.iterrows():
        for i, c in enumerate(show_cols):
            pdf.cell(col_w[i], 6, str(r.get(c, ""))[:25], border=1)
        pdf.ln()

    # Try standard in-memory export first
    try:
        out = pdf.output(dest="S")
        if isinstance(out, (bytes, bytearray)):
            return bytes(out)
        # pyfpdf returns str
        return out.encode("latin-1", errors="ignore")
    except Exception:
        pass

    # Fallback: write to /tmp and read back (Streamlit Cloud safe)
    try:
        tmp_path = "/tmp/_cop_winding.pdf"
        pdf.output(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        # Last fallback: return an empty but valid bytes object to avoid Streamlit type errors
        st.error(f"PDF export failed: {e}")
        return b"%PDF-1.4\n%%EOF"

# ------------------------------------------------------
# MONGO SAVE
# ------------------------------------------------------
def upsert_mongo(header: Dict[str, Any], df: pd.DataFrame, img_name: str, img_bytes: bytes):
    doc = {
        "register_type": "Cop Winding (Weft)",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": img_name,
        "extracted_data": df.to_dict(orient="records"),
        "validated": False,
    }
    key = {"original_image_name": img_name, "header.Date": header.get("Date")}
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True,
                                    return_document=ReturnDocument.AFTER)

# ------------------------------------------------------
# UI
# ------------------------------------------------------
st.title("🧵 Mill Register OCR Dashboard — 2-Step Verified Tallies (Gemini 2.5 Flash)")

with st.sidebar:
    st.subheader("Settings")
    reg = st.selectbox("Select Register Type", REGISTER_TYPES, index=6)
    st.caption("Implemented for Cop Winding (Weft)")
    st.markdown("---")
    cam = st.camera_input("📸 Capture Image")
    up = st.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg"])

colL, colR = st.columns([1.3, 1])
with colL:
    st.subheader(f"Selected Register: {reg}")
    if reg != "Cop Winding (Weft)":
        st.info("Currently implemented only for Cop Winding (Weft)")
    else:
        img_bytes = img_name = mime = None
        if cam:
            img_bytes = cam.getvalue(); img_name = f"cam_{dt.datetime.utcnow().isoformat()}.jpg"; mime = "image/jpeg"
        elif up:
            img_bytes = up.getvalue(); img_name = up.name; mime = up.type

        if img_bytes:
            st.image(img_bytes, caption="Input Image", use_container_width=True)
            st.markdown("**Step 1:** Extracting with Gemini (2-step tally verification)…")
            data = call_gemini_vision_for_cop(img_bytes, mime)

            # ✅ Guard against invalid shapes
            if not isinstance(data, dict):
                st.error("❌ Invalid OCR response."); data = {"header": {}, "rows": []}

            rows = data.get("rows", []) or []
            df = normalize_and_verify(rows)
            header = data.get("header", {}) or {}

            st.markdown("**Step 2:** Review & Edit")
            edited = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Verified": st.column_config.SelectboxColumn(options=["Yes", "No"], width="small"),
                    "Lot_Verified": st.column_config.SelectboxColumn(options=["Yes", "No", "Unknown"], width="small"),
                },
            )

            bad_rows = edited[(edited["Verified"] == "No") & (edited["Final_Tally"].notna())]
            if len(bad_rows):
                st.warning(f"⚠️ {len(bad_rows)} row(s) failed row-level verification.")
            bad_lots = edited[(edited["Lot_Verified"] == "No")]
            if len(bad_lots):
                st.warning(f"⚠️ {len(bad_lots['Quality'].unique())} lot(s) failed lot verification.")

            st.markdown("**Step 3:** Header Details")
            c1, c2, c3 = st.columns(3)
            regn = c1.text_input("Register Name", header.get("Register_Name") or "S/Weft Wind")
            shift = c2.text_input("Shift", header.get("Shift") or "")
            date = c3.text_input("Date", header.get("Date") or "")
            header_edit = {"Register_Name": regn, "Shift": shift, "Date": date}

            st.markdown("**Step 4:** Save / Export (Final Columns Only)**")
            final_cols = [
                "Sl_No","Frame_No","Worker_Name","Labour_No","Quality",
                "Frames","Winders","Marka","Spindle","Final_Tally","Lot_Footer_Total"
            ]
            # Ensure columns exist to avoid KeyError on empty sets
            for c in final_cols:
                if c not in edited.columns:
                    edited[c] = None
            df_final = edited[final_cols].copy()

            if st.button("💾 Save to MongoDB", type="primary"):
                s = upsert_mongo(header_edit, df_final, img_name, img_bytes)
                st.success("✅ Saved to MongoDB (only key columns stored)")
                st.json({"_id": str(s.get("_id")), "Date": date})

            c1, c2, c3 = st.columns(3)
            csv_bytes = df_final.to_csv(index=False).encode()
            c1.download_button("⬇️ CSV", csv_bytes, "cop_winding_clean.csv", "text/csv")

            json_bytes = df_final.to_json(orient="records", indent=2).encode()
            c2.download_button("⬇️ JSON", json_bytes, "cop_winding_clean.json", "application/json")

            pdf_bytes = save_pdf(df_final, header_edit)
            c3.download_button("⬇️ PDF", pdf_bytes, "cop_winding_clean.pdf", "application/pdf")

            st.caption("💡 Only key columns saved/exported. Verification fields stay for UI review.")
        else:
            st.info("📷 Capture or Upload an image to begin")

with colR:
    st.subheader("2-Step Tally Verification")
    st.write(
        "- Step A: Count marks — `|=1`, `||=2`, `|||=3`, `||||=4`, `||||/=5`\n"
        "- Step B: Read numeric total (last col)\n"
        "- Verified = **Yes** when both match. Final_Tally trusts numeric total."
    )
    st.markdown("---")
    st.subheader("MongoDB Status")
    try:
        st.write(f"DB: `{DB_NAME}`  Collection: `{COLL_NAME}`")
        st.write(f"Documents: **{coll.estimated_document_count()}**")
    except Exception as e:
        st.error(f"Mongo error: {e}")
