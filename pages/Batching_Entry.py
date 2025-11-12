# ================================================
# pages/Batching_Entry.py — Batching Entry OCR page
# (Gemini 2.5 Flash • Mongo • CSV/JSON/XLSX)
# Matches the provided Batching Entry photo layout
# ================================================
import os, io, re, json, datetime as dt
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai

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
    if isinstance(s, dict):
        return s
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

def parse_num(x):
    """Parse float; 'x'->0, '4:50'->4.50, '2.5MT'->2.5, '—'->None"""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s in {"-", "—"}:
        return None
    if s.lower() == "x":
        return 0.0
    s = s.replace(":", ".")
    s = re.sub(r"[^\d.\-]", "", s)  # keep digits . -
    if s in ("", ".", "-"):
        return None
    try:
        return float(s)
    except Exception:
        return None

def parse_int_or_x(x):
    """Parse int; 'x'->0, else int, else None."""
    if x is None:
        return None
    s = str(x).strip()
    if s.lower() == "x":
        return 0
    nums = re.findall(r"-?\d+", s)
    if not nums:
        return None
    try:
        return int(nums[0])
    except:
        return None

def split_qty_emulsion(s: str) -> Tuple[float|None, float|None]:
    """
    For Emulsion->Quantity cells like '663.60kg/25.00kg':
    returns (qty_kg, emul_kg)
    """
    if not s:
        return (None, None)
    # normalize separators
    s = s.replace("\\", "/")
    parts = [p.strip() for p in s.split("/") if p.strip()]
    if len(parts) == 1:
        return (parse_num(parts[0]), None)
    q = parse_num(parts[0])
    e = parse_num(parts[1])
    return (q, e)

def parse_count_with_hours(s: str) -> Tuple[int|None, int|None]:
    """
    For Machine Allocation cells like '01(5Hrs)' or '1(5hrs)' → (1,5)
    Plain '08' -> (8,None), 'x' -> (0,None)
    """
    if s is None:
        return (None, None)
    txt = str(s).strip()
    if txt.lower() == "x":
        return (0, None)
    m = re.match(r"^\s*(\d+)\s*(?:\((\d+)\s*Hrs?\))?\s*$", txt, flags=re.I)
    if m:
        return (int(m.group(1)), int(m.group(2)) if m.group(2) else None)
    # fallback to first int found
    val = parse_int_or_x(txt)
    return (val, None)

# ------------------ NORMALIZERS ------------------
def normalize_emulsion(emul_like: Dict[str, Any]) -> pd.DataFrame:
    """
    Expected fields in OCR:
    {
      "Fresh_A": "06", "Fresh_BC": "10",
      "Wastage_A": "x", "Wastage_BC": "x",
      "Quantity_A": "663.60kg/25.00kg",
      "Quantity_BC": "1106.00kg/25.00kg"
    }
    """
    emul_like = emul_like or {}
    fresh_a = parse_int_or_x(emul_like.get("Fresh_A"))
    fresh_bc = parse_int_or_x(emul_like.get("Fresh_BC"))
    wast_a = parse_int_or_x(emul_like.get("Wastage_A"))
    wast_bc = parse_int_or_x(emul_like.get("Wastage_BC"))
    qty_a, emul_a = split_qty_emulsion(emul_like.get("Quantity_A", ""))
    qty_bc, emul_bc = split_qty_emulsion(emul_like.get("Quantity_BC", ""))

    df = pd.DataFrame(
        [
            {"Row": "Fresh",   "A": fresh_a, "B+C": fresh_bc},
            {"Row": "Wastage", "A": wast_a,  "B+C": wast_bc},
            {"Row": "Quantity (kg / Emulsion kg)",
             "A": f"{qty_a if qty_a is not None else ''} / {emul_a if emul_a is not None else ''}",
             "B+C": f"{qty_bc if qty_bc is not None else ''} / {emul_bc if emul_bc is not None else ''}"},
        ],
        columns=["Row", "A", "B+C"]
    )
    return df

def normalize_production_table(prod_like: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Expected rows:
      {"Item":"Ropes & Habijabi", "A":"•", "B+C":"2.0MT"}
      {"Item":"Cutting", "A":"x", "B+C":"2.5MT"}
    Convert bullets/dots to None, 'x'->0, '2.0MT'->2.0
    """
    rows = []
    for r in prod_like or []:
        item = (r.get("Item") or "").strip()
        a = r.get("A")
        bc = r.get("B+C") or r.get("BC") or r.get("B_C")
        # dot or blank should be None
        if isinstance(a, str) and a.strip() in {"•", ".", ""}:
            a_val = None
        else:
            a_val = parse_num(a)
        bc_val = parse_num(bc)
        rows.append({"Item": item, "A_MT": a_val, "B+C_MT": bc_val})
    if not rows:
        rows = [{"Item":"Ropes & Habijabi","A_MT":None,"B+C_MT":None},
                {"Item":"Cutting","A_MT":None,"B+C_MT":None}]
    df = pd.DataFrame(rows, columns=["Item","A_MT","B+C_MT"])
    df.loc[len(df)] = {"Item":"TOTAL","A_MT":df["A_MT"].sum(skipna=True),"B+C_MT":df["B+C_MT"].sum(skipna=True)}
    return df

def normalize_machine_alloc(tbl_like: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rows like:
      {"Machine":"Spreader","A":"08","B":"08","C":"09"}
      {"Machine":"Cutting","B":"01(5Hrs)"}
      {"Machine":"Ropes & Habijabi","C":"01"}
    Returns two tables:
      - counts A/B/C
      - hours A/B/C (only when specified)
    """
    machines = []
    hours = []
    for r in tbl_like or []:
        name = (r.get("Machine") or "").strip()
        a_count,a_hr = parse_count_with_hours(r.get("A"))
        b_count,b_hr = parse_count_with_hours(r.get("B"))
        c_count,c_hr = parse_count_with_hours(r.get("C"))
        machines.append({"Machine":name, "A":a_count, "B":b_count, "C":c_count})
        hours.append({"Machine":name, "A_Hrs":a_hr, "B_Hrs":b_hr, "C_Hrs":c_hr})

    if not machines:
        machines = [{"Machine":m, "A":None, "B":None, "C":None}
                    for m in ["Spreader","Softener","Inter Spreader","Cutting","Ropes & Habijabi"]]
        hours = [{"Machine":m, "A_Hrs":None, "B_Hrs":None, "C_Hrs":None}
                 for m in ["Spreader","Softener","Inter Spreader","Cutting","Ropes & Habijabi"]]

    df_counts = pd.DataFrame(machines, columns=["Machine","A","B","C"])
    df_counts = df_counts.fillna(0)
    df_counts.loc[len(df_counts)] = {
        "Machine":"TOTAL",
        "A": df_counts["A"].iloc[:-1].sum(),
        "B": df_counts["B"].iloc[:-1].sum(),
        "C": df_counts["C"].iloc[:-1].sum(),
    }

    df_hours = pd.DataFrame(hours, columns=["Machine","A_Hrs","B_Hrs","C_Hrs"]).fillna("")
    return df_counts, df_hours

def normalize_pile(rows_like) -> pd.DataFrame:
    """
    Pile Made (MT) with columns A/B/C and totals.
    """
    rows = []
    for r in rows_like or []:
        if not isinstance(r, dict):
            continue
        rows.append({
            "Quality": r.get("Quality"),
            "A": parse_num(r.get("A")),
            "B": parse_num(r.get("B")),
            "C": parse_num(r.get("C")),
        })
    if not rows:
        rows = [{"Quality":"G.AS","A":None,"B":None,"C":None},
                {"Quality":"A5","A":None,"B":None,"C":None},
                {"Quality":"A7","A":None,"B":None,"C":None},
                {"Quality":"A6","A":None,"B":None,"C":None}]
    df = pd.DataFrame(rows, columns=["Quality","A","B","C"])
    # Totals row
    totals = pd.DataFrame([{
        "Quality": "TOTAL",
        "A": df["A"].sum(skipna=True),
        "B": df["B"].sum(skipna=True),
        "C": df["C"].sum(skipna=True),
    }])
    df = pd.concat([df, totals], ignore_index=True)
    # Grand total column
    df["Row_Total"] = df[["A","B","C"]].sum(axis=1, skipna=True)
    return df

def normalize_roll(rows_like) -> pd.DataFrame:
    """
    Roll Made (EA) with columns A/B/C and totals.
    """
    rows = []
    for r in rows_like or []:
        if not isinstance(r, dict):
            continue
        rows.append({
            "Quality": (r.get("Quality") or "").strip(),
            "A": parse_int_or_x(r.get("A")),
            "B": parse_int_or_x(r.get("B")),
            "C": parse_int_or_x(r.get("C")),
        })
    if not rows:
        for q in ["P","O","T","J","B","K","11","A","S"]:
            rows.append({"Quality":q,"A":None,"B":None,"C":None})
    df = pd.DataFrame(rows, columns=["Quality","A","B","C"])
    totals = pd.DataFrame([{
        "Quality":"TOTAL",
        "A": df["A"].sum(skipna=True),
        "B": df["B"].sum(skipna=True),
        "C": df["C"].sum(skipna=True),
    }])
    df = pd.concat([df, totals], ignore_index=True)
    df["Row_Total"] = df[["A","B","C"]].sum(axis=1, skipna=True)
    return df

# ------------------ GEMINI OCR ------------------
def call_gemini_for_batching(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    Strict schema tailored to the photo layout.
    """
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing")
        return {
            "header": {}, "emulsion": {}, "production_mt": [],
            "machine_allocation": [], "pile_made_mt": [], "roll_made_ea": [],
            "footers": {}
        }

    prompt = """
You are reading a BATCHING ENTRY register exactly in this structure.

Return STRICT JSON ONLY with these keys:

{
  "header": { "Date": "DD/MM/YY or DD/MM/YYYY", "Unit": "HASTINGS JUTE MILL" },
  "emulsion": {
    "Fresh_A": str, "Fresh_BC": str,
    "Wastage_A": str, "Wastage_BC": str,
    "Quantity_A": str,    // e.g. "663.60kg/25.00kg"
    "Quantity_BC": str    // e.g. "1106.00kg/25.00kg"
  },
  "production_mt": [
    {"Item": "Ropes & Habijabi", "A": str, "B+C": str},
    {"Item": "Cutting", "A": str, "B+C": str}
  ],
  "machine_allocation": [
    {"Machine":"Spreader","A":str,"B":str,"C":str},
    {"Machine":"Softener","A":str,"B":str,"C":str},
    {"Machine":"Inter Spreader","A":str,"B":str,"C":str},
    {"Machine":"Cutting","A":str,"B":str,"C":str},
    {"Machine":"Ropes & Habijabi","A":str,"B":str,"C":str}
  ],
  "pile_made_mt": [
    {"Quality": str, "A": str, "B": str, "C": str}, ...
  ],
  "roll_made_ea": [
    {"Quality": str, "A": str, "B": str, "C": str}, ...
  ],
  "footers": {
    "Pile_Total": str,     // e.g. "36.80"
    "Roll_Total": str      // e.g. "949"
  }
}

Rules:
- Keep 'x' as 'x' in extraction, numbers as written (e.g., '2.0MT').
- For Quantity cells, include both values e.g. '663.60kg/25.00kg'.
- Use EXACT labels shown here for keys/Items/Machines.
Return only JSON.
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        cfg = {"response_mime_type": "application/json"}
        resp = model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}],
            generation_config=cfg
        )
        data = json_safe_load(resp.text) or {}
        # guarantee keys
        data.setdefault("header", {})
        data.setdefault("emulsion", {})
        data.setdefault("production_mt", [])
        data.setdefault("machine_allocation", [])
        data.setdefault("pile_made_mt", [])
        data.setdefault("roll_made_ea", [])
        data.setdefault("footers", {})
        return data
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {
            "header": {}, "emulsion": {}, "production_mt": [],
            "machine_allocation": [], "pile_made_mt": [], "roll_made_ea": [],
            "footers": {}
        }

# ------------------ MONGO UPSERT ------------------
def upsert_mongo(header: Dict[str, Any],
                 df_emul: pd.DataFrame,
                 df_prod: pd.DataFrame,
                 df_mac_counts: pd.DataFrame,
                 df_mac_hrs: pd.DataFrame,
                 df_pile: pd.DataFrame,
                 df_roll: pd.DataFrame,
                 foot: Dict[str, Any],
                 image_name: str,
                 raw_bytes: bytes):
    doc = {
        "register_type": "Batching Entry",
        "header": header,
        "timestamp": dt.datetime.utcnow(),
        "original_image_name": image_name,
        "emulsion": df_emul.to_dict(orient="records"),
        "production_mt": df_prod.to_dict(orient="records"),
        "machine_allocation_counts": df_mac_counts.to_dict(orient="records"),
        "machine_allocation_hours": df_mac_hrs.to_dict(orient="records"),
        "pile_made_mt": df_pile.to_dict(orient="records"),
        "roll_made_ea": df_roll.to_dict(orient="records"),
        "footers": foot,
        "validated": False,
    }
    key = {"original_image_name": image_name, "header.Date": header.get("Date")}
    return coll.find_one_and_update(key, {"$set": doc}, upsert=True, return_document=ReturnDocument.AFTER)

# ------------------ UI ------------------
st.title("🧪 Batching Entry — OCR (Photo-Accurate Layout)")

with st.sidebar:
    st.subheader("Input")
    cam = st.camera_input("📸 Capture Image (optional)")
    up = st.file_uploader("📁 Upload Image", type=["png","jpg","jpeg"])

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
    st.info("Upload or capture a Batching Entry sheet image to start.")
    st.stop()

st.image(img_bytes, caption="Input Image", use_column_width=True)
st.markdown("**Step 1:** Extracting with Gemini…")

data = call_gemini_for_batching(img_bytes, mime)

header_raw = data.get("header", {}) or {}
emul_raw = data.get("emulsion", {}) or {}
prod_raw = data.get("production_mt", []) or []
mac_raw = data.get("machine_allocation", []) or []
pile_raw = data.get("pile_made_mt", []) or []
roll_raw = data.get("roll_made_ea", []) or []
foot_raw = data.get("footers", {}) or {}

# Normalize to match the sheet
df_emul = normalize_emulsion(emul_raw)
df_prod = normalize_production_table(prod_raw)
df_mac_counts, df_mac_hrs = normalize_machine_alloc(mac_raw)
df_pile = normalize_pile(pile_raw)
df_roll = normalize_roll(roll_raw)

# ------------------ Editors (mirroring layout) ------------------
st.markdown("**Step 2: Review & Edit (matches sheet)**")

# Top: Header (Date)
h1, h2 = st.columns([1,1])
with h1:
    date_val = st.text_input("Date", value=header_raw.get("Date") or "")
with h2:
    unit_val = st.text_input("Unit", value=header_raw.get("Unit") or "HASTINGS JUTE MILL")
header = {"Date": date_val, "Unit": unit_val}

# Emulsion + Production
c1, c2 = st.columns(2)
with c1:
    st.subheader("Emulsion")
    df_emul_edit = st.data_editor(df_emul, use_container_width=True, num_rows="fixed", key="emul")
with c2:
    st.subheader("Production (MT)")
    df_prod_edit = st.data_editor(df_prod, use_container_width=True, num_rows="dynamic", key="prod")

# Machine Allocation (Counts + Hours)
c3, c4 = st.columns(2)
with c3:
    st.subheader("Machine Allocation — Counts (A/B/C)")
    df_mac_counts_edit = st.data_editor(df_mac_counts, use_container_width=True, num_rows="dynamic", key="mac_cnt")
with c4:
    st.subheader("Machine Allocation — Hours (if any)")
    df_mac_hrs_edit = st.data_editor(df_mac_hrs, use_container_width=True, num_rows="dynamic", key="mac_hrs")

# Pile & Roll sections
c5, c6 = st.columns(2)
with c5:
    st.subheader("Pile Made (MT)")
    df_pile_edit = st.data_editor(df_pile, use_container_width=True, num_rows="dynamic", key="pile")

with c6:
    st.subheader("Roll Made (EA)")
    df_roll_edit = st.data_editor(df_roll, use_container_width=True, num_rows="dynamic", key="roll")

# ------------------ Verifications (like the sheet totals) ------------------
st.markdown("**Step 3: Totals & Checks**")

# Pile totals check
pile_a = pd.to_numeric(df_pile_edit["A"], errors="coerce").sum(skipna=True)
pile_b = pd.to_numeric(df_pile_edit["B"], errors="coerce").sum(skipna=True)
pile_c = pd.to_numeric(df_pile_edit["C"], errors="coerce").sum(skipna=True)
pile_total_calc = pile_a + pile_b + pile_c
pile_footer = parse_num(foot_raw.get("Pile_Total"))

r1, r2, r3, r4 = st.columns(4)
r1.metric("Pile A (MT)", f"{pile_a:.2f}")
r2.metric("Pile B (MT)", f"{pile_b:.2f}")
r3.metric("Pile C (MT)", f"{pile_c:.2f}")
r4.metric("Pile Total (A+B+C)", f"{pile_total_calc:.2f}",
          None if pile_footer is None else ("Match ✅" if abs(pile_total_calc - pile_footer) < 1e-6
                                            else f"⚠️ Sheet:{pile_footer}"))

# Roll totals check
roll_a = pd.to_numeric(df_roll_edit["A"], errors="coerce").sum(skipna=True)
roll_b = pd.to_numeric(df_roll_edit["B"], errors="coerce").sum(skipna=True)
roll_c = pd.to_numeric(df_roll_edit["C"], errors="coerce").sum(skipna=True)
roll_total_calc = (roll_a or 0) + (roll_b or 0) + (roll_c or 0)
roll_footer = parse_int_or_x(foot_raw.get("Roll_Total"))

t1, t2, t3, t4 = st.columns(4)
t1.metric("Roll A (EA)", int(roll_a if pd.notna(roll_a) else 0))
t2.metric("Roll B (EA)", int(roll_b if pd.notna(roll_b) else 0))
t3.metric("Roll C (EA)", int(roll_c if pd.notna(roll_c) else 0))
t4.metric("Roll Total (A+B+C)", int(roll_total_calc),
          None if roll_footer is None else ("Match ✅" if roll_total_calc == roll_footer
                                            else f"⚠️ Sheet:{roll_footer}"))

# ------------------ Save / Export ------------------
st.markdown("**Step 4: Save / Export**")
b1, b2, b3, b4 = st.columns(4)

if b1.button("💾 Save to MongoDB", type="primary"):
    saved = upsert_mongo(
        header,
        df_emul_edit, df_prod_edit,
        df_mac_counts_edit, df_mac_hrs_edit,
        df_pile_edit, df_roll_edit,
        {"Pile_Total": f"{pile_total_calc:.2f}", "Roll_Total": int(roll_total_calc)},
        img_name, img_bytes
    )
    st.success("✅ Saved to MongoDB")
    st.json({"_id": str(saved.get("_id")), "Date": header["Date"]})

# JSON (full structure)
json_bytes = json.dumps({
    "header": header,
    "emulsion": df_emul_edit.to_dict(orient="records"),
    "production_mt": df_prod_edit.to_dict(orient="records"),
    "machine_allocation_counts": df_mac_counts_edit.to_dict(orient="records"),
    "machine_allocation_hours": df_mac_hrs_edit.to_dict(orient="records"),
    "pile_made_mt": df_pile_edit.to_dict(orient="records"),
    "roll_made_ea": df_roll_edit.to_dict(orient="records"),
    "totals": {"Pile_Total": f"{pile_total_calc:.2f}", "Roll_Total": int(roll_total_calc)},
}, indent=2).encode()
b2.download_button("⬇️ JSON", json_bytes, "batching_entry.json", "application/json")

# XLSX: each section as a sheet, including totals rows
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    pd.DataFrame([header]).to_excel(writer, index=False, sheet_name="Header")
    df_emul_edit.to_excel(writer, index=False, sheet_name="Emulsion")
    df_prod_edit.to_excel(writer, index=False, sheet_name="ProductionMT")
    df_mac_counts_edit.to_excel(writer, index=False, sheet_name="MachineCounts")
    df_mac_hrs_edit.to_excel(writer, index=False, sheet_name="MachineHours")
    df_pile_edit.to_excel(writer, index=False, sheet_name="PileMadeMT")
    df_roll_edit.to_excel(writer, index=False, sheet_name="RollMadeEA")
    pd.DataFrame([{"Pile_Total": f"{pile_total_calc:.2f}", "Roll_Total": int(roll_total_calc)}])\
        .to_excel(writer, index=False, sheet_name="Totals")

b3.download_button(
    "⬇️ XLSX",
    xlsx_buf.getvalue(),
    "batching_entry.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Quick CSVs (each major table)
b4.download_button("⬇️ Emulsion CSV", df_emul_edit.to_csv(index=False).encode(), "emulsion.csv", "text/csv")
st.download_button("⬇️ Production CSV", df_prod_edit.to_csv(index=False).encode(), "production_mt.csv", "text/csv")
st.download_button("⬇️ Machine Counts CSV", df_mac_counts_edit.to_csv(index=False).encode(), "machine_counts.csv", "text/csv")
st.download_button("⬇️ Machine Hours CSV", df_mac_hrs_edit.to_csv(index=False).encode(), "machine_hours.csv", "text/csv")
st.download_button("⬇️ Pile CSV", df_pile_edit.to_csv(index=False).encode(), "pile_made_mt.csv", "text/csv")
st.download_button("⬇️ Roll CSV", df_roll_edit.to_csv(index=False).encode(), "roll_made_ea.csv", "text/csv")

st.markdown("---")
st.caption("Tailored to your sheet: Emulsion, Production(MT), Machine Allocation (counts + hours), Pile Made (MT), Roll Made (EA) with totals and verification.")
