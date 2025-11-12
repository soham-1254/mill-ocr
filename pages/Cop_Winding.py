# ======================================================
# app.py — Mill Register OCR Dashboard (Gemini 2.5 Flash + 2-Step Tally Verification)
# ======================================================
import os, io, json, re, datetime as dt, requests
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai
from fpdf import FPDF
from utils.pdf_utils import ensure_font_available, get_pdf_base


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
    "Cop Winding (Weft)",  # ✅ implemented
    "Warp Spool Winding Khata",
]

# Final schema (includes double-verification & lot footer)
COP_COLUMNS = [
    "Sl_No","Frame_No","Worker_Name","Labour_No",
    "Quality","Frames","Winders","Marka","Spindle",
    "Tally_Marks","Tally_From_Marks","Tally_LastCol","Final_Tally","Verified",
    "Remarks","Lot_Total_Tally","Lot_Footer_Total","Lot_Verified"
]

HEADER_FIELDS = ["Register_Name","Shift","Date"]

# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def parse_frames_winders(text:str)->Tuple[int,int]:
    m=re.search(r"(\d+)\s*[-/xX]\s*(\d+)",str(text or ""))
    return (int(m.group(1)),int(m.group(2))) if m else (None,None)

def parse_marka_spindle(text:str)->Tuple[str,int]:
    m=re.search(r"([A-Za-z/]+)\s*[-/]\s*(\d+)",str(text or ""))
    return (m.group(1).upper(),int(m.group(2))) if m else (None,None)

def fallback_tally_count(text: str) -> int | None:
    """
    Tally rules (your chart):
      ||||/ = 5,  |||| = 4,  ||| = 3,  || = 2,  | = 1
    Multiple groups like "||||/ ||| ||" are summed.
    """
    if not text:
        return None
    text = str(text).strip().replace("\\", "/")
    groups = re.findall(r'[|/]+', text)
    total = 0
    for g in groups:
        if '/' in g:            # crossed group
            total += 5
        else:
            count = g.count('|')
            if count > 5:       # very long run of |
                total += (count // 5) * 5 + (count % 5)
            else:
                total += count
    return total if total > 0 else None

def json_safe_load(s:str)->Dict[str,any]:
    try: return json.loads(s)
    except:
        m=re.search(r'\{.*\}',s,flags=re.S)
        if m:
            try: return json.loads(m.group(0))
            except: pass
    return {}

def to_int(x):
    try:
        if x in [None,"","null"]: return None
        return int(str(x).strip())
    except:
        return None

def normalize_and_verify(rows:List[Dict[str,any]])->pd.DataFrame:
    """
    - Ensure columns exist
    - Derive Frames/Winders/Marka/Spindle from 'Quality' hints if missing
    - 2-step verification:
        * compute marks_count from Tally_Marks (fallback_tally_count)
        * read last col as Tally_LastCol (int)
        * Final_Tally = Tally_LastCol if present else marks_count
        * Verified = Yes if both present and equal, else No (or Unknown if one missing)
    - Lot totals: sum Final_Tally by Quality
      and compare with Lot_Footer_Total (if present) => Lot_Verified
    """
    norm=[]
    for r in rows:
        row={c:r.get(c) for c in COP_COLUMNS if c in r}  # keep known keys if present

        # essentials
        f,w=parse_frames_winders(r.get("Quality","") or "")
        mk,sp=parse_marka_spindle(r.get("Quality","") or "")
        row["Frames"]=to_int(row.get("Frames")) or f
        row["Winders"]=to_int(row.get("Winders")) or w
        row["Marka"]=row.get("Marka") or mk
        row["Spindle"]=to_int(row.get("Spindle")) or sp

        # tally from marks
        marks_count = to_int(row.get("Tally_From_Marks"))
        if marks_count is None:
            marks_count = fallback_tally_count(r.get("Tally_Marks"))

        # last column numeric (OCR should read this from the final column)
        last_col = to_int(row.get("Tally_LastCol"))

        # decide final + verified
        if marks_count is not None and last_col is not None:
            verified = "Yes" if marks_count == last_col else "No"
            final_tally = last_col   # prefer the explicit numeric column
        elif last_col is not None:
            verified = "No"          # single-source (numeric only)
            final_tally = last_col
        elif marks_count is not None:
            verified = "No"          # single-source (marks only)
            final_tally = marks_count
        else:
            verified = "No"
            final_tally = None

        row["Tally_From_Marks"]=marks_count
        row["Tally_LastCol"]=last_col
        row["Final_Tally"]=final_tally
        row["Verified"]=verified

        # simple coercions
        row["Sl_No"]=to_int(row.get("Sl_No"))
        row["Labour_No"]=to_int(row.get("Labour_No"))
        row["Lot_Footer_Total"]=to_int(row.get("Lot_Footer_Total"))

        norm.append(row)

    df=pd.DataFrame(norm,columns=[c for c in COP_COLUMNS if c not in ["Lot_Total_Tally","Lot_Verified"]])

    # per-lot totals using Final_Tally
    if not df.empty:
        df["Final_Tally"]=pd.to_numeric(df["Final_Tally"],errors="coerce").fillna(0).astype(int)
        lot_sum = df.groupby("Quality", dropna=False)["Final_Tally"].sum().reset_index()
        lot_sum.rename(columns={"Final_Tally":"Lot_Total_Tally"}, inplace=True)
        df = df.merge(lot_sum, on="Quality", how="left")

        # bring forward any footer total present for the lot (first non-null)
        footer = df.groupby("Quality", dropna=False)["Lot_Footer_Total"].max().reset_index()
        df = df.drop(columns=["Lot_Footer_Total"]).merge(footer, on="Quality", how="left")

        # verify lot using footer total when present
        def lot_verify(row):
            if row.get("Lot_Footer_Total") is None:
                return "Unknown"
            try:
                return "Yes" if int(row["Lot_Total_Tally"]) == int(row["Lot_Footer_Total"]) else "No"
            except:
                return "Unknown"
        df["Lot_Verified"] = df.apply(lot_verify, axis=1)
    else:
        df["Lot_Total_Tally"]=None
        df["Lot_Verified"]="Unknown"

    # ensure all columns exist
    for c in COP_COLUMNS:
        if c not in df.columns:
            df[c]=None
    return df[COP_COLUMNS]

# ------------------------------------------------------
# GEMINI OCR (with explicit 2-step instructions)
# ------------------------------------------------------
def call_gemini_vision_for_cop(image_bytes:bytes,mime_type:str)->Dict[str,any]:
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing");return {"header":{}, "rows":[]}

    prompt=f"""
You are extracting rows from a *Cop Winding (Weft)* register image.

Each record appears across 3 sub-rows:
• Row 1 → Quality (e.g. "21 lbs")
• Row 2 → Frames–Winders (e.g. "8-11" → Frames=8 Winders=11)
• Row 3 → Marka–Spindle (e.g. "Y-96" → Marka="Y" Spindle=96)

STRICT COLUMN MAPPING:
1 Sl_No, 2 Frame_No, 3 Worker_Name, 4 Labour_No, 5 Quality,
6 Frames, 7 Winders, 8 Marka, 9 Spindle,
10 Tally_Marks, 11 Tally_From_Marks, 12 Tally_LastCol, 13 Final_Tally, 14 Verified,
15 Remarks, 16 Lot_Total_Tally, 17 Lot_Footer_Total, 18 Lot_Verified.

DOUBLE VERIFICATION FOR TALLY (MANDATORY):
A) First, count from tallies you SEE:
   Convert visually drawn groups using:
     "||||/"=5, "||||"=4, "|||=3", "||"=2, "|"=1.
   Sum multiple groups (e.g. "||||/ |||| ||" → 5+4+2=11).
   Put this value in **Tally_From_Marks** (integer).
B) Then read the numeric value written in the final (rightmost) column of the row.
   Put this in **Tally_LastCol** (integer).
C) **Final_Tally** must be the trusted tally number for the row:
   - If both values exist and match → Final_Tally = that number and Verified="Yes".
   - If both exist but differ → Final_Tally = Tally_LastCol and Verified="No".
   - If only one exists → Final_Tally = the one that exists and Verified="No".
D) For each lot (same Quality), if a total is written after the lot (e.g., 104),
   output it in **Lot_Footer_Total** for the last row of that lot (others null).

Return **pure JSON** only:
{{
  "header": {{
    "Register_Name": str,
    "Shift": str,
    "Date": str(DD/MM/YYYY)
  }},
  "rows": [
    {{
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
      "Final_Tally": int,
      "Verified": "Yes" | "No",
      "Remarks": str,
      "Lot_Footer_Total": int
    }}
  ]
}}
"""

    try:
        model=genai.GenerativeModel("gemini-2.5-flash")
        cfg={"response_mime_type":"application/json"}
        resp=model.generate_content([prompt,{"mime_type":mime_type,"data":image_bytes}],
                                    generation_config=cfg)
        return json_safe_load(resp.text) or {"header":{}, "rows":[]}
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return {"header":{}, "rows":[]}

# ------------------------------------------------------
# PDF (Unicode Noto Sans)
# ------------------------------------------------------
def save_pdf(df:pd.DataFrame,header:Dict[str,Any])->bytes:
    pdf=FPDF(orientation="L",unit="mm",format="A4")
    pdf.add_page()
    font_path=os.path.join(os.path.dirname(__file__),"NotoSans-Regular.ttf")
    if not os.path.exists(font_path):
        url="https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
        r=requests.get(url);open(font_path,"wb").write(r.content)
    pdf.add_font("NotoSans","",font_path,uni=True)
    pdf.set_font("NotoSans","",14)
    def safe(t):return re.sub(r"[—–−]","-",str(t or ""))
    pdf.cell(0,8,safe("Cop Winding (Weft) - OCR Extract (2-Step Verified)"),ln=1)
    pdf.set_font("NotoSans","",11)
    rn=safe(header.get("Register_Name")or"S/Weft Wind")
    sh=safe(header.get("Shift")or"-")
    dtv=safe(header.get("Date")or"-")
    pdf.cell(0,7,f"Register:{rn}   Shift:{sh}   Date:{dtv}",ln=1)
    pdf.ln(2)
    pdf.set_font("NotoSans","",7)

    # Narrower columns to fit the extra verification fields
    col_w=[8,22,35,16,22,12,14,14,16,18,16,16,16,12,18,16,18,14]
    for i,c in enumerate(COP_COLUMNS): pdf.cell(col_w[i],6,str(c),border=1,align="C")
    pdf.ln()
    for _,r in df.iterrows():
        vals=[str(r.get(c,"") if r.get(c,"") is not None else "")[:22] for c in COP_COLUMNS]
        for i,v in enumerate(vals): pdf.cell(col_w[i],6,v,border=1)
        pdf.ln()
    return pdf.output(dest="S").encode("latin-1",errors="ignore")

# ------------------------------------------------------
# MONGO SAVE
# ------------------------------------------------------
def upsert_mongo(header,df,img_name,img_bytes):
    doc={
        "register_type":"Cop Winding (Weft)",
        "header":header,
        "timestamp":dt.datetime.utcnow(),
        "original_image_name":img_name,
        "extracted_data":df.to_dict(orient="records"),
        "validated":False,
    }
    key={"original_image_name":img_name,"header.Date":header.get("Date")}
    return coll.find_one_and_update(key,{"$set":doc},upsert=True,
                                    return_document=ReturnDocument.AFTER)

# ------------------------------------------------------
# UI
# ------------------------------------------------------
st.title("🧵 Mill Register OCR Dashboard — 2-Step Verified Tallies (Gemini 2.5 Flash)")

with st.sidebar:
    st.subheader("Settings")
    reg=st.selectbox("Select Register Type",REGISTER_TYPES,index=6)
    st.caption("Implemented for Cop Winding (Weft)")
    st.markdown("---")
    cam=st.camera_input("📸 Capture Image")
    up=st.file_uploader("📁 Upload Image",type=["png","jpg","jpeg"])

colL,colR=st.columns([1.3,1])
with colL:
    st.subheader(f"Selected Register: {reg}")
    if reg!="Cop Winding (Weft)":
        st.info("Currently implemented only for Cop Winding (Weft)")
    else:
        img_bytes=img_name=mime=None
        if cam: img_bytes, img_name, mime=cam.getvalue(),f"cam_{dt.datetime.utcnow().isoformat()}.jpg","image/jpeg"
        elif up: img_bytes, img_name, mime=up.getvalue(),up.name,up.type

        if img_bytes:
            st.image(img_bytes,caption="Input Image",use_column_width=True)
            st.markdown("**Step 1:** Extracting with Gemini (2-step tally verification)…")
            data=call_gemini_vision_for_cop(img_bytes,mime)

            # Normalize + verify
            df=normalize_and_verify(data.get("rows",[]))
            header=data.get("header",{}) or {}

            st.markdown("**Step 2:** Review & Edit**")
            edited=st.data_editor(
                df, use_container_width=True, num_rows="dynamic",
                column_config={
                    "Verified": st.column_config.SelectboxColumn(options=["Yes","No"], width="small"),
                    "Lot_Verified": st.column_config.SelectboxColumn(options=["Yes","No","Unknown"], width="small")
                }
            )

            # quick validation summary
            bad_rows = edited[(edited["Verified"]=="No") & (edited["Final_Tally"].notna())]
            if len(bad_rows):
                st.warning(f"⚠️ {len(bad_rows)} row(s) failed row-level verification (marks vs last column).")
            bad_lots = edited[(edited["Lot_Verified"]=="No")]
            if len(bad_lots):
                st.warning(f"⚠️ {len(bad_lots['Quality'].unique())} lot(s) failed lot footer verification.")

            st.markdown("**Step 3:** Header Details**")
            c1,c2,c3=st.columns(3)
            regn=c1.text_input("Register Name",header.get("Register_Name")or"S/Weft Wind")
            shift=c2.text_input("Shift",header.get("Shift")or"")
            date=c3.text_input("Date",header.get("Date")or"")
            header_edit={"Register_Name":regn,"Shift":shift,"Date":date}

            st.markdown("**Step 4:** Save / Export (Final Columns Only)**")

            # ✅ Keep all columns for UI editing
            df_display = edited.copy()

            # ✅ But only keep selected columns for export / Mongo
            final_cols = [
                "Sl_No","Frame_No","Worker_Name","Labour_No","Quality",
                "Frames","Winders","Marka","Spindle","Final_Tally","Lot_Footer_Total"
            ]
            df_final = edited[final_cols].copy()

            # ---------------- SAVE TO MONGO ----------------
            if st.button("💾 Save to MongoDB", type="primary"):
                s = upsert_mongo(header_edit, df_final, img_name, img_bytes)
                st.success("✅ Saved to MongoDB (only key columns stored)")
                st.json({"_id": str(s.get("_id")), "Date": date})

            # ---------------- EXPORTS ----------------
            c1, c2, c3 = st.columns(3)

            csv_bytes = df_final.to_csv(index=False).encode()
            c1.download_button("⬇️ CSV", csv_bytes, "cop_winding_clean.csv", "text/csv")

            json_bytes = df_final.to_json(orient="records", indent=2).encode()
            c2.download_button("⬇️ JSON", json_bytes, "cop_winding_clean.json", "application/json")

            pdf_bytes = save_pdf(df_final, header_edit)
            c3.download_button("⬇️ PDF", pdf_bytes, "cop_winding_clean.pdf", "application/pdf")

            st.caption("💡 Note: Only the selected key columns are saved/exported. Extra verification fields remain for UI review only.")

        else:
            st.info("📷 Capture or Upload an image to begin")

with colR:
    st.subheader("2-Step Tally Verification")
    st.write(
        "- Step A: Count from marks — `|=1`, `||=2`, `|||=3`, `||||=4`, `||||/=5`\n"
        "- Step B: Read numeric total in last column.\n"
        "- Verified = **Yes** when both match. Final_Tally trusts the last column if both exist."
    )
    st.markdown("---")
    st.subheader("MongoDB Status")
    try:
        st.write(f"DB:`{DB_NAME}`  Collection:`{COLL_NAME}`")
        st.write(f"Documents: **{coll.estimated_document_count()}**")
    except Exception as e:
        st.error(f"Mongo error: {e}")
