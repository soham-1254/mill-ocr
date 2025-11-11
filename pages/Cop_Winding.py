# ======================================================
# app.py — Mill Register OCR Dashboard (Gemini 2.5 Flash + 2-Step Tally Verification)
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
def parse_frames_winders(text:str)->Tuple[int,int]:
    m=re.search(r"(\d+)\s*[-/xX]\s*(\d+)",str(text or ""))
    return (int(m.group(1)),int(m.group(2))) if m else (None,None)

def parse_marka_spindle(text:str)->Tuple[str,int]:
    m=re.search(r"([A-Za-z/]+)\s*[-/]\s*(\d+)",str(text or ""))
    return (m.group(1).upper(),int(m.group(2))) if m else (None,None)

def fallback_tally_count(text: str) -> int | None:
    if not text:
        return None
    text = str(text).strip().replace("\\", "/")
    groups = re.findall(r'[|/]+', text)
    total = 0
    for g in groups:
        if '/' in g:
            total += 5
        else:
            count = g.count('|')
            total += (count // 5) * 5 + (count % 5)
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
    norm=[]
    for r in rows:
        row={c:r.get(c) for c in COP_COLUMNS if c in r}

        f,w=parse_frames_winders(r.get("Quality","") or "")
        mk,sp=parse_marka_spindle(r.get("Quality","") or "")
        row["Frames"]=to_int(row.get("Frames")) or f
        row["Winders"]=to_int(row.get("Winders")) or w
        row["Marka"]=row.get("Marka") or mk
        row["Spindle"]=to_int(row.get("Spindle")) or sp

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
            "Lot_Footer_Total": to_int(row.get("Lot_Footer_Total"))
        })
        norm.append(row)

    df=pd.DataFrame(norm,columns=[c for c in COP_COLUMNS if c not in ["Lot_Total_Tally","Lot_Verified"]])

    if not df.empty:
        df["Final_Tally"]=pd.to_numeric(df["Final_Tally"],errors="coerce").fillna(0).astype(int)
        lot_sum = df.groupby("Quality", dropna=False)["Final_Tally"].sum().reset_index()
        lot_sum.rename(columns={"Final_Tally":"Lot_Total_Tally"}, inplace=True)
        df = df.merge(lot_sum, on="Quality", how="left")

        footer = df.groupby("Quality", dropna=False)["Lot_Footer_Total"].max().reset_index()
        df = df.drop(columns=["Lot_Footer_Total"]).merge(footer, on="Quality", how="left")

        def lot_verify(row):
            if row.get("Lot_Footer_Total") is None: return "Unknown"
            try: return "Yes" if int(row["Lot_Total_Tally"]) == int(row["Lot_Footer_Total"]) else "No"
            except: return "Unknown"
        df["Lot_Verified"] = df.apply(lot_verify, axis=1)
    else:
        df["Lot_Total_Tally"]=None; df["Lot_Verified"]="Unknown"

    for c in COP_COLUMNS:
        if c not in df.columns:
            df[c]=None
    return df[COP_COLUMNS]

# ------------------------------------------------------
# GEMINI OCR
# ------------------------------------------------------
def call_gemini_vision_for_cop(image_bytes:bytes,mime_type:str)->Dict[str,any]:
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing");return {"header":{}, "rows":[]}

    prompt = """You are extracting rows from a *Cop Winding (Weft)* register...
(Return pure JSON only)"""
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
# ✅ PDF (Centralized /tmp Font via get_pdf_base)
# ------------------------------------------------------
def save_pdf(df:pd.DataFrame, header:Dict[str,Any]) -> bytes:
    pdf = get_pdf_base("Cop Winding (Weft) — OCR Extract (2-Step Verified)", header)

    pdf.set_font("NotoSans", "", 7)
    col_w=[8,22,35,16,22,12,14,14,16,18,16,16,16,12,18,16,18,14]

    for i, c in enumerate(COP_COLUMNS):
        pdf.cell(col_w[i], 6, str(c), border=1, align="C")
    pdf.ln()

    for _, r in df.iterrows():
        vals=[str(r.get(c,"") if r.get(c,"") is not None else "")[:22] for c in COP_COLUMNS]
        for i,v in enumerate(vals): pdf.cell(col_w[i],6,v,border=1)
        pdf.ln()

    return pdf.output(dest="S").encode("latin-1", errors="ignore")

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
            st.image(img_bytes,caption="Input Image",use_container_width=True)
            st.markdown("**Step 1:** Extracting with Gemini (2-step tally verification)…")
            data=call_gemini_vision_for_cop(img_bytes,mime)

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

            bad_rows = edited[(edited["Verified"]=="No") & (edited["Final_Tally"].notna())]
            if len(bad_rows): st.warning(f"⚠️ {len(bad_rows)} row(s) failed row-level verification.")
            bad_lots = edited[(edited["Lot_Verified"]=="No")]
            if len(bad_lots): st.warning(f"⚠️ {len(bad_lots['Quality'].unique())} lot(s) failed lot verification.")

            st.markdown("**Step 3:** Header Details**")
            c1,c2,c3=st.columns(3)
            regn=c1.text_input("Register Name",header.get("Register_Name")or"S/Weft Wind")
            shift=c2.text_input("Shift",header.get("Shift")or"")
            date=c3.text_input("Date",header.get("Date")or"")
            header_edit={"Register_Name":regn,"Shift":shift,"Date":date}

            st.markdown("**Step 4:** Save / Export (Final Columns Only)**")
            final_cols=["Sl_No","Frame_No","Worker_Name","Labour_No","Quality","Frames","Winders","Marka","Spindle","Final_Tally","Lot_Footer_Total"]
            df_final=edited[final_cols].copy()

            if st.button("💾 Save to MongoDB", type="primary"):
                s = upsert_mongo(header_edit, df_final, img_name, img_bytes)
                st.success("✅ Saved to MongoDB (only key columns stored)")
                st.json({"_id": str(s.get("_id")), "Date": date})

            c1,c2,c3=st.columns(3)
            csv_bytes=df_final.to_csv(index=False).encode()
            c1.download_button("⬇️ CSV",csv_bytes,"cop_winding_clean.csv","text/csv")

            json_bytes=df_final.to_json(orient="records",indent=2).encode()
            c2.download_button("⬇️ JSON",json_bytes,"cop_winding_clean.json","application/json")

            pdf_bytes=save_pdf(df_final,header_edit)
            c3.download_button("⬇️ PDF",pdf_bytes,"cop_winding_clean.pdf","application/pdf")

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
        st.write(f"DB:`{DB_NAME}`  Collection:`{COLL_NAME}`")
        st.write(f"Documents: **{coll.estimated_document_count()}**")
    except Exception as e:
        st.error(f"Mongo error: {e}")
