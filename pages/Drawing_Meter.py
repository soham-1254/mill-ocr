# ======================================================
# pages/Drawing_Meter_Reading.py — FIXED COLUMN LOGIC
# ======================================================

import os, io, json, re, datetime as dt
import pandas as pd
import streamlit as st
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Drawing Meter Reading OCR", layout="wide")
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mill_registers")
COLL_NAME = os.getenv("DRAWING_COLLECTION", "drawing_meter_entries")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
coll = db[COLL_NAME]

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY missing in .env")

# ------------------ COLUMNS ------------------
ROW_COLUMNS = [
    "Drawing_Stage",
    "Mc_No",                   # FIXED → Machine Number
    "Efficiency_at_100%",      # FIXED → second column values like 1030, 1050
    "Opening_Meter_Reading",
    "Closing_Meter_Reading",
    "Difference",
    "Efficiency",
    "Worker_Name"
]

# ------------------ HELPERS ------------------
def json_safe_load(s: str) -> dict:
    try:
        return json.loads(s)
    except:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try: return json.loads(m.group(0))
            except: pass
    return {}

def to_int(x):
    try:
        if x in [None, "", "null", "-", "—", ",", ",,"]:
            return None
        return int(str(x).strip())
    except:
        return None

def to_float(x):
    try:
        if x in [None, "", "null", "-", "—", ",", ",,"]:
            return None
        return float(str(x).strip())
    except:
        return None

# ------------------ NORMALIZE ------------------
def normalize_rows(rows: list) -> pd.DataFrame:
    norm = []

    last_mc = None
    last_eff100 = None

    for r in rows or []:

        # raw extracted values
        mc_raw = str(r.get("Mc_No") or "").strip()
        eff100_raw = str(r.get("Efficiency_at_100%") or "").strip()

        # ---------- APPLY ,, CARRY FORWARD RULE ----------
        if mc_raw in ["", ",", ",,"]:
            mc = last_mc
        else:
            mc = to_int(mc_raw)
            last_mc = mc

        if eff100_raw in ["", ",", ",,"]:
            eff100 = last_eff100
        else:
            eff100 = to_int(eff100_raw)
            last_eff100 = eff100

        row = {
            "Drawing_Stage": (r.get("Drawing_Stage") or "").strip(),
            "Mc_No": mc,
            "Efficiency_at_100%": eff100,
            "Opening_Meter_Reading": to_int(r.get("Opening_Meter_Reading")),
            "Closing_Meter_Reading": to_int(r.get("Closing_Meter_Reading")),
            "Worker_Name": str(r.get("Worker_Name") or "").strip()
        }

        # difference calculation
        if row["Opening_Meter_Reading"] is not None and row["Closing_Meter_Reading"] is not None:
            row["Difference"] = row["Closing_Meter_Reading"] - row["Opening_Meter_Reading"]
        else:
            row["Difference"] = None

        row["Efficiency"] = to_float(r.get("Efficiency"))

        norm.append(row)

    return pd.DataFrame(norm, columns=ROW_COLUMNS)
