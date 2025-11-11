import os, requests
import streamlit as st
from fpdf import FPDF

# ==================================================
# 🔤 Font Setup Utility (shared by all pages)
# ==================================================
def ensure_font_available():
    """
    Ensures NotoSans-Regular.ttf is available for PDF generation.
    Downloads once into /tmp (works on Streamlit Cloud).
    """
    tmp_font = "/tmp/NotoSans-Regular.ttf"
    if not os.path.exists(tmp_font):
        try:
            url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
            r = requests.get(url, timeout=30)
            with open(tmp_font, "wb") as f:
                f.write(r.content)
        except Exception as e:
            st.warning(f"⚠️ Could not download NotoSans font: {e}")
            return None
    return tmp_font


# ==================================================
# 🧾 Standardized PDF Export
# ==================================================
def get_pdf_base(title: str, header: dict):
    """
    Creates an FPDF object with NotoSans font ready.
    Returns a configured FPDF instance.
    """
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()

    font_path = ensure_font_available()
    if font_path:
        pdf.add_font("NotoSans", "", font_path, uni=True)
        pdf.set_font("NotoSans", "", 11)
    else:
        pdf.set_font("Helvetica", "", 11)

    pdf.cell(0, 8, title, ln=1)
    hdr_line = " | ".join([f"{k}: {v}" for k, v in header.items() if v])
    pdf.cell(0, 7, hdr_line, ln=1)
    pdf.ln(3)
    return pdf
