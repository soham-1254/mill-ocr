import streamlit as st
from session_manager import init_session, login_user, logout_user, check_timeout

# ======================================================
# ⚙️ PAGE CONFIGURATION
# ======================================================
st.set_page_config(page_title="Mill Registers — Production OCR", layout="centered")
init_session()
check_timeout()

# ======================================================
# 👤 DEMO USER CREDENTIALS
# ======================================================
USERS = {
    "admin": {"password": "stil@admin", "role": "Admin"},
    "manager": {"password": "stil@manager123", "role": "Manager"},
    "user": {"password": "stil@user", "role": "User"},
}

# ======================================================
# 🔐 LOGIN PAGE
# ======================================================
def login_page():
    st.markdown(
        """
        <style>
        body {background-color: #000000;}
        .login-box {
            background: linear-gradient(135deg, #8ec5fc, #e0c3fc);
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            width: 400px;
            margin: 5rem auto;
            text-align: center;
        }
        .login-box h2 {
            color: #1e3a8a;
            margin-bottom: 0.5rem;
        }
        .login-sub {
            color: #334155;
            font-size: 0.95rem;
            margin-bottom: 1.8rem;
        }
        div[data-testid="stTextInput"] label {
            color: white !important;
            font-weight: 600 !important;
        }
        div[data-testid="stTextInput"] input {
            background-color: rgba(30,41,59,0.8) !important;
            color: white !important;
            border: 1px solid #475569 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stButton"] > button {
            background: linear-gradient(90deg,#1A73E8,#4AB3F4);
            color: white;
            border: none;
            width: 100%;
            padding: 0.9rem 0;
            border-radius: 8px;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button:hover {
            filter: brightness(1.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class='login-box'>
        <h2>🔐 Mill Production OCR Portal</h2>
        <p class='login-sub'>Secure access for authorized personnel only</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("👤 Username", placeholder="Enter your username")
    password = st.text_input("🔑 Password", placeholder="Enter your password", type="password")

    if st.button("Login", use_container_width=True):
        user = USERS.get(username)
        if user and user["password"] == password:
            login_user(username, user["role"])
            st.success("✅ Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Try again.")

# ======================================================
# 🏠 MAIN OCR DASHBOARD
# ======================================================
def homepage():
    # --- Sidebar user info ---
    st.sidebar.markdown(
        f"""
        <div style="padding:15px; background:linear-gradient(135deg,#3b82f6,#60a5fa);
        border-radius:10px; color:white; text-align:center;">
            👋 <b>{st.session_state.user}</b><br>
            <span style="font-size:14px;">({st.session_state.role})</span>
        </div>
        """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.button("🚪 Logout", on_click=logout_user)

    # --- Detect active theme ---
    is_dark = st.get_option("theme.base") == "dark"

    # --- Dynamic colors ---
    bg_color = "#0b1221" if is_dark else "#ffffff"
    card_color = "#1e293b" if is_dark else "#ffffff"
    text_color = "#e6eaf3" if is_dark else "#0b1221"
    muted_color = "#a3acc2" if is_dark else "#606a7a"
    accent = "#4AB3F4"
    accent2 = "#1A73E8"
    border_color = "#334155" if is_dark else "#e7eaf0"
    shadow = "0 6px 18px rgba(0,0,0,0.4)" if is_dark else "0 6px 18px rgba(16,24,40,0.08)"

    # --- CSS Theme ---
    st.markdown(f"""
    <style>
    html, body, [class*="stAppViewContainer"] {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .app-title {{
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
        background: linear-gradient(90deg, {accent}, {accent2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .app-sub {{
        text-align: center;
        color: {muted_color};
        font-size: 18px;
        margin-bottom: 25px;
    }}
    .hr {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {border_color}, transparent);
        margin: 18px 0;
    }}
    .card {{
        background: {card_color};
        border: 1px solid {border_color};
        border-radius: 12px;
        box-shadow: {shadow};
        padding: 14px;
        transition: all .2s ease;
    }}
    .card:hover {{
        transform: translateY(-3px);
        border-color: {accent};
        box-shadow: 0 10px 20px rgba(0,0,0,0.25);
    }}
    .card div[data-testid="stButton"] > button {{
        width: 100%;
        padding: 14px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, {accent2}, {accent});
        color: white;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: all 0.15s ease-in-out;
    }}
    .card div[data-testid="stButton"] > button:hover {{
        filter: brightness(1.1);
        transform: scale(1.02);
    }}
    .footer {{
        text-align: center;
        color: {muted_color};
        margin-top: 25px;
        font-size: 14px;
    }}
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown('<h1 class="app-title">🧵 Production Mill Register OCR — Main Menu</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-sub">Select a Register to Continue</p>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # --- Button Cards ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("📘 Cop Winding (Weft)"):
            st.switch_page("pages/Cop_Winding.py")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("📘 Batching Entry Khata"):
            st.switch_page("pages/Batching_Entry.py")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("📘 Roll Stock Consumption"):
            st.switch_page("pages/Roll_Stock_Consumption.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("🧵 Spool Winding (Warp)"):
            st.switch_page("pages/Spool_Winding.py")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("📘 Drawing Meter Reading"):
            st.switch_page("pages/Drawing_Meter.py")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("📘 Roll Stock Carding"):
            st.switch_page("pages/Roll_Stock_Carding.py")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.button("📘 Spinning Production Form"):
        st.switch_page("pages/Spinning_Production.py")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<p class="footer">Powered by <b>Gemini 2.5 Flash</b> ⚡ + <b>Streamlit</b></p>', unsafe_allow_html=True)

# ======================================================
# 🚦 PAGE ROUTING
# ======================================================
if not st.session_state.user:
    login_page()
else:
    homepage()
