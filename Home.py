import streamlit as st
from session_manager import init_session, login_user, logout_user, check_timeout

# ======================================================
# ⚙️ PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="Mill Registers — Production OCR",
    layout="centered",
    initial_sidebar_state="collapsed"  # hide sidebar on login
)

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
    # Detect dark or light theme
    is_dark = st.get_option("theme.base") == "dark"

    input_bg = "#1e293b" if is_dark else "#ffffff"
    input_text = "#f1f5f9" if is_dark else "#1e293b"
    input_border = "#475569" if is_dark else "#94a3b8"
    label_color = "#e2e8f0" if is_dark else "#1e3a8a"
    login_bg = "linear-gradient(135deg, #1e293b, #0f172a)" if is_dark else "linear-gradient(135deg, #8ec5fc, #e0c3fc)"
    title_color = "#93c5fd" if is_dark else "#1e3a8a"
    sub_color = "#cbd5e1" if is_dark else "#334155"

    # CSS Styling
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"], header, footer {{visibility: hidden !important;}}
        body {{background-color: #000000 !important;}}

        .login-box {{
            background: {login_bg};
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            width: 400px;
            margin: 8rem auto;
            text-align: center;
            transition: all 0.3s ease;
        }}
        .login-box h2 {{
            color: {title_color};
            margin-bottom: 0.5rem;
        }}
        .login-sub {{
            color: {sub_color};
            font-size: 0.95rem;
            margin-bottom: 1.8rem;
        }}
        div[data-testid="stTextInput"] label {{
            color: {label_color} !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stTextInput"] input {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            border: 1px solid {input_border} !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.15);
        }}
        div[data-testid="stTextInput"] input:focus {{
            border: 1.5px solid #1A73E8 !important;
            box-shadow: 0 0 6px rgba(26,115,232,0.3);
        }}
        div[data-testid="stButton"] > button {{
            background: linear-gradient(90deg,#1A73E8,#4AB3F4);
            color: white;
            border: none;
            width: 100%;
            padding: 0.9rem 0;
            border-radius: 8px;
            font-weight: 600;
            margin-top: 0.5rem;
        }}
        div[data-testid="stButton"] > button:hover {{
            filter: brightness(1.1);
        }}
        .forgot {{
            color: {title_color};
            font-size: 0.9rem;
            text-align: right;
            margin-top: 0.4rem;
        }}
        .forgot a {{
            color: {title_color};
            text-decoration: none;
        }}
        .forgot a:hover {{
            text-decoration: underline;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Login Box
    st.markdown("""
    <div class='login-box'>
        <h2>🔐 Mill Production OCR Portal</h2>
        <p class='login-sub'>Secure access for authorized personnel only</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("👤 Username", placeholder="Enter your username")

    # Show/Hide password toggle
    show_pass = st.checkbox("👁️ Show password", value=False)
    password = st.text_input(
        "🔑 Password",
        placeholder="Enter your password",
        type="default" if show_pass else "password"
    )

    # Forgot password link
    st.markdown(
        "<div class='forgot'><a href='#' onclick='return false;'>Forgot Password?</a></div>",
        unsafe_allow_html=True
    )

    # Login button
    if st.button("Login", use_container_width=True):
        user = USERS.get(username)
        if user and user["password"] == password:
            login_user(username, user["role"])
            st.session_state["authenticated"] = True
            st.experimental_set_query_params(page="home")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Try again.")

    # Handle “Forgot Password” click (show info)
    if st.session_state.get("forgot_shown", False) is False:
        js = """
        <script>
        const forgot = document.querySelector('.forgot a');
        if(forgot){
            forgot.addEventListener('click', () => {
                window.parent.postMessage({ type: 'forgot_password' }, '*');
            });
        }
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)
        st.session_state["forgot_shown"] = True

    # Listen for custom event from JS
    event = st.session_state.get("forgot_event")
    if event == "show_info":
        st.info("📧 Please contact your IT administrator to reset your password.")


# ======================================================
# 🏠 MAIN OCR DASHBOARD
# ======================================================
def homepage():
    # Sidebar
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

    # Theme colors
    is_dark = st.get_option("theme.base") == "dark"
    bg_color = "#0b1221" if is_dark else "#ffffff"
    card_color = "#1e293b" if is_dark else "#ffffff"
    text_color = "#e6eaf3" if is_dark else "#0b1221"
    muted_color = "#a3acc2" if is_dark else "#606a7a"
    accent = "#4AB3F4"
    accent2 = "#1A73E8"
    border_color = "#334155" if is_dark else "#e7eaf0"
    shadow = "0 6px 18px rgba(0,0,0,0.4)" if is_dark else "0 6px 18px rgba(16,24,40,0.08)"

    # CSS
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

    # Header
    st.markdown('<h1 class="app-title">🧵 Production Mill Register OCR — Main Menu</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-sub">Select a Register to Continue</p>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        for name, page in [
            ("📘 Cop Winding (Weft)", "pages/Cop_Winding.py"),
            ("📘 Batching Entry Khata", "pages/Batching_Entry.py"),
            ("📘 Roll Stock Consumption", "pages/Roll_Stock_Consumption.py"),
        ]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button(name):
                st.switch_page(page)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        for name, page in [
            ("🧵 Spool Winding (Warp)", "pages/Spool_Winding.py"),
            ("📘 Drawing Meter Reading", "pages/Drawing_Meter.py"),
            ("📘 Roll Stock Carding", "pages/Roll_Stock_Carding.py"),
        ]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button(name):
                st.switch_page(page)
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
if not st.session_state.get("user"):
    login_page()
else:
    homepage()
