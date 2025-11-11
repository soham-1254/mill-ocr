import streamlit as st
import time

# ================================================
# ⚙️ Session Manager — Login / Logout / Timeout
# ================================================

SESSION_TIMEOUT = 3600  # 1 hour (in seconds)


def init_session():
    """Initialize session state variables once."""
    defaults = {
        "user": None,
        "role": None,
        "login_time": None,
        "expired": False,
        "authenticated": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def login_user(username: str, role: str):
    """Start session for a logged-in user."""
    st.session_state["user"] = username
    st.session_state["role"] = role
    st.session_state["login_time"] = time.time()
    st.session_state["authenticated"] = True
    st.session_state["expired"] = False


def logout_user():
    """End user session and clear all session variables."""
    for key in ["user", "role", "login_time", "authenticated", "expired"]:
        st.session_state[key] = None if key != "expired" else False
    st.info("👋 Logged out successfully. Please log in again.")


def check_timeout():
    """Check if the current session has expired."""
    if st.session_state.get("authenticated") and st.session_state.get("login_time"):
        elapsed = time.time() - st.session_state["login_time"]
        if elapsed > SESSION_TIMEOUT:
            st.session_state["expired"] = True
            st.session_state["authenticated"] = False
            logout_user()
            st.warning("⏰ Session expired due to inactivity.")
