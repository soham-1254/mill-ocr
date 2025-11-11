import streamlit as st
import time

# ================================================
# ⚙️ Session Manager — for Login / Logout / Timeout
# ================================================

SESSION_TIMEOUT = 3600  # seconds (1 hour)

def init_session():
    """Initialize session state variables."""
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "role" not in st.session_state:
        st.session_state["role"] = None
    if "login_time" not in st.session_state:
        st.session_state["login_time"] = None
    if "_rerun" not in st.session_state:
        st.session_state["_rerun"] = False


def login_user(username, role):
    """Login user and start session."""
    st.session_state["user"] = username
    st.session_state["role"] = role
    st.session_state["login_time"] = time.time()
    st.session_state["_rerun"] = True


def logout_user():
    """Logout user and clear session."""
    for key in ["user", "role", "login_time"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["login_time"] = None
    st.session_state["_rerun"] = True
    st.success("👋 Logged out successfully.")


def check_timeout():
    """Automatically logout if session timeout reached."""
    if st.session_state.get("login_time"):
        if time.time() - st.session_state["login_time"] > SESSION_TIMEOUT:
            st.warning("⏰ Session expired. Please log in again.")
            logout_user()

    # Handle rerun flag safely outside callbacks
    if st.session_state.get("_rerun"):
        st.session_state["_rerun"] = False
        st.experimental_rerun()
