import streamlit as st
import streamlit_app

# Define pages with custom titles and icons
home = st.Page("streamlit_app.py", title="主頁", icon="📝")
debug = st.Page("pages/debug_log.py", title="除錯紀錄", icon="🔍")

# Setup navigation
pg = st.navigation([home, debug])
pg.run()
