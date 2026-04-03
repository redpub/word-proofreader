import streamlit as st

home = st.Page("pages/home.py", title="主頁", icon="📝", default=True)
debug = st.Page("pages/debug_log.py", title="除錯紀錄", icon="🔍")

pg = st.navigation([home, debug])
pg.run()
