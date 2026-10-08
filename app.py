import streamlit as st
from common import style

st.set_page_config(page_title="Waypoint Planning Console", page_icon="🚚", layout="wide")
style()

pages = [
    st.Page("pages/overview.py", title="Tonight's run", default=True),
    st.Page("pages/route_risk.py", title="Route risk"),
    st.Page("pages/demand.py", title="Demand outlook"),
    st.Page("pages/peak_day.py", title="Peak day plan"),
    st.Page("pages/method.py", title="How it works"),
]
with st.sidebar:
    st.markdown("### Waypoint Planning Console")
    st.caption("Team Synapse, Tech Triathlon 2026 Datathon")
st.navigation(pages).run()
