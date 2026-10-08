import streamlit as st
from common import style

st.set_page_config(page_title="Waypoint Planning Console", page_icon="🚚", layout="wide")
style()

nav = {
    "Business view": [
        st.Page("views/b_overview.py", title="Overview", default=True),
        st.Page("views/b_tonight.py", title="Tonight's deliveries"),
        st.Page("views/b_weeks.py", title="The weeks ahead"),
        st.Page("views/b_peak.py", title="Peak day decisions"),
    ],
    "Our approach": [
        st.Page("views/a_journey.py", title="How we built it"),
    ],
    "Technical view": [
        st.Page("views/t_performance.py", title="Model performance"),
        st.Page("views/t_explain.py", title="Why the model decides"),
        st.Page("views/t_labels.py", title="Labels and data"),
        st.Page("views/t_engines.py", title="Forecast and optimiser"),
    ],
    "Explore": [
        st.Page("views/x_route.py", title="Route simulator"),
        st.Page("views/x_trip.py", title="Trip planner"),
    ],
}
with st.sidebar:
    st.markdown("### Waypoint Planning Console")
    st.caption("Team Synapse, Tech Triathlon 2026 Datathon")
st.navigation(nav).run()
