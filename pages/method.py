import streamlit as st
from common import ROOT

st.title("How it works")
st.markdown('<p class="lede">One pipeline from raw route records to the three answers. Every model here was trained '
            'from scratch on the supplied data.</p>', unsafe_allow_html=True)

st.subheader("Building the labels")
st.markdown("Neither target was in the data, so we built them from the route records.\n\n"
            "- **Service time** = leave time minus the later of arrival and window opening. A vehicle that arrives early waits, "
            "and waiting is not handling. For the 4.4% of vehicles that arrive early, leaving this out would add 20 minutes.\n"
            "- **Late** = the actual arrival is after the window closes. About 20% of deliveries are late, almost all of them Fresh.")

st.subheader("Our expected slack feature")
st.markdown("The plan assumes clear roads and standard handling. Along each route we add up the expected travel overrun "
            "(from traffic and road disruption) and the handling overrun at earlier stops (from each outlet's history). "
            "Planned slack minus that expected delay is the strongest single predictor of lateness: its AUC is 0.943, against 0.873 for planned slack.")

st.subheader("Results")
c1, c2, c3 = st.columns(3)
c1.metric("Service time error", "4.00 min", "-3.42 min against the planning allowance", delta_color="inverse")
c2.metric("Late risk AUC", "0.976", "+0.110 against slack groups")
c3.metric("Weekly volume error", "7.0%", "-1.7 points against naive", delta_color="inverse")

st.subheader("Architecture")
st.image(str(ROOT / "assets" / "01_preprocessing_pipeline.png"), caption="Preprocessing pipeline")
st.image(str(ROOT / "assets" / "02_model_architecture.png"), caption="Models for the three tasks")
st.image(str(ROOT / "assets" / "03_deployment_approach.png"), caption="Proposed deployment")
