import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, INK, TEAL, card, chart_layout, flag_stats, section_intro, tile, validation

v = validation()
s = flag_stats(v, 0.5)
section_intro("Know which deliveries will run late, before the trucks leave",
              "Waypoint's three brands share one fleet. On busy days there are not enough trucks, and dispatchers find out about "
              "problems only after a store has been let down. This console turns past delivery records into warnings and plans "
              "the team can act on the evening before.")

k1, k2, k3 = st.columns(3)
with k1: tile(f"{s['recall']:.0%}", "of late deliveries are flagged in advance", "good")
with k2: tile("4 min", "average error on unloading time, about half the error of today's fixed estimates")
with k3: tile("7%", "average error when planning volume up to 10 weeks ahead")
st.caption("Measured on the last three months of records, which the system never saw while learning.")

st.markdown('<p class="big-q">Can the team trust a warning?</p>', unsafe_allow_html=True)
st.markdown("Every delivery gets a risk from 0 to 100%. When we grouped real deliveries by the risk the system gave them, "
            "the share that actually arrived late matched closely.")
bands = pd.cut(v["pred_late_prob"], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], labels=["Under 10%", "10 to 30%", "30 to 50%", "50 to 70%", "70 to 90%", "Over 90%"])
g = v.groupby(bands, observed=True).agg(deliveries=("late", "size"), late=("late", "mean")).reset_index()
fig = go.Figure(go.Bar(x=g["pred_late_prob"].astype(str), y=g["late"], marker_color=[TEAL, TEAL, "#C9A227", AMBER, AMBER, AMBER],
                       text=[f"{x:.0%}" for x in g["late"]], textposition="outside",
                       customdata=g["deliveries"], hovertemplate="Risk given: %{x}<br>Actually late: %{y:.0%}<br>Deliveries: %{customdata:,}<extra></extra>"))
chart_layout(fig, 340, xaxis_title="Risk the system gave", yaxis=dict(title="Share that really arrived late", tickformat=".0%", range=[0, 1.12]))
st.plotly_chart(fig, width="stretch")

st.markdown('<p class="big-q">What each team gets</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: card("Dispatchers", "A list of tonight's riskiest stops, so they can resequence a route or send a vehicle earlier before anyone is late.", AMBER)
with c2: card("Store managers", "A realistic arrival and unloading time, so they can have staff ready when the delivery actually arrives.", TEAL)
with c3: card("Fleet planners", "Ten weeks of expected volume per depot and brand, so extra trucks and refrigerated space are booked before festival weeks.", INK)

st.markdown('<p class="big-q">How it works, in three steps</p>', unsafe_allow_html=True)
for n, t in enumerate(["Learn from 92,000 past deliveries: when each truck really arrived and how long unloading really took.",
                       "Each evening, score every planned stop for unloading time and the chance of arriving after the store's delivery window.",
                       "On peak days, decide which orders go and which wait, with a clear reason for every order that waits."], 1):
    st.markdown(f'<div class="step"><b class="n">{n}</b><div>{t}</div></div>', unsafe_allow_html=True)
