import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, INK, MUTED, ROOT, TEAL, chart_layout, hhmm, section_intro, validation

v = validation()
section_intro("Labels and data", "Neither target was given. Both were built from the route records, and getting that right "
              "matters more than any model setting.")

st.markdown('<p class="big-q">Build a service time label yourself</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
arr = c1.slider("Vehicle arrives", 270, 400, 300, 5, format="%d", help="Minutes after midnight")
wo = c2.slider("Window opens", 270, 400, 330, 5)
lv = c3.slider("Vehicle leaves", 280, 460, 350, 5)
wc = 450
start = max(arr, wo)
naive, ours = lv - arr, lv - start
fig = go.Figure()
fig.add_shape(type="rect", x0=wo, x1=wc, y0=-0.5, y1=1.5, fillcolor="#E3EEEA", line_width=0, layer="below")
fig.add_bar(y=["Naive: leave minus arrival"], x=[max(naive, 0)], base=[arr], orientation="h", marker_color="#B9C1C7", width=0.45, name="naive")
if start > arr:
    fig.add_bar(y=["Ours: leave minus later of arrival and window open"], x=[start - arr], base=[arr], orientation="h",
                marker=dict(color="rgba(0,0,0,0)", line=dict(color=MUTED, width=2)), width=0.45, name="waiting")
fig.add_bar(y=["Ours: leave minus later of arrival and window open"], x=[max(ours, 0)], base=[start], orientation="h", marker_color=TEAL, width=0.45, name="ours")
ticks = list(range(270, 471, 30))
chart_layout(fig, 230, showlegend=False, barmode="overlay", xaxis=dict(range=[265, 465], tickvals=ticks, ticktext=[hhmm(t) for t in ticks]))
st.plotly_chart(fig, width="stretch")
if lv <= start:
    st.warning("The vehicle must leave after handling starts. Move 'Vehicle leaves' later.")
else:
    st.markdown(f"Arrives {hhmm(arr)}, window opens {hhmm(wo)}, leaves {hhmm(lv)}: naive label **{naive} min**, our label **{ours} min**"
                + (f", so **{naive - ours} min of waiting** is removed." if naive != ours else ", the same because the vehicle was not early."))

st.code("service_min = leave_outlet_min - max(arrival_min, window_open_min)\n"
        "late        = 1 if arrival_min > window_close_min else 0   # actual arrival, closing minute itself is on time", language="python")

st.markdown('<p class="big-q">Checks before modelling</p>', unsafe_allow_html=True)
st.dataframe(pd.DataFrame([
    ["Every dispatched order joins to exactly one route leg", "91,894 of 91,894", "Passed"],
    ["Outlet on the order equals outlet on the leg", "0 mismatches", "Passed"],
    ["Route date equals dispatch date (deferred orders)", "0 mismatches", "Passed"],
    ["Next leg departs at the previous leave time", "100% of legs", "Passed"],
    ["Negative or missing labels", "0", "Passed"],
    ["Deliveries crossing midnight", "0 (arrivals 02:30 to 21:55)", "Passed"],
], columns=["Check", "Result", "Status"]), hide_index=True, width="stretch")

st.markdown('<p class="big-q">The signal the labels reveal</p>', unsafe_allow_html=True)
bins = [-999, 0, 15, 30, 45, 60, 90, 120, 999]
names = ["0 or less", "0 to 15", "15 to 30", "30 to 45", "45 to 60", "60 to 90", "90 to 120", "over 120"]
g = v.groupby(pd.cut(v.planned_slack_min, bins, labels=names), observed=True).late.agg(["mean", "size"]).reset_index()
e = v.groupby(pd.cut(v.exp_slack, bins, labels=names), observed=True).late.agg(["mean", "size"]).reset_index()
fig = go.Figure()
fig.add_bar(x=g.iloc[:, 0].astype(str), y=g["mean"], name="Grouped by planned slack", marker_color="#B9C1C7")
fig.add_bar(x=e.iloc[:, 0].astype(str), y=e["mean"], name="Grouped by expected slack (ours)", marker_color=AMBER)
chart_layout(fig, 340, barmode="group", xaxis_title="Slack before the window closes (minutes)", yaxis=dict(title="Late rate", tickformat=".0%"),
             legend=dict(orientation="h", y=1.15, x=0))
st.plotly_chart(fig, width="stretch")
st.caption("Validation period. Expected slack separates late from on time more sharply: its groups sit closer to 0% and 100%.")
st.image(str(ROOT / "assets" / "01_preprocessing_pipeline.png"), caption="Preprocessing pipeline")
