import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, TEAL, chart_layout, routes, section_intro, tile

df = routes()
section_intro("Tonight's deliveries", "Pick a night to see where trouble is likely, and which stores to call ahead.")
c1, c2 = st.columns(2)
day = c1.selectbox("Delivery date", sorted(df["date"].unique()))
depot = c2.radio("Depot", ["Both", "Peliyagoda", "Kandy"], horizontal=True)
d = df[df["date"] == day]
if depot != "Both":
    d = d[d["depot"] == depot]
risky = d[d["pred_late_prob"] >= 0.5]
k1, k2, k3, k4 = st.columns(4)
with k1: tile(f"{d['route_id'].nunique()}", "trucks and vans on the road")
with k2: tile(f"{len(d)}", "store deliveries")
with k3: tile(f"{len(risky)}", "deliveries likely to be late", "alert" if len(risky) else "good")
with k4: tile(f"{risky['outlet_id'].nunique()}", "stores to call ahead")

st.markdown('<p class="big-q">Where and when the risk builds up</p>', unsafe_allow_html=True)
d = d.assign(hour=(d["planned_arrival_min"] // 60).astype(int))
pv = d.pivot_table(index="district", columns="hour", values="pred_late_prob", aggfunc="sum", fill_value=0, observed=True)
pv = pv.loc[pv.sum(axis=1).sort_values(ascending=False).index]
fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{h:02d}:00" for h in pv.columns], y=pv.index.astype(str),
                           colorscale=[[0, "#F4F5F1"], [0.35, "#F3D79A"], [1, "#B8461B"]],
                           hovertemplate="%{y}, %{x}<br>Expected late deliveries: %{z:.1f}<extra></extra>", colorbar=dict(title="Expected<br>late")))
chart_layout(fig, 60 + 34 * len(pv), xaxis_title="Planned arrival time")
st.plotly_chart(fig, width="stretch")
st.caption("Each square adds up the late risk of the deliveries in that district and hour. Darker means more deliveries expected to miss their window.")

st.markdown('<p class="big-q">Stores to call ahead</p>', unsafe_allow_html=True)
top = d.sort_values("pred_late_prob", ascending=False).head(12)
st.dataframe(pd.DataFrame({
    "Store": top["outlet_id"].astype(str), "Brand": top["brand"].astype(str), "District": top["district"].astype(str),
    "Planned arrival": top["planned_arrival_time"], "Window closes": top["window_close_time"],
    "Expected unloading (min)": top["pred_service_min"].round(0).astype(int), "Chance of being late": top["pred_late_prob"]}),
    hide_index=True, width="stretch",
    column_config={"Chance of being late": st.column_config.ProgressColumn("Chance of being late", min_value=0, max_value=1, format="percent")})
st.page_link("views/x_route.py", label="Try a fix in the route simulator")
