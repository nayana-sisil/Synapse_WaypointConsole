import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, TEAL, RULE, chart_layout, hhmm, routes, tile

df = routes()
dates = sorted(df["date"].unique())

st.title("Tonight's run")
st.markdown('<p class="lede">Every planned stop, scored before the trucks leave: how long it will take at the outlet, '
            'and the chance it arrives after the window closes.</p>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 1])
day = c1.selectbox("Delivery date", dates, index=0)
depot = c2.selectbox("Depot", ["Both depots", "Peliyagoda", "Kandy"])
d = df[df["date"] == day]
if depot != "Both depots":
    d = d[d["depot"] == depot]

risky = d[d["pred_late_prob"] >= 0.5]
k1, k2, k3, k4 = st.columns(4)
with k1: tile(f"{d['route_id'].nunique()}", "routes planned")
with k2: tile(f"{len(d)}", "stops to serve")
with k3: tile(f"{len(risky)}", "stops likely to arrive late", "alert" if len(risky) else "good")
with k4: tile(f"{d['pred_service_min'].sum() / 60:.0f} h", "predicted handling time at outlets")

st.subheader("Stops most likely to arrive late")
top = d.sort_values("pred_late_prob", ascending=False).head(10)
show = pd.DataFrame({
    "Route": top["route_id"].astype(str), "Stop": top["seq"] + 1, "Outlet": top["outlet_id"].astype(str),
    "Brand": top["brand"].astype(str), "District": top["district"].astype(str),
    "Planned arrival": top["planned_arrival_time"], "Window closes": top["window_close_time"],
    "Expected slack (min)": top["exp_slack"].round(0).astype(int),
    "Late risk": top["pred_late_prob"],
})
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"Late risk": st.column_config.ProgressColumn("Late risk", min_value=0, max_value=1, format="percent")})
st.page_link("pages/route_risk.py", label="Open a route and test what happens if it leaves late")

st.subheader("When the risk builds up")
d = d.assign(hour=(d["planned_arrival_min"] // 60).astype(int))
by = d.groupby("hour").agg(stops=("delivery_id", "size"), late=("pred_late_prob", "sum")).reset_index()
fig = go.Figure()
fig.add_bar(x=[f"{h:02d}:00" for h in by["hour"]], y=by["stops"] - by["late"], name="Expected on time", marker_color=TEAL)
fig.add_bar(x=[f"{h:02d}:00" for h in by["hour"]], y=by["late"], name="Expected late", marker_color=AMBER)
chart_layout(fig, 320, barmode="stack", legend=dict(orientation="h", y=1.12, x=0), yaxis_title="Stops", xaxis_title="Planned arrival hour")
st.plotly_chart(fig, width="stretch")
st.caption("Expected late = the sum of late probabilities in each hour.")
