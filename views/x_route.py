import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, INK, TEAL, chart_layout, hhmm, routes, section_intro, what_if

df = routes()
section_intro("Route simulator", "Pick a route and change the plan. Both models score every stop again, live.")
c1, c2, c3 = st.columns([1, 1, 2])
day = c1.selectbox("Delivery date", sorted(df["date"].unique()))
brand = c2.selectbox("Brand", ["All", "Fresh", "Style", "Tech"])
d = df[df["date"] == day]
if brand != "All":
    d = d[d["brand"] == brand]
summary = (d.groupby("route_id", observed=True)
           .agg(brand=("brand", "first"), district=("district", "first"), stops=("seq", "size"), risk=("pred_late_prob", "max"))
           .sort_values("risk", ascending=False).reset_index())
labels = {r.route_id: f"{r.route_id}: {r.brand}, {r.district}, {r.stops} stops, highest risk {r.risk:.0%}" for r in summary.itertuples()}
route_id = c3.selectbox("Route (riskiest first)", list(labels), format_func=labels.get)
r = d[d["route_id"] == route_id].sort_values("seq").reset_index(drop=True)

w1, w2 = st.columns(2)
delay = w1.slider("Leave the depot earlier or later (minutes)", -60, 60, 0, step=5, help="Negative means leave earlier.")
base_dis = int(r["disruption_index"].iloc[0])
dis = w2.slider("Road conditions in the district (100 = clear roads)", 40, 100, base_dis, step=5,
                help="From road_conditions.csv for this date. Lower means roadworks, flooding or an incident.")
changed = delay != 0 or dis != base_dis
new = what_if(r, delay, dis) if changed else r.copy()

b, a = r["pred_late_prob"].sum(), new["pred_late_prob"].sum()
k1, k2, k3 = st.columns(3)
k1.metric("Expected late stops", f"{a:.1f}", f"{a - b:+.1f}" if changed else None, delta_color="inverse")
k2.metric("Highest stop risk", f"{new['pred_late_prob'].max():.0%}", f"{(new['pred_late_prob'].max() - r['pred_late_prob'].max()) * 100:+.0f} points" if changed else None, delta_color="inverse")
k3.metric("Predicted handling on route", f"{new['pred_service_min'].sum():.0f} min")

fig = go.Figure()
y = [f"Stop {i + 1}  {o}" for i, o in enumerate(r["outlet_id"].astype(str))]
for i, row in r.iterrows():
    fig.add_shape(type="rect", x0=row.window_open_min, x1=row.window_close_min, y0=i - 0.38, y1=i + 0.38, fillcolor="#E3EEEA", line=dict(width=0), layer="below")
frames = [(r, "Plan", 0.3), (new, "After change", 1.0)] if changed else [(r, "Plan", 1.0)]
for frame, name, op in frames:
    arr = frame["planned_arrival_min"] + frame["exp_delay"].clip(lower=0)
    start = np.maximum(arr, frame["window_open_min"])
    colors = [AMBER if p >= 0.5 else ("#C9A227" if p >= 0.2 else TEAL) for p in frame["pred_late_prob"]]
    fig.add_bar(y=y, x=frame["pred_service_min"], base=start, orientation="h", marker_color=colors, opacity=op, width=0.42, name=name,
                customdata=np.stack([frame["pred_late_prob"], frame["pred_service_min"], frame["exp_slack"]], axis=1),
                hovertemplate="%{y}<br>Handling %{customdata[1]:.0f} min<br>Late risk %{customdata[0]:.0%}<br>Expected slack %{customdata[2]:.0f} min<extra>" + name + "</extra>")
    fig.add_scatter(y=y, x=frame["planned_arrival_min"], mode="markers", marker=dict(color=INK, size=10, symbol="circle-open", line=dict(width=2), opacity=op), hoverinfo="skip")
    fig.add_scatter(y=y, x=arr, mode="markers", marker=dict(color=INK, size=9, symbol="diamond", opacity=op), hoverinfo="skip")
lo = min(r["window_open_min"].min(), new["planned_arrival_min"].min()) - 20
hi = max(r["window_close_min"].max(), (new["planned_arrival_min"] + new["exp_delay"] + new["pred_service_min"]).max(),
         (r["planned_arrival_min"] + r["exp_delay"] + r["pred_service_min"]).max()) + 20
ticks = list(range(int(lo // 30 * 30), int(hi) + 30, 30))
chart_layout(fig, 90 + 62 * len(r), barmode="overlay", showlegend=False, xaxis=dict(range=[lo, hi], tickvals=ticks, ticktext=[hhmm(t) for t in ticks], side="top"),
             yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")
st.caption("Pale band: delivery window. Circle: planned arrival. Diamond: expected arrival after traffic, road conditions and earlier stops. "
           "Bar: predicted handling, coloured by late risk (teal under 20%, gold 20 to 50%, amber over 50%). Faded bars show the original plan.")

table = pd.DataFrame({"Stop": r["seq"] + 1, "Outlet": r["outlet_id"].astype(str), "Dock": r["dock_type"].astype(str), "Units": r["order_units"],
                      "Window": r["window_open_time"] + " to " + r["window_close_time"], "Planned arrival": [hhmm(m) for m in new["planned_arrival_min"]],
                      "Expected slack (min)": new["exp_slack"].round(0).astype(int), "Handling (min)": new["pred_service_min"].round(1),
                      "Late risk before": r["pred_late_prob"], "Late risk now": new["pred_late_prob"]})
st.dataframe(table, hide_index=True, width="stretch",
             column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="percent") for c in ["Late risk before", "Late risk now"]})
