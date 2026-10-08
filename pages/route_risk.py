import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BRAND, INK, MUTED, RUST, TEAL, chart_layout, hhmm, routes, what_if

df = routes()
st.title("Route risk")
st.markdown('<p class="lede">Pick a route to see each stop against its delivery window. Then change the plan and '
            'the models score it again, live.</p>', unsafe_allow_html=True)

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

st.subheader("Try a change")
w1, w2 = st.columns(2)
delay = w1.slider("Vehicle leaves the depot late by (minutes)", 0, 60, 0, step=5)
base_dis = int(r["disruption_index"].iloc[0])
dis = w2.slider("Road conditions in the district (100 = clear roads)", 40, 100, base_dis, step=5,
                help="Taken from road_conditions.csv for this date. Lower means roadworks, flooding or an incident.")
changed = delay != 0 or dis != base_dis
new = what_if(r, delay, dis) if changed else r.copy()


def strip(base, after):
    fig = go.Figure()
    y = [f"Stop {i + 1}  {o}" for i, o in enumerate(base["outlet_id"].astype(str))]
    for i, row in base.iterrows():
        fig.add_shape(type="rect", x0=row.window_open_min, x1=row.window_close_min, y0=i - 0.38, y1=i + 0.38,
                      fillcolor="#E3EEEA", line=dict(width=0), layer="below")
    for frame, name, op in [(base, "Plan", 0.35), (after, "After your change", 1.0)] if changed else [(base, "Plan", 1.0)]:
        start = np.maximum(frame["planned_arrival_min"] + frame["exp_delay"].clip(lower=0), frame["window_open_min"])
        colors = [AMBER if p >= 0.5 else ("#C9A227" if p >= 0.2 else TEAL) for p in frame["pred_late_prob"]]
        fig.add_bar(y=y, x=frame["pred_service_min"], base=start, orientation="h", marker_color=colors, opacity=op, name=name,
                    width=0.42, customdata=np.stack([frame["pred_late_prob"], frame["pred_service_min"], frame["exp_slack"]], axis=1),
                    hovertemplate="%{y}<br>Handling %{customdata[1]:.0f} min<br>Late risk %{customdata[0]:.0%}<br>Expected slack %{customdata[2]:.0f} min<extra>" + name + "</extra>")
        fig.add_scatter(y=y, x=frame["planned_arrival_min"], mode="markers", marker=dict(color=INK, size=10, symbol="circle-open", line=dict(width=2), opacity=op),
                        name=f"Planned arrival ({name.lower()})", hovertemplate="Planned arrival %{customdata}<extra></extra>",
                        customdata=[hhmm(m) for m in frame["planned_arrival_min"]])
        fig.add_scatter(y=y, x=frame["planned_arrival_min"] + frame["exp_delay"].clip(lower=0), mode="markers", marker=dict(color=INK, size=9, symbol="diamond", opacity=op),
                        name=f"Expected arrival ({name.lower()})", hoverinfo="skip")
    lo = min(r["window_open_min"].min(), r["planned_arrival_min"].min()) - 20
    hi = max(r["window_close_min"].max(), (new["planned_arrival_min"] + new["exp_delay"] + new["pred_service_min"]).max()) + 20
    ticks = list(range(int(lo // 30 * 30), int(hi) + 30, 30))
    chart_layout(fig, 90 + 62 * len(base), barmode="overlay", showlegend=False,
                 xaxis=dict(range=[lo, hi], tickvals=ticks, ticktext=[hhmm(t) for t in ticks], side="top"),
                 yaxis=dict(autorange="reversed"))
    return fig


st.plotly_chart(strip(r, new), width="stretch")
st.caption("Pale band: the outlet's delivery window. Circle: planned arrival. Diamond: expected arrival after traffic, road conditions and earlier stops. "
           "Bar: predicted handling time, coloured by late risk (teal under 20%, gold 20 to 50%, amber over 50%).")

table = pd.DataFrame({
    "Stop": r["seq"] + 1, "Outlet": r["outlet_id"].astype(str), "Dock": r["dock_type"].astype(str),
    "Units": r["order_units"], "Window": r["window_open_time"] + " to " + r["window_close_time"],
    "Planned arrival": [hhmm(m) for m in new["planned_arrival_min"]],
    "Expected slack (min)": new["exp_slack"].round(0).astype(int),
    "Handling (min)": new["pred_service_min"].round(1),
    "Late risk": new["pred_late_prob"],
})
if changed:
    table.insert(len(table.columns) - 1, "Late risk before", r["pred_late_prob"])
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"Late risk": st.column_config.ProgressColumn("Late risk", min_value=0, max_value=1, format="percent"),
                            "Late risk before": st.column_config.ProgressColumn("Late risk before", min_value=0, max_value=1, format="percent")})
if changed:
    b, a = r["pred_late_prob"].sum(), new["pred_late_prob"].sum()
    st.info(f"Expected late stops on this route: {b:.1f} before, {a:.1f} after your change.")
