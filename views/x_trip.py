import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, INK, TEAL, chart_layout, peak_day, section_intro, trip_minutes

orders, fleet, vehicles, travel, allow, alloc = peak_day()
section_intro("Trip planner", "Sketch a vehicle's day and check it against the time budget and the vehicle's limits, "
              "using the same trip time rules as the peak day optimiser.")
brand = st.radio("Brand", ["Fresh", "Style", "Tech"], horizontal=True)
budget = 270 if brand == "Fresh" else 480
dist_list = list(travel.index)


def trip_inputs(n, default_district):
    st.markdown(f"**Trip {n}**")
    district = st.selectbox("District", dist_list, index=dist_list.index(default_district), key=f"d{n}")
    c1, c2, c3 = st.columns(3)
    docks = (["rear_dock"] * c1.number_input("Rear dock stops", 0, 12, 2, key=f"r{n}")
             + ["street"] * c2.number_input("Street stops", 0, 12, 1, key=f"s{n}")
             + ["mall_bay"] * c3.number_input("Mall bay stops", 0, 12, 0, key=f"m{n}"))
    return district, docks


c1, c2 = st.columns(2)
with c1:
    d1, k1 = trip_inputs(1, "Gampaha")
with c2:
    use2 = st.toggle("Add a second trip", True)
    d2, k2 = trip_inputs(2, "Colombo") if use2 else (None, [])

parts = []
for n, d, k in [(1, d1, k1), (2, d2, k2)]:
    if d and k:
        out = travel.loc[d, "depot_to_district_freeflow_min"]
        between = travel.loc[d, "inter_stop_freeflow_min"] * (len(k) - 1)
        handling = trip_minutes(brand, d, k, travel, allow) - out - between
        parts += [(f"Trip {n} to {d}: drive out", out, "#9AA5AE"), (f"Trip {n}: between stops", between, "#C9CFD4"), (f"Trip {n}: unloading", handling, TEAL if n == 1 else "#5FA89C")]
total = sum(p[1] for p in parts)
fig = go.Figure()
for name, val, col in parts:
    fig.add_bar(y=["Vehicle day"], x=[val], name=name, orientation="h", marker_color=col, text=f"{val:.0f}", textposition="inside",
                hovertemplate=f"{name}: {val:.0f} min<extra></extra>")
fig.add_vline(x=budget, line=dict(color=AMBER, width=4), annotation_text=f"{budget} minute budget", annotation_position="top")
chart_layout(fig, 220, barmode="stack", xaxis=dict(range=[0, max(budget, total) * 1.1], title="Minutes"), legend=dict(orientation="h", y=-0.5, x=0))
st.plotly_chart(fig, width="stretch")
if total <= budget:
    st.success(f"Fits: {total:.0f} of {budget} minutes used, {budget - total:.0f} to spare. The return drive is already allowed for in the budget.")
else:
    st.error(f"Does not fit: {total:.0f} minutes against a {budget} minute budget, {total - budget:.0f} over. Drop a stop or move a trip to a nearer district.")

st.markdown('<p class="big-q">Which vehicles could carry the load?</p>', unsafe_allow_html=True)
a, b, c = st.columns(3)
vol = a.number_input("Load volume (m³)", 0.0, 60.0, 12.0, 0.5)
wt = b.number_input("Load weight (kg)", 0.0, 9000.0, 2500.0, 100.0)
need = c.multiselect("Needs", ["Refrigeration", "Van access"])
av = fleet.merge(vehicles, on="vehicle_id")
av = av[av.depot == "Peliyagoda"]
ok = av[(av.volume_cap_m3 >= vol) & (av.weight_cap_kg >= wt)]
if "Refrigeration" in need: ok = ok[ok.temp == "reefer"]
if "Van access" in need: ok = ok[ok.type == "van"]
st.dataframe(ok[["vehicle_id", "type", "temp", "volume_cap_m3", "weight_cap_kg", "status"]]
             .rename(columns={"vehicle_id": "Vehicle", "type": "Type", "temp": "Temperature", "volume_cap_m3": "Volume (m³)", "weight_cap_kg": "Weight (kg)", "status": "Peak day status"}),
             hide_index=True, width="stretch")
st.caption(f"{(ok.status == 'available').sum()} available Peliyagoda vehicles fit this load on the peak day.")
