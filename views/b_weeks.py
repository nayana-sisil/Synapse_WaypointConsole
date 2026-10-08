import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, INK, TEAL, chart_layout, demand, peak_day, section_intro, tile

hist, fc, bt, fest = demand()
_, _, vehicles, _, _, _ = peak_day()
section_intro("The weeks ahead", "How much each depot will need to move over the next 10 weeks, shown as truckloads, "
              "so extra vehicles and refrigerated space can be lined up before the busy weeks.")
depot = st.radio("Depot", ["Peliyagoda", "Kandy"], horizontal=True)
f = fc[fc.depot == depot].copy()
v = vehicles[vehicles.depot == depot]
reefer_cap = v[(v.temp == "reefer") & (v.type == "truck")]["volume_cap_m3"].mean()
truck_cap = v[(v.temp == "ambient") & (v.type == "truck")]["volume_cap_m3"].mean()
wk = (f.groupby(["iso_week", "week_start", "festivals", "op_days"])
      .agg(total=("pred_total_volume_m3", "sum"), chilled=("pred_chilled_volume_m3", "sum")).reset_index())
wk["ambient"] = wk["total"] - wk["chilled"]
wk["reefer_loads"] = wk["chilled"] / reefer_cap
wk["truck_loads"] = wk["ambient"] / truck_cap
h = hist[hist.depot == depot].groupby("t").agg(total=("total", "sum")).tail(8)
busy = wk.loc[wk.total.idxmax()]

k1, k2, k3 = st.columns(3)
with k1: tile(f"{wk.total.sum():,.0f} m³", "to deliver over the 10 weeks")
with k2: tile(f"Week {int(busy.iso_week)}", f"busiest week, {busy.total / h.total.mean() - 1:+.0%} on a normal week", "alert")
with k3: tile(f"{wk.reefer_loads.max():.0f}", "full refrigerated truckloads in the peak week")

fig = go.Figure()
labels = [f"Week {w}<br>{pd.Timestamp(s):%d %b}" for w, s in zip(wk.iso_week, wk.week_start)]
fig.add_bar(x=labels, y=wk.reefer_loads, name="Refrigerated truckloads", marker_color=TEAL,
            hovertemplate="%{x}<br>%{y:.0f} refrigerated loads<extra></extra>")
fig.add_bar(x=labels, y=wk.truck_loads, name="Dry truckloads", marker_color="#9AA5AE",
            hovertemplate="%{x}<br>%{y:.0f} dry loads<extra></extra>")
for i, r in wk.reset_index(drop=True).iterrows():
    if r.festivals > 0:
        fig.add_annotation(x=labels[i], y=r.reefer_loads + r.truck_loads, text="Festival", showarrow=False, yshift=14, font=dict(color=AMBER, size=12))
    if r.op_days < 6:
        fig.add_annotation(x=labels[i], y=0, text=f"{int(r.op_days)} delivery days", showarrow=False, yshift=-12, font=dict(size=11, color="#5B6670"))
chart_layout(fig, 430, barmode="stack", legend=dict(orientation="h", y=1.1, x=0), yaxis_title="Full truckloads per week")
st.plotly_chart(fig, width="stretch")
st.caption(f"Truckloads are approximate: forecast volume divided by the average capacity of {depot}'s trucks "
           f"({reefer_cap:.0f} m³ refrigerated, {truck_cap:.0f} m³ dry). Real loads are limited by weight, timing and routes too.")

st.markdown('<p class="big-q">What to plan for</p>', unsafe_allow_html=True)
pre = wk[wk.total > h.total.mean() * 1.15]
low = wk[wk.op_days < 6]
lines = []
if len(pre): lines.append(f"**Book extra capacity for week{'s' if len(pre) > 1 else ''} {', '.join(str(int(w)) for w in pre.iso_week)}**: volume is more than 15% above a normal week as stores stock up for festivals.")
if len(low): lines.append(f"**Expect a dip in week{'s' if len(low) > 1 else ''} {', '.join(str(int(w)) for w in low.iso_week)}**: fewer delivery days because of public holidays, so orders bunch up around them.")
lines.append(f"**Refrigerated space is the tight resource**: chilled goods need up to {wk.reefer_loads.max():.0f} refrigerated truckloads in a single week.")
for l in lines:
    st.markdown("- " + l)
