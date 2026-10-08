import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BRAND, INK, TEAL, card, chart_layout, peak_day, section_intro, tile

orders, fleet, vehicles, travel, allow, alloc = peak_day()
a = alloc[["order_ref", "decision", "vehicle_id", "trip_id"]].merge(orders, on="order_ref")
served, waits = a[a.decision == "served"], a[a.decision == "deferred"]
section_intro("Peak day decisions", "A festival is a week away, Fresh orders are rising, and five of the nine refrigerated "
              "vehicles are being repaired. Not every order can go. Here is what we sent, what waits, and why.")
k1, k2, k3 = st.columns(3)
with k1: tile(f"{served.outlet_id.nunique()} of {a.outlet_id.nunique()}", "stores get a delivery today", "good")
with k2: tile(f"{len(waits)} orders", "wait for tomorrow's run", "alert")
with k3: tile("0", "stores left without a delivery two days in a row", "good")

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown('<p class="big-q">Who gets served</p>', unsafe_allow_html=True)
    g = a.assign(kind=a.brand + ", " + a.temp_requirement).groupby(["kind", "decision"]).order_volume_m3.sum().unstack(fill_value=0)
    fig = go.Figure()
    fig.add_bar(y=g.index, x=g.get("served", 0), name="Delivered today", orientation="h", marker_color=TEAL,
                hovertemplate="%{y}<br>Delivered: %{x:.0f} m³<extra></extra>")
    fig.add_bar(y=g.index, x=g.get("deferred", 0), name="Waits for tomorrow", orientation="h", marker_color=AMBER,
                hovertemplate="%{y}<br>Waits: %{x:.0f} m³<extra></extra>")
    chart_layout(fig, 300, barmode="stack", xaxis_title="Volume (m³)", legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown('<p class="big-q">Why some orders wait</p>', unsafe_allow_html=True)
    st.markdown("**Chilled goods ran out of fridge time, not fridge space.** Only four refrigerated vehicles were available, "
                "and the far districts (Galle, Matara, Kurunegala, Puttalam) take two to three hours just to reach. "
                "Seven chilled orders could not fit before stores open at 8 AM.\n\n"
                "**One clothing order is simply too big.** At 41 m³ it is larger than any vehicle in the fleet, and orders are never split.")

st.markdown('<p class="big-q">The three rules we followed</p>', unsafe_allow_html=True)
r1, r2, r3 = st.columns(3)
with r1: card("Never skip a store twice", "Every store that missed yesterday's delivery is served today. All 10 of them.", TEAL)
with r2: card("Fridges carry only chilled goods", "Refrigerated vehicles are the scarce resource, so dry goods travel on ordinary trucks.", INK)
with r3: card("Then reach as many stores as possible", "After that, the plan serves the most stores and the most goods, giving extra weight to stores that waited longer.", AMBER)

st.markdown('<p class="big-q">What would fix it</p>', unsafe_allow_html=True)
st.markdown("- **Get one refrigerated truck back from the workshop**: it adds 270 minutes of refrigerated driving time, the exact resource that ran out.\n"
            "- **Split the oversized clothing order into two loads**, with the store's agreement.\n"
            "- **Tomorrow, the seven waiting chilled stores go first**, because of rule one.")
