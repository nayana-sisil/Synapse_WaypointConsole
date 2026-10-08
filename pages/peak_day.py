import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BRAND, INK, MUTED, RULE, TEAL, chart_layout, check_allocation, peak_day, tile, trip_minutes

orders, fleet, vehicles, travel, allow, alloc = peak_day()
a = alloc[["order_ref", "decision", "vehicle_id", "trip_id"]].merge(orders, on="order_ref")
a = a.merge(allow, on=["brand", "dock_type"], how="left")
served, deferred = a[a.decision == "served"], a[a.decision == "deferred"]

st.title("Peak day plan")
st.markdown('<p class="lede">Festival week at Peliyagoda, with five of nine refrigerated vehicles in the workshop. '
            'This is the plan our integer program chose, and why eight orders wait.</p>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1: tile(f"{len(served)} of {len(a)}", "orders served")
with k2: tile(f"{served.outlet_id.nunique()} of {a.outlet_id.nunique()}", "outlets get a delivery")
with k3: tile(f"{served.deferred_yesterday.sum()} of {a.deferred_yesterday.sum()}", "outlets skipped yesterday are served", "good")
with k4: tile(f"{served.order_volume_m3.sum():.0f} m³", f"of {a.order_volume_m3.sum():.0f} m³ delivered")

st.subheader("Refrigerated vehicles ran out of time, not space")
vmap = vehicles.set_index("vehicle_id")
reefers = fleet.merge(vehicles, on="vehicle_id").query("status == 'available' and temp == 'reefer'")["vehicle_id"].tolist()
fig = go.Figure()
for k, color in [(1, TEAL), (2, "#5FA89C")]:
    xs, ys, txt = [], [], []
    for v in reefers:
        t = served[(served.vehicle_id == v) & (served.trip_id == k)]
        mins = trip_minutes(t.brand.iloc[0], t.district.iloc[0], list(t.dock_type), travel, allow) if len(t) else 0
        xs.append(mins); ys.append(f"{v} ({vmap.loc[v, 'type']})")
        txt.append(f"Trip {k}: {t.district.iloc[0]}, {len(t)} stop{'s' if len(t) != 1 else ''}, {mins} min" if len(t) else "")
    fig.add_bar(x=xs, y=ys, orientation="h", marker_color=color, name=f"Trip {k}", text=txt, textposition="inside", insidetextanchor="start")
fig.add_vline(x=270, line=dict(color=AMBER, width=3), annotation_text="270 minute Fresh window", annotation_position="top")
chart_layout(fig, 300, barmode="stack", xaxis_title="Fresh minutes used", legend=dict(orientation="h", y=1.18, x=0, traceorder="normal"),
             yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 300]))
st.plotly_chart(fig, width="stretch")
st.caption("Each vehicle may run two trips. Far districts are expensive: Puttalam takes 173 minutes just to reach, "
           "Kurunegala 127. Chilled demand is 182 m³; even perfectly full trips on all four reefers carry 172 m³.")

st.subheader("Why eight orders wait")
reasons = {
    "S1-078": ("Cannot be served", "40.7 m³ is bigger than the largest vehicle (38 m³), and orders are never split"),
    "S1-021": ("Our choice", "Serving it instead serves the same number of orders and 4.7 m³ less"),
    "S1-056": ("Our choice", "Serving it instead serves the same number of orders and 4.7 m³ less"),
    "S1-067": ("Our choice", "Serving it instead serves the same number of orders and 3.3 m³ less"),
    "S1-064": ("Our choice", "Serving it instead serves 1 order and 7.4 m³ less"),
    "S1-071": ("Our choice", "Serving it instead serves 1 order and 0.1 m³ less"),
    "S1-073": ("Our choice", "Serving it instead serves 1 order and 0.6 m³ less"),
    "S1-075": ("Our choice", "Serving it instead serves 1 order and 0.1 m³ less"),
}
dt = deferred.assign(Verdict=deferred.order_ref.map(lambda o: reasons.get(o, ("", ""))[0]),
                     Why=deferred.order_ref.map(lambda o: reasons.get(o, ("", ""))[1]))
st.dataframe(dt[["order_ref", "outlet_id", "brand", "district", "temp_requirement", "order_volume_m3", "Verdict", "Why"]]
             .rename(columns={"order_ref": "Order", "outlet_id": "Outlet", "brand": "Brand", "district": "District",
                              "temp_requirement": "Goods", "order_volume_m3": "Volume (m³)"}),
             hide_index=True, width="stretch")
st.markdown("The seven chilled outlets still receive their ambient order today, and they go first on tomorrow's run, "
            "because our first rule never skips an outlet two days in a row.")

st.subheader("Priority rules")
r1, r2, r3 = st.columns(3)
r1.markdown("**1. No outlet skipped twice in a row.** All 10 orders from outlets deferred yesterday are served.")
r2.markdown("**2. Refrigerated vehicles carry only chilled goods.** They are the scarce resource.")
r3.markdown("**3. Then the most outlets and volume.** Each order scores 10 + volume + 3 × extra days waited.")

st.subheader("Check the plan")
if st.button("Check all seven feasibility rules", type="primary"):
    for rule, ok, detail in check_allocation(a, vehicles, fleet, travel, allow):
        (st.success if ok else st.error)(f"{'Passed' if ok else 'Failed'}: {rule}. {detail}.")

with st.expander("Every trip in the plan"):
    trips = (served.groupby(["vehicle_id", "trip_id"])
             .agg(Brand=("brand", "first"), District=("district", "first"), Stops=("order_ref", "size"),
                  Weight_kg=("order_weight_kg", "sum"), Volume_m3=("order_volume_m3", "sum"), docks=("dock_type", list)).reset_index())
    trips["Minutes"] = [trip_minutes(b, d, k, travel, allow) for b, d, k in zip(trips.Brand, trips.District, trips.docks)]
    trips["Volume used"] = trips.Volume_m3 / trips.vehicle_id.map(vmap["volume_cap_m3"])
    st.dataframe(trips.drop(columns="docks").rename(columns={"vehicle_id": "Vehicle", "trip_id": "Trip", "Weight_kg": "Weight (kg)", "Volume_m3": "Volume (m³)"}),
                 hide_index=True, width="stretch",
                 column_config={"Volume used": st.column_config.ProgressColumn("Volume used", min_value=0, max_value=1, format="percent")})
