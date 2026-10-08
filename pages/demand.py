import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BRAND, INK, MUTED, TEAL, chart_layout, demand, tile

hist, fc, bt, fest = demand()
st.title("Demand outlook")
st.markdown('<p class="lede">Ordered volume for the next 10 weeks, per depot and brand, so fleet planners can line up '
            'vehicles, drivers and refrigerated space before the festival weeks.</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
depot = c1.selectbox("Depot", ["Peliyagoda", "Kandy"])
brand = c2.selectbox("Brand", ["Fresh", "Style", "Tech"])
h = hist[(hist.depot == depot) & (hist.brand == brand)].sort_values("t").tail(40)
f = fc[(fc.depot == depot) & (fc.brand == brand)].sort_values(["iso_year", "iso_week"])
peak = f.loc[f.pred_total_volume_m3.idxmax()]
low = f.loc[f.pred_total_volume_m3.idxmin()]
recent = h["total"].tail(8).mean()

k1, k2, k3 = st.columns(3)
with k1: tile(f"{f.pred_total_volume_m3.sum():,.0f} m³", "forecast over the next 10 weeks")
with k2: tile(f"{peak.pred_total_volume_m3:,.0f} m³", f"busiest week: week {int(peak.iso_week)}, {peak.pred_total_volume_m3 / recent - 1:+.0%} on the recent average", "alert")
with k3: tile(f"{low.pred_total_volume_m3:,.0f} m³", f"quietest week: week {int(low.iso_week)}, {int(low.op_days)} delivery days")

fig = go.Figure()
x_h = pd.to_datetime(h["week_start"]); x_f = pd.to_datetime(f["week_start"])
for ws in pd.to_datetime(pd.concat([h[h.festivals > 0]["week_start"], f[f.festivals > 0]["week_start"]])):
    fig.add_vrect(x0=ws, x1=ws + pd.Timedelta(days=7), fillcolor="#F6E3BF", opacity=0.6, line_width=0, layer="below")
fig.add_scatter(x=x_h, y=h["total"], name="Total, actual", line=dict(color=INK, width=2.5))
fig.add_scatter(x=[x_h.iloc[-1]] + list(x_f), y=[h["total"].iloc[-1]] + list(f["pred_total_volume_m3"]), name="Total, forecast",
                line=dict(color=INK, width=2.5, dash="dot"), mode="lines+markers")
if brand == "Fresh":
    fig.add_scatter(x=x_h, y=h["chilled"], name="Chilled, actual", line=dict(color=TEAL, width=2.5))
    fig.add_scatter(x=[x_h.iloc[-1]] + list(x_f), y=[h["chilled"].iloc[-1]] + list(f["pred_chilled_volume_m3"]), name="Chilled, forecast",
                    line=dict(color=TEAL, width=2.5, dash="dot"), mode="lines+markers")
fig.add_vline(x=x_f.iloc[0] - pd.Timedelta(days=3.5), line=dict(color=MUTED, dash="dash", width=1))
chart_layout(fig, 430, legend=dict(orientation="h", y=1.1, x=0), yaxis_title="m³ per week")
st.plotly_chart(fig, width="stretch")
st.caption("Shaded weeks hold a festival. The forecast knows each week's delivery days, paydays and festival build up in advance.")

table = pd.DataFrame({"Week": f["iso_week"].astype(int), "Starts": pd.to_datetime(f["week_start"]).dt.strftime("%d %b"),
                      "Delivery days": f["op_days"].astype(int), "Total (m³)": f["pred_total_volume_m3"].round(1)})
if brand == "Fresh":
    table["Chilled (m³)"] = f["pred_chilled_volume_m3"].round(1).values
table["Festival week"] = (f["festivals"] > 0).map({True: "Yes", False: ""}).values
st.dataframe(table, hide_index=True, width="stretch")

st.subheader("How much to trust it")
bt = bt.set_index("target")
st.markdown(f"Tested from four past dates, including the same season last year. Average error (WAPE): "
            f"**{bt.loc['total', 'lightgbm']:.1%}** for total volume against {bt.loc['total', 'naive']:.1%} for a simple recent average, "
            f"and **{bt.loc['chilled', 'lightgbm']:.1%}** for chilled volume against {bt.loc['chilled', 'naive']:.1%}.")
