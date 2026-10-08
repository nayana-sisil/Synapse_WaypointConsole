import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, INK, ROOT, TEAL, chart_layout, check_allocation, demand, peak_day, section_intro

hist, fc, bt, fest = demand()
orders, fleet, vehicles, travel, allow, alloc = peak_day()
section_intro("Forecast and optimiser", "Task 2A uses one LightGBM model across all six depot and brand series. "
              "Task 2B uses integer programming, not machine learning.")
tab1, tab2, tab3 = st.tabs(["Demand forecast", "Peak day optimiser", "Architecture"])
with tab1:
    st.markdown("Each sample is a (series, origin, horizon) triple. The target is the week's volume divided by the mean of the "
                "8 weeks before the origin, so large and small series share one scale. Features: horizon, depot, brand, the target "
                "week's calendar (operating days, paydays, festival ramp, holidays, monsoon) and last year's ratio. Predicted ratios are clipped to 0.3 to 2.5.")
    st.latex(r"\hat{y}_{s,o+h} = \text{level}_{s,o}\times \operatorname{clip}\big(f(h,\ s,\ \text{calendar}_{o+h},\ r^{\,ly}_{s,o+h}),\ 0.3,\ 2.5\big)")
    b = bt.melt(id_vars="target", var_name="method", value_name="WAPE")
    names = {"naive": "Naive (8 week mean)", "seasonal_naive": "Same week last year", "lightgbm": "LightGBM", "blend": "Blend"}
    fig = go.Figure()
    for m, col in [("naive", "#C9CFD4"), ("seasonal_naive", "#9AA5AE"), ("blend", "#5FA89C"), ("lightgbm", TEAL)]:
        x = b[b.method == m]
        fig.add_bar(x=x.target, y=x.WAPE, name=names[m], marker_color=col, text=[f"{w:.1%}" for w in x.WAPE], textposition="outside")
    chart_layout(fig, 360, barmode="group", yaxis=dict(title="Average WAPE over 4 backtest origins", tickformat=".0%"), legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Backtest origins: 2025 weeks 13, 30, 40 and 2026 week 3. The first covers the same season as the real forecast, including New Year.")
with tab2:
    st.markdown("Two stage mixed integer program, solved with PuLP and CBC. Stage A packs chilled orders onto the 4 available reefers "
                "(the bottleneck), with orders skipped yesterday forced in. Stage B packs ambient orders onto ambient vehicles.")
    st.latex(r"\max \sum_{o,v,k} s_o\,x_{ovk} \;-\; 0.001\sum z_{vkg}, \qquad s_o = 10 + \text{volume}_o + 3\,(\text{days waited}_o - 1)")
    st.markdown("""
| Constraint | Form |
|---|---|
| Each order at most once, never split | Σ over v,k of x(o,v,k) ≤ 1 |
| One brand and district per trip | Σ over g of z(v,k,g) ≤ 1, and x(o,v,k) ≤ z(v,k,g(o)) |
| Weight and volume | Σ over o of weight(o)·x(o,v,k) ≤ cap(v), same for volume |
| Trip 2 only after trip 1 | Σ z(v,2,g) ≤ Σ z(v,1,g) |
| Time budget per vehicle | Σ over trips of [outbound + (stops minus 1)·between + Σ handling] ≤ 270 Fresh, 480 Style and Tech |
| Compatibility | x exists only if temperature, van access, depot and size allow it |
""")
    if st.button("Run the independent rule checker on the submitted plan", type="primary"):
        a = alloc[["order_ref", "decision", "vehicle_id", "trip_id"]].merge(orders, on="order_ref")
        for rule, ok, detail in check_allocation(a, vehicles, fleet, travel, allow):
            (st.success if ok else st.error)(f"{'Passed' if ok else 'Failed'}: {rule}. {detail}.")
    st.markdown("Each chosen deferral was tested by forcing it into the plan and solving again: every alternative serves fewer orders or less volume. "
                "Serving all chilled orders is proven infeasible.")
with tab3:
    st.image(str(ROOT / "assets" / "02_model_architecture.png"), caption="Models for the three tasks")
    st.image(str(ROOT / "assets" / "03_deployment_approach.png"), caption="Proposed deployment")
