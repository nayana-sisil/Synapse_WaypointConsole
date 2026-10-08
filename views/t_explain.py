import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, INK, TEAL, chart_layout, routes, section_intro, task1_models
from pipeline import set_categories

svc, late, meta = task1_models()
F = meta["features"]
df = routes()
section_intro("Why the model decides", "Global importance shows what the models rely on overall. The per stop breakdown uses "
              "LightGBM's built in SHAP contributions to explain one prediction, feature by feature.")

c1, c2 = st.columns(2)
for col, model, title, color in [(c1, late, "Late classifier", AMBER), (c2, svc, "Service time regressor", TEAL)]:
    gain = pd.Series(model.feature_importance("gain"), index=model.feature_name()).sort_values().tail(12)
    gain = gain / model.feature_importance("gain").sum()
    fig = go.Figure(go.Bar(x=gain.values, y=gain.index, orientation="h", marker_color=color, hovertemplate="%{y}: %{x:.1%} of total gain<extra></extra>"))
    chart_layout(fig, 420, title=dict(text=title, x=0), xaxis=dict(tickformat=".0%", title="Share of total gain"))
    col.plotly_chart(fig, width="stretch")
st.caption("exp_slack and exp_delay are our engineered features: planned slack minus the delay expected to build up along the route.")

st.markdown('<p class="big-q">Explain one stop</p>', unsafe_allow_html=True)
a, b, c = st.columns([1, 2, 1])
day = a.selectbox("Date", sorted(df.date.unique()))
d = df[df.date == day].sort_values("pred_late_prob", ascending=False)
pick = b.selectbox("Stop (riskiest first)", d.delivery_id, format_func=lambda i: (lambda r: f"{i}: {r.outlet_id}, {r.district}, stop {r.seq + 1}, late risk {r.pred_late_prob:.0%}")(d.set_index('delivery_id').loc[i]))
which = c.radio("Model", ["Late risk", "Service time"])
row = set_categories(d[d.delivery_id == pick], meta["cat_features"], meta["categories"])[F]
model = late if which == "Late risk" else svc
contrib = model.predict(row, pred_contrib=True)[0]
base, phi = contrib[-1], pd.Series(contrib[:-1], index=F)
top = phi.reindex(phi.abs().sort_values(ascending=False).index).head(10)
rest = phi.sum() - top.sum()
vals = row.iloc[0]
def fmt(x):
    if isinstance(x, (int, float, np.floating, np.integer)):
        return f"{x:.0f}" if float(x).is_integer() or abs(x) >= 10 else f"{x:.2f}"
    return str(x)


labels = [f"{k} = {fmt(vals[k])}" for k in top.index] + ["all other features"]
fig = go.Figure(go.Waterfall(orientation="h", y=["baseline"] + labels + ["prediction"], x=[base] + list(top.values) + [rest, 0],
                             measure=["absolute"] + ["relative"] * (len(top) + 1) + ["total"],
                             increasing=dict(marker_color=AMBER if which == "Late risk" else "#C77B4A"), decreasing=dict(marker_color=TEAL),
                             totals=dict(marker_color=INK), connector=dict(line=dict(color="#C9CFC9"))))
unit = "log odds" if which == "Late risk" else "minutes"
chart_layout(fig, 520, yaxis=dict(autorange="reversed"), xaxis_title=f"Contribution ({unit})")
st.plotly_chart(fig, width="stretch")
if which == "Late risk":
    p = 1 / (1 + np.exp(-(base + phi.sum())))
    st.caption(f"The contributions add up from the average stop (baseline) to this stop's score: {p:.1%} chance of arriving late. "
               "Amber pushes the risk up, teal pulls it down.")
else:
    st.caption(f"Contributions in minutes add up from the average stop to this prediction: {base + phi.sum():.1f} minutes of handling.")
