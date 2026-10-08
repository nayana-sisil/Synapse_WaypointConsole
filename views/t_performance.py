import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, log_loss, brier_score_loss

from common import AMBER, BRAND, INK, MUTED, TEAL, chart_layout, flag_stats, section_intro, tile, validation

v = validation()
section_intro("Model performance", "All numbers below come from a time based holdout: models fit on data up to 14 Nov 2025 "
              "and scored on 15 Nov 2025 to 14 Feb 2026 (10,903 deliveries), mirroring how the test set follows training.")
auc = roc_auc_score(v.late, v.pred_late_prob)
mae = (v.service_min - v.pred_service_min).abs().mean()
mae_a = (v.service_min - v.service_allowance_min).abs().mean()
k1, k2, k3, k4 = st.columns(4)
with k1: tile(f"{mae:.2f}", f"service time MAE (min), allowance baseline {mae_a:.2f}")
with k2: tile(f"{auc:.3f}", "late AUC, slack group baseline 0.866")
with k3: tile(f"{log_loss(v.late, v.pred_late_prob):.3f}", "late log loss, prior baseline 0.447")
with k4: tile(f"{brier_score_loss(v.late, v.pred_late_prob):.3f}", "Brier score")

tab1, tab2 = st.tabs(["Late classifier", "Service time regressor"])
with tab1:
    th = st.slider("Decision threshold for raising a warning", 0.05, 0.95, 0.5, 0.05)
    s = flag_stats(v, th)
    c1, c2 = st.columns([2, 3])
    with c1:
        cm = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted on time", "Predicted late"], y=["Actually on time", "Actually late"],
                                   colorscale=[[0, "#F4F5F1"], [1, TEAL]], showscale=False, text=cm, texttemplate="%{text:,}",
                                   textfont=dict(size=18)))
        chart_layout(fig, 300, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"Recall **{s['recall']:.1%}**, precision **{s['precision']:.1%}**, {s['flagged']:,} warnings raised.")
    with c2:
        fpr, tpr, _ = roc_curve(v.late, v.pred_late_prob)
        pr, rc, _ = precision_recall_curve(v.late, v.pred_late_prob)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, name=f"ROC (AUC {auc:.3f})", line=dict(color=TEAL, width=3))
        fig.add_scatter(x=rc, y=pr, name=f"Precision vs recall (AP {average_precision_score(v.late, v.pred_late_prob):.3f})", line=dict(color=AMBER, width=3))
        fig.add_scatter(x=[s["fp"] / max(1, s["fp"] + s["tn"])], y=[s["recall"]], mode="markers", marker=dict(size=14, color=INK), name="Current threshold on ROC")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash="dash", width=1), showlegend=False)
        chart_layout(fig, 340, xaxis_title="False positive rate  /  recall", yaxis_title="True positive rate  /  precision", legend=dict(orientation="h", y=1.15, x=0))
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Calibration**")
    q = pd.qcut(v.pred_late_prob.rank(method="first"), 20, labels=False)
    cal = v.groupby(q).agg(pred=("pred_late_prob", "mean"), real=("late", "mean"), n=("late", "size"))
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash="dash", width=1), name="Perfect")
    fig.add_scatter(x=cal.pred, y=cal.real, mode="lines+markers", marker=dict(size=9, color=TEAL), line=dict(color=TEAL), name="Model, 20 equal bins",
                    customdata=cal.n, hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>%{customdata} deliveries<extra></extra>")
    chart_layout(fig, 330, xaxis_title="Mean predicted probability", yaxis_title="Observed late rate", legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig, width="stretch")
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        e = pd.DataFrame({"brand": v.brand, "Model": (v.service_min - v.pred_service_min).abs(), "Allowance": (v.service_min - v.service_allowance_min).abs()})
        m = e.groupby("brand")[["Allowance", "Model"]].mean()
        fig = go.Figure()
        fig.add_bar(x=m.index, y=m.Allowance, name="Planning allowance", marker_color="#B9C1C7")
        fig.add_bar(x=m.index, y=m.Model, name="LightGBM", marker_color=TEAL)
        chart_layout(fig, 340, barmode="group", yaxis_title="MAE (minutes)", legend=dict(orientation="h", y=1.15, x=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        samp = v.sample(3000, random_state=1)
        fig = go.Figure()
        for b, col in BRAND.items():
            x = samp[samp.brand == b]
            fig.add_scatter(x=x.service_min, y=x.pred_service_min, mode="markers", name=b, marker=dict(color=col, size=5, opacity=0.45))
        lim = float(np.percentile(v.service_min, 99.5))
        fig.add_scatter(x=[0, lim], y=[0, lim], line=dict(color=MUTED, dash="dash"), showlegend=False)
        chart_layout(fig, 340, xaxis=dict(title="Actual service (min)", range=[0, lim]), yaxis=dict(title="Predicted (min)", range=[0, lim]), legend=dict(orientation="h", y=1.15, x=0))
        st.plotly_chart(fig, width="stretch")
    res = v.service_min - v.pred_service_min
    fig = go.Figure(go.Histogram(x=res.clip(-40, 40), nbinsx=80, marker_color=TEAL))
    chart_layout(fig, 280, xaxis_title="Residual: actual minus predicted (min, clipped to ±40)", yaxis_title="Deliveries")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Median absolute error {res.abs().median():.1f} min. The long right tail is the rare very slow delivery, which we kept in the data but capped per brand for training.")
