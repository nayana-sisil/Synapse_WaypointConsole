import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BRAND, INK, MUTED, ROOT, RUST, TEAL, chart_layout, demand, section_intro, task1_models, validation

section_intro("How we built it", "From raw delivery records to three sets of answers. Follow the pipeline stage by stage: "
              "what we did, why we did it, and what came out.")

STAGES = [
    ("Raw data", "17 files"), ("Join and check", "0 errors"), ("Build labels", "2 targets"), ("Clean", "436 rows capped"),
    ("Features", "43 inputs"), ("Time split", "no peeking"), ("Train models", "LightGBM"), ("Forecast demand", "10 weeks"),
    ("Plan the peak day", "integer program"), ("Deliver", "3 files, 1 app"),
]
strip = "".join(
    f'<div style="background:{"#17202B" if i < 7 else ("#1F7A6C" if i < 9 else "#D98C0A")};color:#fff;border-radius:10px;padding:12px 12px 10px;position:relative">'
    f'<div style="font-size:0.75rem;opacity:.75">Stage {i + 1}</div><div style="font-weight:700;font-size:0.98rem;line-height:1.2;margin:2px 0 4px">{n}</div>'
    f'<div style="font-size:0.8rem;opacity:.85">{s}</div></div>'
    for i, (n, s) in enumerate(STAGES))
st.markdown(f'<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:6px 0 4px">{strip}</div>', unsafe_allow_html=True)
st.caption("Dark: delivery time models (Task 1). Teal: demand forecast and peak day plan (Task 2A and 2B). Amber: outputs.")

# ---------- Sankey: where every record went
st.markdown('<p class="big-q">Where every record went</p>', unsafe_allow_html=True)
nodes = ["Training orders 92,307", "Test orders 5,014", "Dispatched, with route legs 91,894", "Never dispatched 413",
         "Fit period 80,991", "Validation 10,903", "Scored by final models 5,014", "Weekly demand history 97,321"]
src = [0, 0, 2, 2, 1, 0, 1]
tgt = [2, 3, 4, 5, 6, 7, 7]
val = [91894, 413, 80991, 10903, 5014, 92307, 5014]
col = ["rgba(23,32,43,0.30)", "rgba(184,70,27,0.45)", "rgba(23,32,43,0.30)", "rgba(31,122,108,0.45)", "rgba(217,140,10,0.45)",
       "rgba(31,122,108,0.25)", "rgba(31,122,108,0.25)"]
fig = go.Figure(go.Sankey(arrangement="snap",
                          node=dict(label=nodes, pad=22, thickness=18, color=[INK, INK, INK, RUST, "#3A4550", TEAL, AMBER, TEAL],
                                    line=dict(width=0)),
                          link=dict(source=src, target=tgt, value=val, color=col,
                                    hovertemplate="%{source.label} to %{target.label}<br>%{value:,} orders<extra></extra>")))
chart_layout(fig, 380)
fig.update_layout(font=dict(size=13))
st.plotly_chart(fig, width="stretch")
st.caption("Every order, including deferred and never dispatched ones, counts as demand for the forecast. Only dispatched orders have "
           "real arrival and leave times, so only they can teach the delivery time models.")

# ---------- Stage explorer
st.markdown('<p class="big-q">Walk through each stage</p>', unsafe_allow_html=True)
stage = st.segmented_control("Stage", [f"{i + 1}. {n}" for i, (n, _) in enumerate(STAGES)], default="1. Raw data", label_visibility="collapsed")
stage = stage or "1. Raw data"
k = int(stage.split(".")[0])


def two(left, right_fn, ratio=(2, 3)):
    c1, c2 = st.columns(ratio)
    with c1:
        st.markdown(left)
    with c2:
        right_fn()


if k == 1:
    def viz():
        files = pd.DataFrame([["deliveries_train", 92307, "Training"], ["route_legs_train", 91894, "Training"], ["task1_test_inputs", 5014, "Test"],
                              ["route_legs_test", 5014, "Test"], ["road_conditions", 10920, "Reference"], ["calendar", 910, "Reference"],
                              ["traffic_speed", 576, "Reference"], ["outlets", 120, "Reference"], ["task2b orders", 85, "Test"],
                              ["vehicles", 60, "Reference"], ["task2a weeks", 60, "Test"], ["district_travel", 12, "Reference"]])
        files.columns = ["File", "Rows", "Kind"]
        files = files.sort_values("Rows")
        fig = go.Figure(go.Bar(x=files.Rows, y=files.File, orientation="h", text=[f"{r:,}" for r in files.Rows], textposition="outside",
                               marker_color=[{"Training": INK, "Test": AMBER, "Reference": TEAL}[x] for x in files.Kind]))
        chart_layout(fig, 400, xaxis=dict(type="log", title="Rows (log scale)", range=[0.8, 5.5]))
        st.plotly_chart(fig, width="stretch")
    two("**What we had.** Two years of orders (Jan 2024 to Feb 2026), one row per order, and the matching route legs with "
        "planned and actual times. Reference tables describe the 120 outlets, 60 vehicles, the calendar, traffic and road conditions.\n\n"
        "**The catch.** Neither target we had to predict was in the data. The actual times exist only in the training route legs.\n\n"
        "Navy: training. Amber: test. Teal: reference tables.", viz)
elif k == 2:
    def viz():
        st.dataframe(pd.DataFrame([["Order to route leg", "route_id + seq", "91,894 of 91,894 matched"],
                                   ["Outlet agrees", "order vs leg", "0 mismatches"], ["Dates agree", "route date vs dispatch date", "0 mismatches"],
                                   ["Leave time meaning", "next departure vs leave time", "equal on 100% of legs"],
                                   ["Clock times", "HH:MM to minutes after midnight", "02:30 to 21:55, no midnight crossing"]],
                                  columns=["Check", "How", "Result"]), hide_index=True, width="stretch")
    two("**What we did.** Converted every clock time to minutes after midnight (05:30 becomes 330), then joined each order to "
        "exactly one route leg.\n\n**Why it matters.** One silent join error poisons every label after it. The leave time check "
        "proved that `leave_outlet_time` is the moment handling finished, which the service label depends on.", viz)
elif k == 3:
    def viz():
        fig = go.Figure()
        fig.add_shape(type="rect", x0=330, x1=450, y0=-0.5, y1=1.5, fillcolor="#E3EEEA", line_width=0, layer="below")
        fig.add_bar(y=["Naive"], x=[50], base=[300], orientation="h", marker_color="#B9C1C7", width=0.45)
        fig.add_bar(y=["Ours"], x=[30], base=[300], orientation="h", marker=dict(color="rgba(0,0,0,0)", line=dict(color=MUTED, width=2)), width=0.45)
        fig.add_bar(y=["Ours"], x=[20], base=[330], orientation="h", marker_color=TEAL, width=0.45)
        for x, t in [(300, "arrives 05:00"), (330, "window opens 05:30"), (350, "leaves 05:50")]:
            fig.add_annotation(x=x, y=1.55, text=t, showarrow=False, font=dict(size=12, color=MUTED))
        chart_layout(fig, 230, showlegend=False, barmode="overlay", xaxis=dict(range=[290, 380], showticklabels=False))
        st.plotly_chart(fig, width="stretch")
        st.code("service_min = leave_outlet_min - max(arrival_min, window_open_min)\nlate = arrival_min > window_close_min", language="python")
    two("**Service time.** An early vehicle waits for the window, and waiting is not handling. So handling starts at the later of "
        "arrival and window opening. 4.4% of vehicles arrived early; for them the naive label adds 20 minutes.\n\n"
        "**Late.** The actual arrival is after the window closes. 19.6% of deliveries were late: Fresh 20.4%, Style 5.5%, Tech 2.5%.", viz)
elif k == 4:
    def viz():
        v = validation()
        fig = go.Figure()
        for b, cap in [("Fresh", 64), ("Style", 200), ("Tech", 227)]:
            x = v[v.brand == b].service_min
            fig.add_box(x=x, name=b, marker_color=BRAND[b], boxpoints="outliers", orientation="h")
            fig.add_scatter(x=[cap, cap], y=[b, b], mode="markers", marker=dict(symbol="line-ns-open", size=28, color=RUST, line=dict(width=3)),
                            name=f"{b} cap {cap} min", showlegend=False, hovertemplate=f"{b} cap: {cap} min<extra></extra>")
        chart_layout(fig, 300, showlegend=False, xaxis_title="Service time (minutes), red marks = training cap")
        st.plotly_chart(fig, width="stretch")
    two("**What we did.** Kept every real delivery, including 255 that took over two hours. For training only, each brand is capped at "
        "its own 99.5th percentile (Fresh 64, Style 200, Tech 227 minutes), which touched 436 rows (0.47%).\n\n"
        "**Why per brand.** Two hours is normal for a Tech appliance and extreme for a Fresh delivery. Models are always scored on the uncapped values.", viz)
elif k == 5:
    def viz():
        _, late, meta = task1_models()
        gain = pd.Series(late.feature_importance("gain"), index=late.feature_name())
        groups = {"Expected delay (ours)": ["exp_slack", "exp_delay", "exp_travel", "travel_overrun"],
                  "Plan": ["planned_slack_min", "planned_early_min", "planned_arrival_min", "planned_depart_min", "planned_travel_duration_min", "service_allowance_min", "route_start_min"],
                  "Route": ["seq", "n_stops", "stops_left", "distance_km", "cum_distance", "depart_hour"],
                  "Outlet": ["outlet_id", "dock_type", "parking_constraint", "window_open_min", "window_close_min", "window_len", "outlet_svc_mean", "outlet_svc_median", "outlet_late_rate", "district", "depot"],
                  "Order": ["brand", "order_units", "order_weight_kg", "order_volume_m3", "temp_requirement", "is_deferred", "vehicle_type", "vehicle_temp"],
                  "Conditions": ["speed_index", "disruption_index", "monsoon", "dow", "is_payday", "festival_ramp", "is_holiday"]}
        ids, labels, parents, values, colors = [], [], [], [], []
        palette = {"Expected delay (ours)": AMBER, "Plan": INK, "Route": "#3A6B8C", "Outlet": TEAL, "Order": "#7A8791", "Conditions": "#8A6FA8"}
        for g, fs in groups.items():
            ids.append(g); labels.append(g); parents.append(""); values.append(float(gain.reindex(fs).fillna(0).sum())); colors.append(palette[g])
            for f in fs:
                if f in gain:
                    ids.append(f"{g}/{f}"); labels.append(f); parents.append(g); values.append(float(gain[f])); colors.append(palette[g])
        fig = go.Figure(go.Treemap(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total", marker=dict(colors=colors),
                                   texttemplate="%{label}<br>%{percentRoot:.1%}", hovertemplate="%{label}<br>%{percentRoot:.1%} of the late model's gain<extra></extra>"))
        chart_layout(fig, 420)
        st.plotly_chart(fig, width="stretch")
    two("**Only what is known before departure.** 43 inputs in six groups. Actual times never enter as features, because the test set does not have them.\n\n"
        "**Our idea: expected delay.** The plan assumes clear roads and standard handling. Along each route we add up the travel overrun "
        "(from traffic and road disruption) and the handling overrun at earlier stops (from each outlet's history). Planned slack minus that delay is expected slack.\n\n"
        "The map shows how much each group drives the late model. Click a group to zoom in.", viz)
elif k == 6:
    def viz():
        fig = go.Figure()
        for name, a, b, c in [("Fit (train)", "2024-01-01", "2025-11-14", INK), ("Validate", "2025-11-15", "2026-02-14", TEAL), ("Test (predict)", "2026-02-16", "2026-03-28", AMBER)]:
            fig.add_bar(y=["Timeline"], x=[pd.Timestamp(b) - pd.Timestamp(a)], base=[pd.Timestamp(a)], orientation="h", name=name, marker_color=c,
                        hovertemplate=f"{name}: {a} to {b}<extra></extra>")
        chart_layout(fig, 200, barmode="overlay", legend=dict(orientation="h", y=1.3, x=0), xaxis=dict(type="date"), yaxis=dict(showticklabels=False))
        st.plotly_chart(fig, width="stretch")
    two("**Split by time, not at random.** The test period comes after training, so we checked the models the same way: fit on everything "
        "up to 14 Nov 2025, score the next three months.\n\n**Why.** A random split lets the model see the future of each route and outlet, "
        "and gives scores that look better than reality. Outlet history for the validation period is also built from the fit period only.", viz)
elif k == 7:
    def viz():
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Bar(x=["Planning allowance", "Outlet average", "LightGBM"], y=[7.42, 5.79, 4.00], marker_color=["#B9C1C7", "#9AA5AE", TEAL],
                               text=["7.42", "5.79", "4.00"], textposition="outside"))
        chart_layout(fig, 300, title=dict(text="Service time MAE (min)", x=0), yaxis=dict(range=[0, 8.5]))
        c1.plotly_chart(fig, width="stretch")
        fig = go.Figure(go.Bar(x=["Same rate for all", "Slack groups", "LightGBM"], y=[0.5, 0.866, 0.976], marker_color=["#B9C1C7", "#9AA5AE", AMBER],
                               text=["0.500", "0.866", "0.976"], textposition="outside"))
        chart_layout(fig, 300, title=dict(text="Late AUC", x=0), yaxis=dict(range=[0, 1.1]))
        c2.plotly_chart(fig, width="stretch")
    two("**Baselines first.** A model is only useful if it beats simple rules, so we measured those first.\n\n"
        "**Two LightGBM models**, trained from scratch: a regressor for service time and a classifier for late. Early stopping on the "
        "validation set chose the number of trees (302 and 271). Then both were retrained on all history with 10% more trees.\n\n"
        "Late probabilities are well calibrated: when the model says 30%, about 30% are late.", viz)
elif k == 8:
    def viz():
        hist, fc, bt, fest = demand()
        h = hist[(hist.depot == "Peliyagoda") & (hist.brand == "Fresh")].sort_values("t").tail(16)
        f = fc[(fc.depot == "Peliyagoda") & (fc.brand == "Fresh")].sort_values("iso_week")
        level = h.total.tail(8).mean()
        fig = go.Figure()
        fig.add_scatter(x=list(h.iso_week.astype(str) + "/" + h.iso_year.astype(str).str[2:]), y=h.total, name="History", line=dict(color=INK, width=3))
        fig.add_scatter(x=list(f.iso_week.astype(str) + "/26"), y=f.pred_total_volume_m3, name="Forecast", line=dict(color=TEAL, width=3, dash="dot"), mode="lines+markers")
        fig.add_hline(y=level, line=dict(color=AMBER, dash="dash"), annotation_text="Recent level (8 week mean)", annotation_position="bottom left")
        chart_layout(fig, 320, legend=dict(orientation="h", y=1.15, x=0), yaxis_title="m³ per week, Peliyagoda Fresh", xaxis_title="ISO week / year")
        st.plotly_chart(fig, width="stretch")
    two("**Count all demand.** 97,321 orders, including deferred and never dispatched ones, grouped by the week the store asked for. "
        "That gives 6 series (2 depots × 3 brands) of 117 weeks.\n\n**Predict a ratio.** Each future week is predicted as a ratio to the "
        "recent level, using that week's calendar (delivery days, paydays, festival build up). One model learns all six series together.\n\n"
        "**Backtested from four past dates**: 7.0% error on total volume against 8.7% for a simple average.", viz)
elif k == 9:
    def viz():
        steps = [("85 orders, 28 vehicles", "#3A4550"), ("Who can carry what", "#3A4550"), ("Stage A: chilled on 4 reefers", TEAL),
                 ("Stage B: dry goods on dry trucks", TEAL), ("Check all 7 rules", AMBER), ("77 served, 8 wait", AMBER)]
        html = "".join(f'<div style="background:{c};color:#fff;border-radius:10px;padding:12px;font-weight:600"><span style="opacity:.7;font-size:.8rem">Step {i + 1}</span><br>{t}</div>'
                       for i, (t, c) in enumerate(steps))
        st.markdown(f'<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:8px 0 16px">{html}</div>', unsafe_allow_html=True)
        st.latex(r"\max \sum s_o\,x_{ovk},\quad s_o = 10 + \text{volume}_o + 3(\text{days waited}_o - 1)")
        st.markdown("Subject to: one brand and district per trip, chilled on reefers, vans for van only outlets, home depot, no split orders, "
                    "weight and volume limits, two trips and the 270 or 480 minute budgets.")
    two("**No machine learning here.** It is an allocation puzzle, so we used integer programming (PuLP with the CBC solver), which finds "
        "the best plan that obeys every rule.\n\n**The bottleneck.** Five of nine reefers were in the workshop, and refrigerated driving "
        "time ran out before space did. The solver proves that serving every chilled order is impossible.\n\n"
        "**The policy.** Never skip a store twice in a row, keep reefers for chilled goods, then serve the most stores and volume.", viz)
else:
    def viz():
        st.dataframe(pd.DataFrame([["submission_task1.csv", "5,014 deliveries", "service minutes and late probability"],
                                   ["submission_task2a.csv", "60 depot, brand and week rows", "total and chilled m³"],
                                   ["submission_task2b.csv", "85 orders", "served or deferred, vehicle and trip"],
                                   ["Saved models", "6 files", "reloaded to reproduce every submission exactly"],
                                   ["This console", "11 pages", "the same models, running live"]],
                                  columns=["Output", "Size", "Contents"]), hide_index=True, width="stretch")
    two("**Final run.** Both Task 1 models were retrained on all history; the forecast model on everything up to 2026 week 13.\n\n"
        "**Proof it is reproducible.** The final notebook reloads the saved models, prepares the raw test files from scratch, and "
        "reproduces all three submission files exactly. This console loads the same models and gives the same predictions.", viz)

st.image(str(ROOT / "assets" / "01_preprocessing_pipeline.png"), caption="The full preprocessing pipeline in one picture")
