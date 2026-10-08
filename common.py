"""Shared look, data loading and helpers for the Waypoint Planning Console."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import add_delay_features, set_categories

ROOT = Path(__file__).parent
DATA = ROOT / "app_data"

INK, PAPER, PANEL, RULE = "#17202B", "#F4F5F1", "#FFFFFF", "#D9DDD6"
MUTED = "#5B6670"
TEAL, AMBER, RUST = "#1F7A6C", "#D98C0A", "#B8461B"
BRAND = {"Fresh": "#3E8E41", "Style": "#8A4F9E", "Tech": "#2E63B8"}
FONT = "Public Sans, system-ui, sans-serif"


def style():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;800&family=Public+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, td, th { font-family: 'Public Sans', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Archivo', 'Public Sans', sans-serif !important; letter-spacing: -0.01em; color: #17202B; }
h1 { font-weight: 800 !important; font-size: 2.6rem !important; line-height: 1.05 !important; }
h2 { font-weight: 700 !important; font-size: 1.5rem !important; margin-top: 1.2rem !important; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
.lede { font-size: 1.12rem; color: #3A4550; max-width: 62ch; margin: -0.4rem 0 1.4rem 0; }
.figure { font-family: 'Archivo', sans-serif; font-weight: 800; font-size: 2.3rem; line-height: 1; color: #17202B; font-variant-numeric: tabular-nums; }
.figure-note { color: #5B6670; font-size: 0.92rem; margin-top: 0.35rem; }
.tile { background: #FFFFFF; border: 1px solid #D9DDD6; border-radius: 10px; padding: 18px 20px 16px; height: 100%; }
.tile.alert { border-left: 6px solid #D98C0A; }
.tile.good { border-left: 6px solid #1F7A6C; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 600; }
.pill.late { background: #FBE9D0; color: #8A4E00; }
.pill.ok { background: #DDEFEA; color: #135A50; }
.pill.mid { background: #F3EBD8; color: #6B5413; }
.note { color: #5B6670; font-size: 0.9rem; }
[data-testid="stSidebar"] { background: #17202B; }
[data-testid="stSidebar"] * { color: #E8EDF2 !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(232,237,242,0.10); border-radius: 6px; }
</style>
""",
        unsafe_allow_html=True,
    )


def hhmm(m):
    m = int(round(m))
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def risk_pill(p):
    if p >= 0.5:
        return f'<span class="pill late">{p:.0%} late risk</span>'
    if p >= 0.2:
        return f'<span class="pill mid">{p:.0%} late risk</span>'
    return f'<span class="pill ok">{p:.0%} late risk</span>'


def tile(figure, note, kind=""):
    st.markdown(f'<div class="tile {kind}"><div class="figure">{figure}</div><div class="figure-note">{note}</div></div>',
                unsafe_allow_html=True)


def chart_layout(fig, height=380, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font=dict(family=FONT, color=INK, size=13), hoverlabel=dict(font_family=FONT), **kw)
    fig.update_xaxes(gridcolor="#EEF0EC", linecolor=RULE, zeroline=False)
    fig.update_yaxes(gridcolor="#EEF0EC", linecolor=RULE, zeroline=False)
    return fig


# ---------------- Task 1
@st.cache_resource
def task1_models():
    meta = json.loads((DATA / "task1_meta.json").read_text())
    return (lgb.Booster(model_file=str(DATA / "service_model.txt")),
            lgb.Booster(model_file=str(DATA / "late_model.txt")), meta)


@st.cache_data
def routes():
    _, _, meta = task1_models()
    df = pd.read_csv(DATA / "routes.csv")
    return set_categories(df, meta["cat_features"], meta["categories"])


def predict(df):
    svc, late, meta = task1_models()
    X = set_categories(df, meta["cat_features"], meta["categories"])[meta["features"]]
    out = df.copy()
    out["pred_service_min"] = np.clip(svc.predict(X), 1, None)
    out["pred_late_prob"] = np.clip(late.predict(X), 0.001, 0.999)
    return out


def what_if(route_df, depart_delay=0, disruption=None):
    """Shift the whole route later and/or set road disruption, rebuild the delay features, and score again."""
    df = route_df.copy()
    for c in ["planned_depart_min", "planned_arrival_min", "route_start_min"]:
        df[c] = df[c] + depart_delay
    df["planned_slack_min"] = df["planned_slack_min"] - depart_delay
    df["planned_early_min"] = df["planned_early_min"] - depart_delay
    if disruption is not None:
        df["disruption_index"] = disruption
    df = add_delay_features(df)
    return predict(df)


# ---------------- Task 2A
@st.cache_data
def demand():
    hist = pd.read_csv(DATA / "weekly_history.csv")
    fc = pd.read_csv(DATA / "forecast.csv")
    bt = pd.read_csv(DATA / "backtest.csv")
    fest = pd.read_csv(DATA / "festivals.csv")
    return hist, fc, bt, fest


# ---------------- Task 2B
@st.cache_data
def peak_day():
    orders = pd.read_csv(DATA / "task2b_peak_day_scenarios.csv")
    fleet = pd.read_csv(DATA / "task2b_peak_day_fleet.csv")
    vehicles = pd.read_csv(DATA / "vehicles.csv")
    travel = pd.read_csv(DATA / "district_travel.csv").set_index("district")
    allow = pd.read_csv(DATA / "service_allowance.csv")
    alloc = pd.read_csv(DATA / "allocation.csv")
    return orders, fleet, vehicles, travel, allow, alloc


def trip_minutes(brand, district, dock_types, travel, allow):
    lookup = allow.set_index(["brand", "dock_type"])["service_allowance_min"]
    n = len(dock_types)
    return (travel.loc[district, "depot_to_district_freeflow_min"]
            + travel.loc[district, "inter_stop_freeflow_min"] * (n - 1)
            + sum(lookup.loc[(brand, d)] for d in dock_types))


def check_allocation(df, vehicles, fleet, travel, allow):
    """The seven booklet rules, checked from the allocation table alone. Returns a list of (rule, passed, detail)."""
    vmap = vehicles.set_index("vehicle_id")
    served = df[df["decision"] == "served"]
    results = []
    bad = [o for _, o in served.iterrows() if o["brand"] != served[(served.vehicle_id == o.vehicle_id) & (served.trip_id == o.trip_id)]["brand"].iloc[0]
           or o["district"] != served[(served.vehicle_id == o.vehicle_id) & (served.trip_id == o.trip_id)]["district"].iloc[0]]
    results.append(("One brand and one district per trip", not bad, f"{served.groupby(['vehicle_id', 'trip_id']).ngroups} trips checked"))
    bad = served[(served.temp_requirement == "chilled") & (served.vehicle_id.map(vmap["temp"]) != "reefer")]
    results.append(("Chilled orders on refrigerated vehicles", bad.empty, f"{(served.temp_requirement == 'chilled').sum()} chilled orders checked"))
    bad = served[(served.parking_constraint == "van_only") & (served.vehicle_id.map(vmap["type"]) != "van")]
    results.append(("Van only outlets served by vans", bad.empty, f"{(served.parking_constraint == 'van_only').sum()} van only orders checked"))
    bad = served[served.depot != served.vehicle_id.map(vmap["depot"])]
    workshop = set(served.vehicle_id) & set(fleet.loc[fleet.status != "available", "vehicle_id"])
    results.append(("Home depot only, no workshop vehicles", bad.empty and not workshop, "Peliyagoda vehicles, all marked available"))
    results.append(("Orders never split", not served.order_ref.duplicated().any(), f"{len(served)} served orders, each on one trip"))
    over = []
    for (vid, k), g in served.groupby(["vehicle_id", "trip_id"]):
        if g.order_weight_kg.sum() > vmap.loc[vid, "weight_cap_kg"] or g.order_volume_m3.sum() > vmap.loc[vid, "volume_cap_m3"]:
            over.append(f"{vid} trip {k}")
    results.append(("Weight and volume within limits", not over, "every trip under both limits" if not over else ", ".join(over)))
    late = []
    for vid, g in served.groupby("vehicle_id"):
        if g.trip_id.nunique() > 2:
            late.append(f"{vid} has more than two trips")
        for brands, budget in [(["Fresh"], 270), (["Style", "Tech"], 480)]:
            mins = sum(trip_minutes(t.brand.iloc[0], t.district.iloc[0], list(t.dock_type), travel, allow)
                       for _, t in g[g.brand.isin(brands)].groupby("trip_id"))
            if mins > budget:
                late.append(f"{vid} {mins} of {budget} min")
    results.append(("Two trips at most, inside the time budgets", not late, "270 Fresh minutes, 480 Style and Tech minutes" if not late else ", ".join(late)))
    return results
