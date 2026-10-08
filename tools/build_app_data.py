"""Builds the small files the app reads. Run once: python tools/build_app_data.py <data_dir> <submission_dir>"""
import sys, json, joblib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import *

DATA, SUBM = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path(__file__).resolve().parent.parent / "app_data"; OUT.mkdir(exist_ok=True)
MOD = SUBM / "models"
G, TR, TE = DATA / "General Data", DATA / "Training Data", DATA / "Test Data"

outlets = pd.read_csv(G / "outlets.csv"); vehicles = pd.read_csv(G / "vehicles.csv"); calendar = pd.read_csv(G / "calendar.csv")
allowance = pd.read_csv(G / "service_allowance.csv"); traffic = pd.read_csv(G / "traffic_speed.csv")
roads = pd.read_csv(G / "road_conditions.csv"); travel = pd.read_csv(G / "district_travel.csv")

# ---------- Task 1
dtr = add_minute_columns(pd.read_csv(TR / "deliveries_train.csv")); ltr = add_minute_columns(pd.read_csv(TR / "route_legs_train.csv"))
dte = add_minute_columns(pd.read_csv(TE / "task1_test_inputs.csv")); lte = add_minute_columns(pd.read_csv(TE / "route_legs_test.csv"))
train = build_labels(join_orders_to_legs(dtr, ltr, LEG_COLS_TRAIN))
test = join_orders_to_legs(dte, lte, [c for c in LEG_COLS_TRAIN if c in lte.columns])
train = prepare(train, outlets, allowance, traffic, roads); test = prepare(test, outlets, allowance, traffic, roads)
ostats = outlet_stats_from(train)
art = joblib.load(MOD / "task1_artifacts.pkl")
feat = set_categories(build_features(test, ostats, calendar), art["cat_features"], art["categories"])
svc = joblib.load(MOD / "task1_service_model.pkl"); late = joblib.load(MOD / "task1_late_model.pkl")
feat["pred_service_min"] = np.clip(svc.predict(feat[art["features"]]), 1, None).round(2)
feat["pred_late_prob"] = np.clip(late.predict_proba(feat[art["features"]])[:, 1], 0.001, 0.999).round(4)
sub1 = pd.read_csv(SUBM / "submissions" / "submission_task1.csv").set_index("delivery_id")
chk = feat.set_index("delivery_id").loc[sub1.index]
d = (chk["pred_late_prob"] - sub1["pred_late_prob"]).abs(); e = (chk["pred_service_min"] - sub1["pred_service_min"]).abs(); print("Task 1 max diff vs submission: late", d.max(), "service", e.max(), "rows differing", (d > 0.001).sum())
keep = ["delivery_id", "date", "route_id", "seq", "outlet_id", "brand", "district", "depot", "vehicle_id", "vehicle_type", "vehicle_temp",
        "temp_requirement", "order_units", "order_volume_m3", "dock_type", "planned_arrival_time", "window_open_time", "window_close_time",
        "planned_dwell", "outlet_svc_mean"] + art["features"] + ["pred_service_min", "pred_late_prob", "svc_overrun"]
keep = list(dict.fromkeys(keep))
feat[keep].to_csv(OUT / "routes.csv", index=False)
svc.booster_.save_model(str(OUT / "service_model.txt")); late.booster_.save_model(str(OUT / "late_model.txt"))
json.dump({"features": art["features"], "cat_features": art["cat_features"], "categories": {k: list(v) for k, v in art["categories"].items()}},
          open(OUT / "task1_meta.json", "w"))

# ---------- Task 2A
a2 = joblib.load(MOD / "task2a_artifacts.pkl")
wh = a2["weekly_history"].copy(); wh["depot"] = wh["depot"].astype(str); wh["brand"] = wh["brand"].astype(str)
cw = a2["cal_week"]
wh = wh.merge(cw[["iso_year", "iso_week", "week_start", "festivals", "op_days"]], on=["iso_year", "iso_week"], how="left")
wh.to_csv(OUT / "weekly_history.csv", index=False)
fc = pd.read_csv(TE / "task2a_test_inputs.csv").merge(pd.read_csv(SUBM / "submissions" / "submission_task2a.csv"), on="row_id")
fc = fc.merge(cw[["iso_year", "iso_week", "week_start", "festivals", "op_days", "ramp_sum"]], on=["iso_year", "iso_week"], how="left")
fc.to_csv(OUT / "forecast.csv", index=False)
a2["backtest_wape"].reset_index().to_csv(OUT / "backtest.csv", index=False)
calendar[calendar["festival"].notna()][["date", "festival", "iso_year", "iso_week"]].to_csv(OUT / "festivals.csv", index=False)

# ---------- Task 2B
for f in ["task2b_peak_day_scenarios.csv", "task2b_peak_day_fleet.csv"]:
    pd.read_csv(TE / f).to_csv(OUT / f, index=False)
pd.read_csv(SUBM / "submissions" / "submission_task2b.csv").to_csv(OUT / "allocation.csv", index=False)
vehicles.to_csv(OUT / "vehicles.csv", index=False); travel.to_csv(OUT / "district_travel.csv", index=False); allowance.to_csv(OUT / "service_allowance.csv", index=False)
print("written:", sorted(p.name for p in OUT.iterdir()))
