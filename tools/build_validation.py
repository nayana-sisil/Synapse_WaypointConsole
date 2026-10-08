"""Rebuilds the validation run from notebook 02 (fit to 14 Nov 2025, validate after) and saves its predictions."""
import sys, json, joblib
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import *
DATA, SUBM = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path(__file__).resolve().parent.parent / "app_data"
G, TR = DATA / "General Data", DATA / "Training Data"
outlets = pd.read_csv(G / "outlets.csv"); calendar = pd.read_csv(G / "calendar.csv")
allowance = pd.read_csv(G / "service_allowance.csv"); traffic = pd.read_csv(G / "traffic_speed.csv"); roads = pd.read_csv(G / "road_conditions.csv")
train = build_labels(join_orders_to_legs(add_minute_columns(pd.read_csv(TR / "deliveries_train.csv")),
                                         add_minute_columns(pd.read_csv(TR / "route_legs_train.csv")), LEG_COLS_TRAIN))
train = prepare(train, outlets, allowance, traffic, roads)
caps = train.groupby("brand")["service_min"].quantile(0.995).round(0)
train["service_min_capped"] = np.minimum(train["service_min"], train["brand"].map(caps))
art = joblib.load(SUBM / "models" / "task1_artifacts.pkl")
F, CAT = art["features"], art["cat_features"]
fitp, valp = train[train.date < "2025-11-15"], train[train.date >= "2025-11-15"]
fs, vs = outlet_stats_from(fitp), outlet_stats_from(fitp)
fit = build_features(fitp, fs, calendar); val = build_features(valp, vs, calendar)
cats = {c: sorted(fit[c].dropna().unique()) for c in CAT}
fit, val = set_categories(fit, CAT, cats), set_categories(val, CAT, cats)
P = dict(learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_estimators=3000, random_state=42, verbose=-1)
reg = lgb.LGBMRegressor(objective="regression", **P).fit(fit[F], fit["service_min_capped"], eval_X=(val[F],), eval_y=(val["service_min"],), eval_metric="l1", callbacks=[lgb.early_stopping(100, verbose=False)])
clf = lgb.LGBMClassifier(objective="binary", **P).fit(fit[F], fit["late"], eval_X=(val[F],), eval_y=(val["late"],), eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
val["pred_service_min"] = np.clip(reg.predict(val[F]), 1, None); val["pred_late_prob"] = np.clip(clf.predict_proba(val[F])[:, 1], 0.001, 0.999)
print("MAE", mean_absolute_error(val.service_min, val.pred_service_min), "allowance MAE", mean_absolute_error(val.service_min, val.service_allowance_min),
      "AUC", roc_auc_score(val.late, val.pred_late_prob), "logloss", log_loss(val.late, val.pred_late_prob))
cols = ["delivery_id", "date", "brand", "district", "dock_type", "seq", "order_units", "planned_slack_min", "exp_slack", "service_min", "late",
        "pred_service_min", "pred_late_prob", "service_allowance_min", "outlet_svc_mean"]
val[cols].to_csv(OUT / "validation.csv", index=False)
# naive arrival rule for comparison (slack groups baseline)
