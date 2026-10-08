"""Feature pipeline copied from Synapse_FinalNotebook.ipynb (Parts 1 and 2)."""
import numpy as np
import pandas as pd

def to_minutes(series):
    parts = series.astype("string").str.split(":", expand=True)
    return pd.to_numeric(parts[0]) * 60 + pd.to_numeric(parts[1])

def add_minute_columns(df):
    for col in [c for c in df.columns if c.endswith("_time")]:
        df[col.replace("_time", "_min")] = to_minutes(df[col])
    return df

LEG_COLS_TRAIN = ["leg_id", "date", "route_id", "seq", "from_point", "to_outlet", "distance_km",
                  "planned_depart_min", "planned_travel_duration_min",
                  "actual_depart_min", "actual_travel_duration_min", "arrival_min", "leave_outlet_min",
                  "monsoon", "dow"]

def join_orders_to_legs(orders, legs, leg_cols):
    dispatched = orders[orders["route_id"].notna()].copy()
    dispatched["seq"] = dispatched["seq_in_route"].astype(int)
    return dispatched.merge(legs[leg_cols], on=["route_id", "seq"], how="left", validate="one_to_one")

def build_labels(df):
    df = df.copy()
    df["service_min"] = df["leave_outlet_min"] - np.maximum(df["arrival_min"], df["window_open_min"])
    df["late"] = (df["arrival_min"] > df["window_close_min"]).astype(int)
    return df

def prepare(df, outlets, allowance, traffic, roads):
    df = df.merge(outlets[["outlet_id", "dock_type", "parking_constraint", "mall_window"]], on="outlet_id", how="left")
    df = df.merge(allowance, on=["brand", "dock_type"], how="left")
    df["planned_slack_min"] = df["window_close_min"] - df["planned_arrival_min"]
    df["depart_hour"] = (df["planned_depart_min"] // 60).astype(int)
    df = df.merge(traffic.rename(columns={"hour": "depart_hour"}), on=["district", "depart_hour", "monsoon"], how="left")
    df = df.merge(roads, on=["district", "date"], how="left")
    return df

def outlet_stats_from(history):
    return (history.groupby("outlet_id")
            .agg(outlet_svc_mean=("service_min", "mean"), outlet_svc_median=("service_min", "median"),
                 outlet_late_rate=("late", "mean")).reset_index())

def build_features(df, outlet_stats, calendar):
    df = df.merge(calendar[["date", "is_payday", "festival_ramp", "is_holiday", "is_weekend"]], on="date", how="left")
    df["window_len"] = df["window_close_min"] - df["window_open_min"]
    df["planned_early_min"] = df["window_open_min"] - df["planned_arrival_min"]
    df["is_deferred"] = (df["dispatch_status"] == "deferred").astype(int)
    df = df.sort_values(["route_id", "seq"]).reset_index(drop=True)
    g = df.groupby("route_id")
    df["n_stops"] = g["seq"].transform("size")
    df["stops_left"] = df["n_stops"] - df["seq"] - 1
    df["route_start_min"] = g["planned_depart_min"].transform("first")
    df["cum_distance"] = g["distance_km"].cumsum()
    df["planned_dwell"] = g["planned_depart_min"].shift(-1) - df["planned_arrival_min"]
    df = df.merge(outlet_stats, on="outlet_id", how="left").sort_values(["route_id", "seq"]).reset_index(drop=True)
    return add_delay_features(df)

def add_delay_features(df):
    """Expected travel and handling overruns, added up along each route."""
    df = df.sort_values(["route_id", "seq"]).reset_index(drop=True)
    expected_service = df["outlet_svc_mean"].fillna(df["service_allowance_min"])
    df["exp_travel"] = df["planned_travel_duration_min"] * (100 / df["speed_index"]) * (100 / df["disruption_index"])
    df["travel_overrun"] = df["exp_travel"] - df["planned_travel_duration_min"]
    df["svc_overrun"] = (expected_service - df["planned_dwell"]).fillna(0)
    g = df.groupby("route_id")
    travel_so_far = g["travel_overrun"].cumsum()
    service_before = g["svc_overrun"].shift(1).fillna(0)
    df["exp_delay"] = travel_so_far + service_before.groupby(df["route_id"]).cumsum()
    df["exp_slack"] = df["planned_slack_min"] - df["exp_delay"]
    return df

def set_categories(df, cat_features, categories):
    df = df.copy()
    for c in cat_features:
        df[c] = pd.Categorical(df[c], categories=categories[c])
    return df
