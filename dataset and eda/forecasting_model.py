"""
PharmaForesight — Forecasting Engine
Team: MatriXplorers | University of Moratuwa

Ensemble of Prophet + XGBoost to forecast weekly medicine demand
at SKU × Region level, 4 weeks ahead.
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import os
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_FILE   = "data/pharmacy_orders.csv"
SIGNAL_FILE = "data/health_signals.csv"
MODEL_DIR   = "models"
FORECAST_WEEKS = 4   # how many weeks ahead to predict


# ── Load & Merge Data ──────────────────────────────────────────────────────────

def load_data():
    orders  = pd.read_csv(DATA_FILE,   parse_dates=["date"])
    signals = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])
    # health_signals uses dengue_cases_national / flu_cases_national — normalise to index
    signals["dengue_index"] = signals["dengue_cases_national"] / signals["dengue_cases_national"].max()
    signals["flu_index"]    = signals["flu_cases_national"]    / signals["flu_cases_national"].max()
    df = orders.merge(signals[["date","dengue_index","flu_index","school_term_active"]],
                      on="date", how="left", suffixes=("","_sig"))
    for col in ["dengue_index","flu_index","school_term_active"]:
        if col+"_sig" in df.columns:
            df[col] = df[col+"_sig"].fillna(df[col])
            df.drop(columns=[col+"_sig"], inplace=True)
    df["school_term_active"] = df["school_term_active"].astype(int)
    return df


# ── Feature Engineering ────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]       = df["date"].dt.month
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]     = df["date"].dt.quarter
    df["lag_1"]       = df.groupby(["region","sku_id"])["units_ordered"].shift(1)
    df["lag_2"]       = df.groupby(["region","sku_id"])["units_ordered"].shift(2)
    df["lag_4"]       = df.groupby(["region","sku_id"])["units_ordered"].shift(4)
    df["rolling_4w"]  = (df.groupby(["region","sku_id"])["units_ordered"]
                           .transform(lambda x: x.shift(1).rolling(4).mean()))
    df.dropna(inplace=True)
    return df


# ── Prophet Model (per SKU × Region) ──────────────────────────────────────────

def train_prophet(series: pd.DataFrame) -> tuple:
    """
    Train a Prophet model on a single SKU×Region time series.
    Returns (model, forecast_df).
    """
    prophet_df = series[["date","units_ordered"]].rename(
        columns={"date":"ds","units_ordered":"y"})

    m = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    # Add external regressors
    m.add_regressor("dengue_index")
    m.add_regressor("flu_index")
    m.add_regressor("school_term_active")

    prophet_df["dengue_index"]       = series["dengue_index"].values
    prophet_df["flu_index"]          = series["flu_index"].values
    prophet_df["school_term_active"] = series["school_term_active"].values

    m.fit(prophet_df)

    # Build future frame (last date + FORECAST_WEEKS)
    future = m.make_future_dataframe(periods=FORECAST_WEEKS, freq="W")
    last = series.iloc[-1]
    future["dengue_index"]       = last["dengue_index"]
    future["flu_index"]          = last["flu_index"]
    future["school_term_active"] = last["school_term_active"]

    forecast = m.predict(future)
    return m, forecast


# ── XGBoost Model (global across all SKU×Regions) ─────────────────────────────

def train_xgboost(df: pd.DataFrame) -> tuple:
    """
    Train a single XGBoost model using all regions & SKUs.
    Categorical features are label-encoded.
    Returns (model, label_encoders).
    """
    df = df.copy()
    le_region   = LabelEncoder().fit(df["region"])
    le_sku      = LabelEncoder().fit(df["sku_id"])
    le_category = LabelEncoder().fit(df["category"])

    df["region_enc"]   = le_region.transform(df["region"])
    df["sku_enc"]      = le_sku.transform(df["sku_id"])
    df["category_enc"] = le_category.transform(df["category"])

    FEATURES = ["region_enc","sku_enc","category_enc",
                "month","week","quarter",
                "dengue_index","flu_index","school_term_active",
                "lag_1","lag_2","lag_4","rolling_4w"]
    TARGET = "units_ordered"

    # Train / validation split — last 8 weeks as validation
    cutoff = df["date"].max() - pd.Timedelta(weeks=8)
    train  = df[df["date"] <= cutoff]
    val    = df[df["date"] >  cutoff]

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(train[FEATURES], train[TARGET],
            eval_set=[(val[FEATURES], val[TARGET])],
            verbose=False)

    val_preds = xgb.predict(val[FEATURES])
    mape = mean_absolute_percentage_error(val[TARGET], val_preds)

    encoders = {"region": le_region, "sku": le_sku, "category": le_category}
    return xgb, encoders, mape, FEATURES


# ── Ensemble Forecast ──────────────────────────────────────────────────────────

def ensemble_forecast(prophet_pred: float, xgb_pred: float,
                      prophet_weight: float = 0.55) -> float:
    """Weighted average of Prophet and XGBoost predictions."""
    return prophet_weight * prophet_pred + (1 - prophet_weight) * xgb_pred


# ── Save / Load Helpers ────────────────────────────────────────────────────────

def save_models(xgb_model, encoders, prophet_models: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/xgb_model.pkl",      "wb") as f: pickle.dump(xgb_model, f)
    with open(f"{MODEL_DIR}/encoders.pkl",        "wb") as f: pickle.dump(encoders, f)
    with open(f"{MODEL_DIR}/prophet_models.pkl",  "wb") as f: pickle.dump(prophet_models, f)
    print(f"✅  Models saved to /{MODEL_DIR}/")


def load_models():
    with open(f"{MODEL_DIR}/xgb_model.pkl",     "rb") as f: xgb   = pickle.load(f)
    with open(f"{MODEL_DIR}/encoders.pkl",       "rb") as f: enc   = pickle.load(f)
    with open(f"{MODEL_DIR}/prophet_models.pkl", "rb") as f: proph = pickle.load(f)
    return xgb, enc, proph


# ── Main Training Pipeline ─────────────────────────────────────────────────────

def train_all():
    print("📂  Loading data...")
    df_raw = load_data()
    df     = add_features(df_raw)

    print(f"    {len(df):,} records loaded | "
          f"{df['region'].nunique()} regions | {df['sku_id'].nunique()} SKUs\n")

    # ── XGBoost (global model) ─────────────────────────────────────────────────
    print("🤖  Training XGBoost (global model)...")
    xgb_model, encoders, mape_xgb, FEATURES = train_xgboost(df)
    print(f"    XGBoost Validation MAPE: {mape_xgb:.2%}\n")

    # ── Prophet (one model per SKU × Region) ──────────────────────────────────
    keys = df_raw.groupby(["sku_id","region"]).groups.keys()
    total = len(keys)
    prophet_models  = {}
    prophet_forecasts = {}

    print(f"🔮  Training Prophet models ({total} SKU×Region combinations)...")
    for i, (sku_id, region) in enumerate(keys, 1):
        series = df_raw[(df_raw["sku_id"]==sku_id) & (df_raw["region"]==region)].copy()
        model, forecast = train_prophet(series)
        prophet_models[(sku_id, region)]   = model
        prophet_forecasts[(sku_id, region)] = forecast
        if i % 20 == 0 or i == total:
            print(f"    [{i}/{total}] done")

    # ── Build ensemble forecast table ──────────────────────────────────────────
    print("\n📊  Building ensemble forecasts...")
    results = []

    for (sku_id, region), forecast in prophet_forecasts.items():
        sku_info = next(s for s in __import__("data_generator").SKUS if s["sku_id"]==sku_id)
        future_rows = forecast.tail(FORECAST_WEEKS)

        for _, row in future_rows.iterrows():
            # XGBoost future features (use last known lags from df)
            last_known = df[(df["sku_id"]==sku_id) & (df["region"]==region)].iloc[-1]
            xgb_features = {
                "region_enc":          encoders["region"].transform([region])[0],
                "sku_enc":             encoders["sku"].transform([sku_id])[0],
                "category_enc":        encoders["category"].transform([sku_info["category"]])[0],
                "month":               row["ds"].month,
                "week":                row["ds"].isocalendar()[1],
                "quarter":             (row["ds"].month - 1) // 3 + 1,
                "dengue_index":        last_known["dengue_index"],
                "flu_index":           last_known["flu_index"],
                "school_term_active":  last_known["school_term_active"],
                "lag_1":               last_known["units_ordered"],
                "lag_2":               last_known.get("lag_1", last_known["units_ordered"]),
                "lag_4":               last_known.get("lag_2", last_known["units_ordered"]),
                "rolling_4w":          last_known.get("rolling_4w", last_known["units_ordered"]),
            }
            xgb_pred    = float(xgb_model.predict(pd.DataFrame([xgb_features])[FEATURES])[0])
            prophet_pred = max(0, float(row["yhat"]))
            ensemble     = ensemble_forecast(prophet_pred, xgb_pred)

            results.append({
                "forecast_date":    row["ds"].strftime("%Y-%m-%d"),
                "region":           region,
                "sku_id":           sku_id,
                "medicine_name":    sku_info["name"],
                "category":         sku_info["category"],
                "prophet_forecast": round(prophet_pred),
                "xgb_forecast":     round(max(0, xgb_pred)),
                "ensemble_forecast":round(max(0, ensemble)),
                "lower_bound":      round(max(0, float(row["yhat_lower"]))),
                "upper_bound":      round(max(0, float(row["yhat_upper"]))),
            })

    forecast_df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    forecast_df.to_csv("data/forecasts.csv", index=False)

    # ── Save models ────────────────────────────────────────────────────────────
    save_models(xgb_model, encoders, prophet_models)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("✅  Training Complete — PharmaForesight Forecasts")
    print("="*55)
    print(f"  XGBoost MAPE     : {mape_xgb:.2%}  (lower = better)")
    print(f"  Forecast horizon : {FORECAST_WEEKS} weeks ahead")
    print(f"  Forecast records : {len(forecast_df):,}")
    print(f"  Output           : data/forecasts.csv")
    print("="*55)

    print("\n📋  Sample forecasts:")
    sample_cols = ["forecast_date","region","medicine_name","ensemble_forecast","lower_bound","upper_bound"]
    print(forecast_df[sample_cols].head(12).to_string(index=False))

    return forecast_df


if __name__ == "__main__":
    train_all()
