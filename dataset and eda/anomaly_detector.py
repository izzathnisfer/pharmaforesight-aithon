"""
PharmaForesight — Anomaly Detection Engine
Team: MatriXplorers | University of Moratuwa

Detects sudden demand spikes using rolling Z-score method.
Flags any week where demand deviates > 2 std deviations from
the rolling window mean, and generates procurement alerts.
"""

import pandas as pd
import numpy as np
import os

DATA_FILE    = "data/pharmacy_orders.csv"
SIGNAL_FILE  = "data/health_signals.csv"
OUTPUT_FILE  = "data/anomalies.csv"
ALERT_FILE   = "data/procurement_alerts.csv"

ROLLING_WINDOW  = 6     # weeks to compute rolling baseline
Z_THRESHOLD     = 2.0   # std deviations to flag as anomaly
LEAD_TIME_WEEKS = 2     # recommended procurement lead time


# ── Load Data ──────────────────────────────────────────────────────────────────

def load_data():
    orders  = pd.read_csv(DATA_FILE,   parse_dates=["date"])
    signals = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])
    signals["dengue_alert_level"] = signals["dengue_alert_level"].fillna("LOW")
    signals["flu_alert_level"]    = signals["flu_alert_level"].fillna("LOW")
    return orders, signals


# ── Rolling Z-Score Anomaly Detection ─────────────────────────────────────────

def detect_anomalies(orders: pd.DataFrame) -> pd.DataFrame:
    """
    For each SKU × Region time series, compute rolling mean & std,
    then flag rows where |Z-score| > Z_THRESHOLD.
    """
    results = []

    for (sku_id, region), group in orders.groupby(["sku_id", "region"]):
        grp = group.sort_values("date").copy()

        grp["rolling_mean"] = (grp["units_ordered"]
                               .shift(1)
                               .rolling(window=ROLLING_WINDOW, min_periods=3)
                               .mean())
        grp["rolling_std"]  = (grp["units_ordered"]
                               .shift(1)
                               .rolling(window=ROLLING_WINDOW, min_periods=3)
                               .std())

        # Avoid divide-by-zero on flat series (chronic meds)
        grp["rolling_std"] = grp["rolling_std"].replace(0, np.nan).fillna(1)

        grp["z_score"] = (grp["units_ordered"] - grp["rolling_mean"]) / grp["rolling_std"]

        grp["is_anomaly"]      = grp["z_score"].abs() > Z_THRESHOLD
        grp["anomaly_type"]    = grp["z_score"].apply(
            lambda z: "SPIKE" if z > Z_THRESHOLD else ("DROP" if z < -Z_THRESHOLD else "NORMAL")
        )
        grp["deviation_pct"]   = ((grp["units_ordered"] - grp["rolling_mean"])
                                  / grp["rolling_mean"] * 100).round(1)

        results.append(grp)

    df = pd.concat(results).sort_values(["date", "region", "sku_id"])
    return df


# ── Severity Scoring ───────────────────────────────────────────────────────────

def severity_label(z: float) -> str:
    az = abs(z)
    if az >= 4.0:  return "CRITICAL"
    if az >= 3.0:  return "HIGH"
    if az >= 2.0:  return "MEDIUM"
    return "NORMAL"


# ── Procurement Alert Generator ────────────────────────────────────────────────

def generate_alerts(anomaly_df: pd.DataFrame,
                    signals: pd.DataFrame) -> pd.DataFrame:
    """
    For each anomaly flagged as SPIKE, generate a procurement alert
    with recommended reorder quantity and urgency level.
    """
    spikes = anomaly_df[anomaly_df["anomaly_type"] == "SPIKE"].copy()

    # Merge health signals for context
    spikes = spikes.merge(
        signals[["date","dengue_alert_level","flu_alert_level","dengue_cases_national","flu_cases_national"]],
        on="date", how="left"
    )

    spikes["severity"]         = spikes["z_score"].apply(severity_label)
    spikes["reorder_qty"]      = (spikes["units_ordered"] * 1.25 * LEAD_TIME_WEEKS).astype(int)
    spikes["reorder_by_date"]  = spikes["date"] + pd.Timedelta(weeks=LEAD_TIME_WEEKS)

    spikes["alert_reason"] = spikes.apply(lambda r: (
        "Dengue outbreak signal" if r.get("dengue_alert_level") in ["HIGH","MEDIUM"]
        else "Flu season spike" if r.get("flu_alert_level") in ["HIGH","MEDIUM"]
        else "Demand spike detected"
    ), axis=1)

    alerts = spikes[[
        "date","region","sku_id","medicine_name","category",
        "units_ordered","rolling_mean","z_score","deviation_pct",
        "severity","alert_reason",
        "reorder_qty","reorder_by_date",
        "dengue_alert_level","flu_alert_level",
    ]].copy()

    alerts.columns = [
        "date","region","sku_id","medicine_name","category",
        "actual_units","baseline_units","z_score","deviation_pct",
        "severity","alert_reason",
        "recommended_reorder_qty","reorder_by_date",
        "dengue_alert_level","flu_alert_level",
    ]

    alerts["baseline_units"] = alerts["baseline_units"].round(0).astype("Int64")
    alerts["z_score"]        = alerts["z_score"].round(2)

    return alerts.sort_values(["severity","date"], ascending=[True, False])


# ── Main ───────────────────────────────────────────────────────────────────────

def run_anomaly_detection():
    print("📂  Loading data...")
    orders, signals = load_data()

    print("🔍  Running rolling Z-score anomaly detection...")
    anomaly_df = detect_anomalies(orders)

    total       = len(anomaly_df.dropna(subset=["z_score"]))
    n_anomalies = int(anomaly_df["is_anomaly"].sum())
    n_spikes    = int((anomaly_df["anomaly_type"] == "SPIKE").sum())
    n_drops     = int((anomaly_df["anomaly_type"] == "DROP").sum())

    anomaly_df.to_csv(OUTPUT_FILE, index=False)

    print("🚨  Generating procurement alerts...")
    alerts = generate_alerts(anomaly_df, signals)
    alerts.to_csv(ALERT_FILE, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*57)
    print("✅  Anomaly Detection Complete")
    print("="*57)
    print(f"  Total weeks analysed : {total:,}")
    print(f"  Anomalies flagged    : {n_anomalies:,}  ({n_anomalies/total*100:.1f}%)")
    print(f"    ↑ Spikes           : {n_spikes:,}")
    print(f"    ↓ Drops            : {n_drops:,}")
    print(f"  Procurement alerts   : {len(alerts):,}")
    print(f"  Output (full)        : {OUTPUT_FILE}")
    print(f"  Output (alerts)      : {ALERT_FILE}")
    print("="*57)

    # Severity breakdown
    sev = alerts.groupby("severity").size().reindex(
        ["CRITICAL","HIGH","MEDIUM"], fill_value=0)
    print("\n🚦  Alert severity breakdown:")
    icons = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡"}
    for level, count in sev.items():
        print(f"    {icons[level]} {level:<10} {count:>4} alerts")

    # Top regions
    print("\n🗺️   Top 5 regions by alert count:")
    top_regions = alerts.groupby("region").size().sort_values(ascending=False).head(5)
    for region, count in top_regions.items():
        print(f"    {region:<20} {count:>4} alerts")

    # Most alerted SKUs
    print("\n💊  Top 5 medicines by alert count:")
    top_skus = alerts.groupby("medicine_name").size().sort_values(ascending=False).head(5)
    for med, count in top_skus.items():
        print(f"    {med:<38} {count:>4} alerts")

    # Sample critical alerts
    critical = alerts[alerts["severity"].isin(["CRITICAL","HIGH"])].head(5)
    if not critical.empty:
        print("\n🔴  Sample HIGH / CRITICAL alerts:")
        for _, row in critical.iterrows():
            print(f"    [{row['severity']}] {row['date'].strftime('%Y-%m-%d')} | "
                  f"{row['region']:<15} | {row['medicine_name'][:28]:<28} | "
                  f"Z={row['z_score']:+.1f} | +{row['deviation_pct']:.0f}% | "
                  f"Reorder: {row['recommended_reorder_qty']:,} units")

    print("\n✅  Ready for the Streamlit dashboard!")
    return anomaly_df, alerts


if __name__ == "__main__":
    run_anomaly_detection()
