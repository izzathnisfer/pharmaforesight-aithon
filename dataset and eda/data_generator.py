"""
PharmaForesight — Synthetic Pharmacy Order Data Generator
Team: MatriXplorers | University of Moratuwa
Generates 12 months of realistic weekly pharmacy order data for Sri Lanka
with seasonal disease patterns, school terms, and regional variation.
"""

import pandas as pd
import numpy as np
import os

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)

# ── Date Range ─────────────────────────────────────────────────────────────────
START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pharmacy_orders.csv")
HEALTH_FILE = os.path.join(OUTPUT_DIR, "health_signals.csv")

# ── Sri Lanka Provinces ────────────────────────────────────────────────────────
# pharmacies: approx count in each province
# urban_factor: scales demand (Western province is highest)
REGIONS = {
    "Western":       {"pharmacies": 650, "population": 5_800_000, "urban_factor": 1.40},
    "Central":       {"pharmacies": 280, "population": 2_600_000, "urban_factor": 0.90},
    "Southern":      {"pharmacies": 260, "population": 2_500_000, "urban_factor": 0.85},
    "Northern":      {"pharmacies": 180, "population": 1_100_000, "urban_factor": 0.75},
    "Eastern":       {"pharmacies": 200, "population": 1_700_000, "urban_factor": 0.70},
    "North Western": {"pharmacies": 290, "population": 2_500_000, "urban_factor": 0.95},
    "North Central": {"pharmacies": 160, "population": 1_200_000, "urban_factor": 0.70},
    "Uva":           {"pharmacies": 130, "population": 1_300_000, "urban_factor": 0.65},
    "Sabaragamuwa":  {"pharmacies": 150, "population": 1_900_000, "urban_factor": 0.75},
}

# ── Medicine SKUs ──────────────────────────────────────────────────────────────
# base_demand = weekly units per average region (before scaling)
# unit_price  = USD equivalent (will convert to LKR)
SKUS = [
    # Antipyretics
    {"sku_id": "SKU001", "name": "Paracetamol 500mg",          "category": "Antipyretic",      "base_demand": 500, "unit_price": 1.5},
    {"sku_id": "SKU002", "name": "Ibuprofen 400mg",            "category": "Antipyretic",      "base_demand": 300, "unit_price": 2.0},
    # Antibiotics
    {"sku_id": "SKU003", "name": "Amoxicillin 500mg",          "category": "Antibiotic",       "base_demand": 200, "unit_price": 8.5},
    {"sku_id": "SKU004", "name": "Azithromycin 250mg",         "category": "Antibiotic",       "base_demand": 150, "unit_price": 12.0},
    # Dengue / Fever
    {"sku_id": "SKU005", "name": "Dengue Rapid Test Kit",      "category": "Diagnostics",      "base_demand": 100, "unit_price": 450.0},
    {"sku_id": "SKU006", "name": "ORS Sachets",                "category": "Rehydration",      "base_demand": 400, "unit_price": 3.5},
    # Paediatric
    {"sku_id": "SKU007", "name": "Paediatric Paracetamol Syrup","category": "Paediatric",      "base_demand": 250, "unit_price": 5.0},
    {"sku_id": "SKU008", "name": "Children's Multivitamin",    "category": "Paediatric",       "base_demand": 180, "unit_price": 18.0},
    # Respiratory / Allergy
    {"sku_id": "SKU009", "name": "Salbutamol Inhaler",         "category": "Respiratory",      "base_demand": 120, "unit_price": 35.0},
    {"sku_id": "SKU010", "name": "Cetirizine 10mg",            "category": "Antihistamine",    "base_demand": 220, "unit_price": 4.0},
    # Chronic Disease (stable demand)
    {"sku_id": "SKU011", "name": "Metformin 500mg",            "category": "Diabetes",         "base_demand": 350, "unit_price": 2.5},
    {"sku_id": "SKU012", "name": "Amlodipine 5mg",             "category": "Cardiovascular",   "base_demand": 300, "unit_price": 3.0},
    # Supplements
    {"sku_id": "SKU013", "name": "Vitamin C 500mg",            "category": "Supplement",       "base_demand": 280, "unit_price": 6.0},
    {"sku_id": "SKU014", "name": "Zinc Supplement",            "category": "Supplement",       "base_demand": 150, "unit_price": 8.0},
    # GI
    {"sku_id": "SKU015", "name": "Omeprazole 20mg",            "category": "Gastrointestinal", "base_demand": 200, "unit_price": 5.5},
]

# ── How much each SKU reacts to seasonal signals (0 = no effect, 1.5 = strong) ─
SKU_SEASONALITY = {
    "SKU001": {"dengue": 0.70, "flu": 0.60, "school": 0.30},
    "SKU002": {"dengue": 0.50, "flu": 0.50, "school": 0.20},
    "SKU003": {"dengue": 0.30, "flu": 0.70, "school": 0.40},
    "SKU004": {"dengue": 0.20, "flu": 0.80, "school": 0.30},
    "SKU005": {"dengue": 1.50, "flu": 0.00, "school": 0.00},  # Dengue kit — very sensitive
    "SKU006": {"dengue": 1.00, "flu": 0.40, "school": 0.20},
    "SKU007": {"dengue": 0.60, "flu": 0.50, "school": 0.80},  # Paediatric spikes during school
    "SKU008": {"dengue": 0.00, "flu": 0.00, "school": 0.90},
    "SKU009": {"dengue": 0.00, "flu": 0.30, "school": 0.20},
    "SKU010": {"dengue": 0.30, "flu": 0.40, "school": 0.20},
    "SKU011": {"dengue": 0.00, "flu": 0.00, "school": 0.00},  # Chronic — flat
    "SKU012": {"dengue": 0.00, "flu": 0.00, "school": 0.00},
    "SKU013": {"dengue": 0.20, "flu": 0.30, "school": 0.30},
    "SKU014": {"dengue": 0.30, "flu": 0.20, "school": 0.20},
    "SKU015": {"dengue": 0.10, "flu": 0.10, "school": 0.10},
}


# ── Seasonal Signal Functions ──────────────────────────────────────────────────

def dengue_index(week: int) -> float:
    """
    Dengue peaks twice in Sri Lanka:
      Peak 1 → May–June   (weeks 18–26) after SW monsoon
      Peak 2 → Nov–Dec    (weeks 44–52) after NE monsoon
    Returns a multiplier ≥ 1.0
    """
    if 18 <= week <= 26:
        return 1.0 + 0.85 * np.sin(np.pi * (week - 18) / 8)
    if 44 <= week <= 52:
        return 1.0 + 0.65 * np.sin(np.pi * (week - 44) / 8)
    return 1.0


def flu_index(week: int) -> float:
    """
    Flu peaks:
      Peak 1 → Jan–Feb   (weeks 1–8)
      Peak 2 → Jul–Aug   (weeks 27–35)
    """
    if 1 <= week <= 8:
        return 1.0 + 0.50 * np.sin(np.pi * week / 8)
    if 27 <= week <= 35:
        return 1.0 + 0.40 * np.sin(np.pi * (week - 27) / 8)
    return 1.0


def school_term_multiplier(week: int) -> float:
    """
    Sri Lanka school terms (approx):
      Term 1 → Jan–Apr   (weeks 1–17)
      Term 2 → May–Aug   (weeks 19–35)
      Term 3 → Sep–Dec   (weeks 37–52)
    Holidays slightly reduce paediatric demand.
    """
    school_weeks = set(range(1, 17)) | set(range(19, 35)) | set(range(37, 52))
    return 1.15 if week in school_weeks else 0.85


# ── Main Generator ─────────────────────────────────────────────────────────────

def generate_pharmacy_orders() -> pd.DataFrame:
    """Generate weekly pharmacy order data for all regions and SKUs."""
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
    records = []

    for date in dates:
        week = int(date.isocalendar()[1])
        d_idx = dengue_index(week)
        f_idx = flu_index(week)
        s_mult = school_term_multiplier(week)

        for region_name, region_info in REGIONS.items():
            # Scale to region size
            region_scale = (region_info["pharmacies"] / 280) * region_info["urban_factor"]

            for sku in SKUS:
                sid = sku["sku_id"]
                sens = SKU_SEASONALITY[sid]

                # Seasonal adjustment on top of base demand
                seasonal_effect = (
                    1.0
                    + sens["dengue"] * (d_idx - 1.0)
                    + sens["flu"]   * (f_idx - 1.0)
                    + sens["school"]* (s_mult - 1.0)
                )

                # ±12% random noise per week/region
                noise = np.random.normal(1.0, 0.12)

                units = max(1, int(sku["base_demand"] * region_scale * seasonal_effect * noise))
                price_lkr = sku["unit_price"] * 310  # approx LKR conversion

                records.append({
                    "date":               date.strftime("%Y-%m-%d"),
                    "week":               week,
                    "year":               date.year,
                    "region":             region_name,
                    "sku_id":             sid,
                    "medicine_name":      sku["name"],
                    "category":           sku["category"],
                    "units_ordered":      units,
                    "unit_price_lkr":     round(price_lkr, 2),
                    "total_value_lkr":    round(units * price_lkr, 2),
                    "dengue_index":       round(d_idx, 3),
                    "flu_index":          round(f_idx, 3),
                    "school_term_active": s_mult > 1.0,
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_health_signals() -> pd.DataFrame:
    """
    Simulates weekly public health bulletin data
    (mirrors Sri Lanka Ministry of Health surveillance style).
    """
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
    records = []

    for date in dates:
        week = int(date.isocalendar()[1])
        records.append({
            "date":                date.strftime("%Y-%m-%d"),
            "week":                week,
            "dengue_cases_national": int(np.random.normal(300 * dengue_index(week), 40)),
            "flu_cases_national":    int(np.random.normal(500 * flu_index(week), 60)),
            "dengue_alert_level":    "HIGH" if dengue_index(week) > 1.4 else
                                     "MEDIUM" if dengue_index(week) > 1.1 else "LOW",
            "flu_alert_level":       "HIGH" if flu_index(week) > 1.3 else
                                     "MEDIUM" if flu_index(week) > 1.1 else "LOW",
            "school_term_active":    school_term_multiplier(week) > 1.0,
        })

    return pd.DataFrame(records)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("⏳  Generating pharmacy order data...")
    orders = generate_pharmacy_orders()
    orders.to_csv(OUTPUT_FILE, index=False)

    print("⏳  Generating public health signals...")
    signals = generate_health_signals()
    signals.to_csv(HEALTH_FILE, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("✅  PharmaForesight — Data Generation Complete")
    print("="*55)
    print(f"  Orders file  : {OUTPUT_FILE}")
    print(f"  Signals file : {HEALTH_FILE}")
    print(f"  Total records: {len(orders):,}")
    print(f"  Date range   : {orders['date'].min().date()} → {orders['date'].max().date()}")
    print(f"  Regions      : {orders['region'].nunique()}")
    print(f"  SKUs         : {orders['sku_id'].nunique()}")
    print(f"  Total value  : LKR {orders['total_value_lkr'].sum():,.0f}")
    print("="*55)

    print("\n📊  Top 5 SKUs by total volume:")
    top = (orders.groupby("medicine_name")["units_ordered"]
                 .sum()
                 .sort_values(ascending=False)
                 .head(5))
    for name, vol in top.items():
        print(f"    {name:<38} {vol:>10,} units")

    print("\n🗺️   Regional totals:")
    reg = (orders.groupby("region")["units_ordered"]
                 .sum()
                 .sort_values(ascending=False))
    for region, vol in reg.items():
        print(f"    {region:<20} {vol:>10,} units")

    print("\n✅  Ready for the forecasting model!")
