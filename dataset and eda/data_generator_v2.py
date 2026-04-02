"""
PharmaForesight — Enhanced Synthetic Pharmacy Order Data Generator v2
Team: MatriXplorers | University of Moratuwa

New in v2:
  + stock_level_units          — warehouse stock at time of order
  + lead_time_days             — regional delivery lead time (weather-adjusted)
  + stockout_flag              — 1 if stock ran out that week
  + rainfall_mm                — weekly rainfall (Sri Lanka monsoon patterns)
  + weather_condition          — Drought / Sunny / Cloudy / Rainy / Heavy Rain
  + expiry_risk_flag           — 1 if overstock risks expiry (short shelf-life SKUs)
  + population_density_per_km2 — people per km² by province
  + pharmacy_count_region      — pharmacies in that region
  + generic_vs_brand           — Generic / Brand
  + shelf_life_weeks           — product shelf life
  - total_value_lkr            — removed (redundant: units × price)
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

START_DATE  = "2024-01-01"
END_DATE    = "2024-12-31"
OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pharmacy_orders.csv")
HEALTH_FILE = os.path.join(OUTPUT_DIR, "health_signals.csv")

# ── Sri Lanka Provinces ────────────────────────────────────────────────────────
REGIONS = {
    "Western":       {"pharmacies": 650, "population": 5_800_000, "urban_factor": 1.40, "area_km2": 3_593,  "base_lead_days": 1},
    "Central":       {"pharmacies": 280, "population": 2_600_000, "urban_factor": 0.90, "area_km2": 5_575,  "base_lead_days": 3},
    "Southern":      {"pharmacies": 260, "population": 2_500_000, "urban_factor": 0.85, "area_km2": 5_559,  "base_lead_days": 2},
    "Northern":      {"pharmacies": 180, "population": 1_100_000, "urban_factor": 0.75, "area_km2": 8_884,  "base_lead_days": 5},
    "Eastern":       {"pharmacies": 200, "population": 1_700_000, "urban_factor": 0.70, "area_km2": 9_996,  "base_lead_days": 4},
    "North Western": {"pharmacies": 290, "population": 2_500_000, "urban_factor": 0.95, "area_km2": 7_888,  "base_lead_days": 2},
    "North Central": {"pharmacies": 160, "population": 1_200_000, "urban_factor": 0.70, "area_km2": 10_472, "base_lead_days": 4},
    "Uva":           {"pharmacies": 130, "population": 1_300_000, "urban_factor": 0.65, "area_km2": 8_500,  "base_lead_days": 4},
    "Sabaragamuwa":  {"pharmacies": 150, "population": 1_900_000, "urban_factor": 0.75, "area_km2": 4_968,  "base_lead_days": 3},
}

# ── Medicine SKUs ──────────────────────────────────────────────────────────────
SKUS = [
    {"sku_id": "SKU001", "name": "Paracetamol 500mg",            "category": "Antipyretic",     "base_demand": 500, "unit_price": 1.5,   "shelf_life_weeks": 260, "is_generic": True},
    {"sku_id": "SKU002", "name": "Ibuprofen 400mg",              "category": "Antipyretic",     "base_demand": 300, "unit_price": 2.0,   "shelf_life_weeks": 260, "is_generic": True},
    {"sku_id": "SKU003", "name": "Amoxicillin 500mg",            "category": "Antibiotic",      "base_demand": 200, "unit_price": 8.5,   "shelf_life_weeks": 130, "is_generic": True},
    {"sku_id": "SKU004", "name": "Azithromycin 250mg",           "category": "Antibiotic",      "base_demand": 150, "unit_price": 12.0,  "shelf_life_weeks": 130, "is_generic": False},
    {"sku_id": "SKU005", "name": "Dengue Rapid Test Kit",        "category": "Diagnostics",     "base_demand": 100, "unit_price": 450.0, "shelf_life_weeks": 78,  "is_generic": False},
    {"sku_id": "SKU006", "name": "ORS Sachets",                  "category": "Rehydration",     "base_demand": 400, "unit_price": 3.5,   "shelf_life_weeks": 104, "is_generic": True},
    {"sku_id": "SKU007", "name": "Paediatric Paracetamol Syrup", "category": "Paediatric",      "base_demand": 250, "unit_price": 5.0,   "shelf_life_weeks": 52,  "is_generic": True},
    {"sku_id": "SKU008", "name": "Children's Multivitamin",      "category": "Paediatric",      "base_demand": 180, "unit_price": 18.0,  "shelf_life_weeks": 104, "is_generic": False},
    {"sku_id": "SKU009", "name": "Salbutamol Inhaler",           "category": "Respiratory",     "base_demand": 120, "unit_price": 35.0,  "shelf_life_weeks": 130, "is_generic": False},
    {"sku_id": "SKU010", "name": "Cetirizine 10mg",              "category": "Antihistamine",   "base_demand": 220, "unit_price": 4.0,   "shelf_life_weeks": 260, "is_generic": True},
    {"sku_id": "SKU011", "name": "Metformin 500mg",              "category": "Diabetes",        "base_demand": 350, "unit_price": 2.5,   "shelf_life_weeks": 260, "is_generic": True},
    {"sku_id": "SKU012", "name": "Amlodipine 5mg",               "category": "Cardiovascular",  "base_demand": 300, "unit_price": 3.0,   "shelf_life_weeks": 260, "is_generic": True},
    {"sku_id": "SKU013", "name": "Vitamin C 500mg",              "category": "Supplement",      "base_demand": 280, "unit_price": 6.0,   "shelf_life_weeks": 130, "is_generic": True},
    {"sku_id": "SKU014", "name": "Zinc Supplement",              "category": "Supplement",      "base_demand": 150, "unit_price": 8.0,   "shelf_life_weeks": 130, "is_generic": True},
    {"sku_id": "SKU015", "name": "Omeprazole 20mg",              "category": "Gastrointestinal","base_demand": 200, "unit_price": 5.5,   "shelf_life_weeks": 130, "is_generic": True},
]

# ── Demand Sensitivities ───────────────────────────────────────────────────────
SKU_SEASONALITY = {
    "SKU001": {"dengue": 0.70, "flu": 0.60, "school": 0.30, "rain": 0.10},
    "SKU002": {"dengue": 0.50, "flu": 0.50, "school": 0.20, "rain": 0.05},
    "SKU003": {"dengue": 0.30, "flu": 0.70, "school": 0.40, "rain": 0.10},
    "SKU004": {"dengue": 0.20, "flu": 0.80, "school": 0.30, "rain": 0.05},
    "SKU005": {"dengue": 1.50, "flu": 0.00, "school": 0.00, "rain": 0.20},
    "SKU006": {"dengue": 1.00, "flu": 0.40, "school": 0.20, "rain": 0.15},
    "SKU007": {"dengue": 0.60, "flu": 0.50, "school": 0.80, "rain": 0.10},
    "SKU008": {"dengue": 0.00, "flu": 0.00, "school": 0.90, "rain": 0.00},
    "SKU009": {"dengue": 0.00, "flu": 0.30, "school": 0.20, "rain": 0.25},
    "SKU010": {"dengue": 0.30, "flu": 0.40, "school": 0.20, "rain": 0.15},
    "SKU011": {"dengue": 0.00, "flu": 0.00, "school": 0.00, "rain": 0.00},
    "SKU012": {"dengue": 0.00, "flu": 0.00, "school": 0.00, "rain": 0.00},
    "SKU013": {"dengue": 0.20, "flu": 0.30, "school": 0.30, "rain": 0.05},
    "SKU014": {"dengue": 0.30, "flu": 0.20, "school": 0.20, "rain": 0.05},
    "SKU015": {"dengue": 0.10, "flu": 0.10, "school": 0.10, "rain": 0.05},
}

# ── Rainfall Model ─────────────────────────────────────────────────────────────
# SW Monsoon → May–Sep (weeks 18–39): wetter in West, South, Sabaragamuwa
# NE Monsoon → Nov–Jan (weeks 44–4):  wetter in North, East, North Central
REGION_RAIN_PROFILE = {
    "Western":       {"sw": 1.8, "ne": 0.6, "base": 90},
    "Central":       {"sw": 1.6, "ne": 0.7, "base": 95},
    "Southern":      {"sw": 1.7, "ne": 0.5, "base": 85},
    "Northern":      {"sw": 0.4, "ne": 1.8, "base": 55},
    "Eastern":       {"sw": 0.5, "ne": 1.9, "base": 65},
    "North Western": {"sw": 1.4, "ne": 0.8, "base": 75},
    "North Central": {"sw": 0.7, "ne": 1.5, "base": 60},
    "Uva":           {"sw": 0.9, "ne": 1.2, "base": 70},
    "Sabaragamuwa":  {"sw": 1.5, "ne": 0.6, "base": 88},
}

SAFETY_STOCK_WEEKS = {
    "Western": 1.5, "North Western": 2.0, "Southern": 2.0,
    "Central": 2.5, "Northern": 3.5,      "Eastern": 3.0,
    "North Central": 3.0, "Uva": 3.5,    "Sabaragamuwa": 2.5,
}


# ── Helper Functions ───────────────────────────────────────────────────────────

def dengue_index(week):
    if 18 <= week <= 26: return 1.0 + 0.85 * np.sin(np.pi * (week - 18) / 8)
    if 44 <= week <= 52: return 1.0 + 0.65 * np.sin(np.pi * (week - 44) / 8)
    return 1.0

def flu_index(week):
    if 1 <= week <= 8:   return 1.0 + 0.50 * np.sin(np.pi * week / 8)
    if 27 <= week <= 35: return 1.0 + 0.40 * np.sin(np.pi * (week - 27) / 8)
    return 1.0

def school_term_multiplier(week):
    school_weeks = set(range(1, 17)) | set(range(19, 35)) | set(range(37, 52))
    return 1.15 if week in school_weeks else 0.85

def get_rainfall(week, region):
    p  = REGION_RAIN_PROFILE[region]
    sw = p["sw"] * max(0, np.sin(np.pi * (week - 18) / 21)) if 18 <= week <= 39 else 0
    ne_w = week - 44 if week >= 44 else week + 8
    ne   = p["ne"] * max(0, np.sin(np.pi * ne_w / 14)) if (week >= 44 or week <= 4) else 0
    inter = 0.4 if week in range(12, 17) or week in range(40, 44) else 0
    raw   = p["base"] * (1 + sw + ne + inter)
    return round(max(0, raw * np.random.normal(1.0, 0.18)), 1)

def get_weather(rain):
    if rain < 20:  return "Drought"
    if rain < 50:  return "Sunny"
    if rain < 90:  return "Cloudy"
    if rain < 150: return "Rainy"
    return "Heavy Rain"

def simulate_stock(units_ordered, region):
    safety  = SAFETY_STOCK_WEEKS.get(region, 2.0)
    avg     = units_ordered * safety
    stock   = max(0, int(np.random.normal(avg, avg * 0.25)))
    stockout = int(stock < units_ordered * 0.8)
    return stock, stockout

def get_lead_time(region, rain):
    base  = REGIONS[region]["base_lead_days"]
    delay = 0
    if rain > 150: delay = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
    elif rain > 90: delay = np.random.choice([0, 1], p=[0.6, 0.4])
    return int(base + delay)

def get_expiry_risk(sku, stock_units, units_ordered):
    shelf = sku["shelf_life_weeks"]
    ratio = stock_units / max(1, units_ordered)
    if shelf < 60  and ratio > 2.5: return 1
    if shelf < 130 and ratio > 3.5: return 1
    return 0


# ── Main Generator ─────────────────────────────────────────────────────────────

def generate_pharmacy_orders():
    dates   = pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
    records = []

    for date in dates:
        week   = int(date.isocalendar()[1])
        d_idx  = dengue_index(week)
        f_idx  = flu_index(week)
        s_mult = school_term_multiplier(week)

        for region_name, region_info in REGIONS.items():
            pop_density  = round(region_info["population"] / region_info["area_km2"], 1)
            rain         = get_rainfall(week, region_name)
            weather      = get_weather(rain)
            region_scale = (region_info["pharmacies"] / 280) * region_info["urban_factor"]

            for sku in SKUS:
                sid  = sku["sku_id"]
                sens = SKU_SEASONALITY[sid]

                seasonal_effect = (
                    1.0
                    + sens["dengue"] * (d_idx - 1.0)
                    + sens["flu"]    * (f_idx - 1.0)
                    + sens["school"] * (s_mult - 1.0)
                    + sens["rain"]   * (min(rain, 200) / 200)
                )
                units = max(1, int(sku["base_demand"] * region_scale * seasonal_effect
                                   * np.random.normal(1.0, 0.12)))

                stock, stockout = simulate_stock(units, region_name)
                lead            = get_lead_time(region_name, rain)
                expiry          = get_expiry_risk(sku, stock, units)

                records.append({
                    # Identifiers
                    "date":                       date.strftime("%Y-%m-%d"),
                    "week":                       week,
                    "year":                       date.year,
                    "region":                     region_name,
                    "sku_id":                     sid,
                    "medicine_name":              sku["name"],
                    "category":                   sku["category"],
                    # Demand
                    "units_ordered":              units,
                    "unit_price_lkr":             round(sku["unit_price"] * 310, 2),
                    # Supply chain
                    "stock_level_units":          stock,
                    "lead_time_days":             lead,
                    "stockout_flag":              stockout,
                    "expiry_risk_flag":           expiry,
                    # Product metadata
                    "generic_vs_brand":           "Generic" if sku["is_generic"] else "Brand",
                    "shelf_life_weeks":           sku["shelf_life_weeks"],
                    # Regional context
                    "pharmacy_count_region":      region_info["pharmacies"],
                    "population_density_per_km2": pop_density,
                    # Weather / environment
                    "rainfall_mm":                rain,
                    "weather_condition":          weather,
                    # Epidemiological signals
                    "dengue_index":               round(d_idx, 3),
                    "flu_index":                  round(f_idx, 3),
                    "school_term_active":         s_mult > 1.0,
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_health_signals():
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
    records = []
    for date in dates:
        week = int(date.isocalendar()[1])
        records.append({
            "date":                  date.strftime("%Y-%m-%d"),
            "week":                  week,
            "dengue_cases_national": int(np.random.normal(300 * dengue_index(week), 40)),
            "flu_cases_national":    int(np.random.normal(500 * flu_index(week), 60)),
            "dengue_alert_level":    "HIGH"   if dengue_index(week) > 1.4 else
                                     "MEDIUM" if dengue_index(week) > 1.1 else "LOW",
            "flu_alert_level":       "HIGH"   if flu_index(week) > 1.3 else
                                     "MEDIUM" if flu_index(week) > 1.1 else "LOW",
            "school_term_active":    school_term_multiplier(week) > 1.0,
        })
    return pd.DataFrame(records)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("⏳  Generating enhanced pharmacy order data (v2)...")
    orders = generate_pharmacy_orders()
    orders.to_csv(OUTPUT_FILE, index=False)

    print("⏳  Generating public health signals...")
    signals = generate_health_signals()
    signals.to_csv(HEALTH_FILE, index=False)

    print("\n" + "="*60)
    print("✅  PharmaForesight v2 — Data Generation Complete")
    print("="*60)
    print(f"  Records   : {len(orders):,}")
    print(f"  Columns   : {len(orders.columns)}  →  {list(orders.columns)}")
    print(f"  Date range: {orders['date'].min().date()} → {orders['date'].max().date()}")
    print("="*60)

    print("\n📦  Supply Chain Stats:")
    print(f"  Stockout flags    : {orders['stockout_flag'].sum():,}  ({orders['stockout_flag'].mean()*100:.1f}%)")
    print(f"  Expiry risk flags : {orders['expiry_risk_flag'].sum():,}  ({orders['expiry_risk_flag'].mean()*100:.1f}%)")

    print("\n🌧️  Weather distribution (all regions):")
    for cond, cnt in orders["weather_condition"].value_counts().items():
        print(f"    {cond:<12} {cnt:>6,}  ({cnt/len(orders)*100:.1f}%)")

    print("\n💊  Generic vs Brand:")
    for g, cnt in orders["generic_vs_brand"].value_counts().items():
        print(f"    {g:<10} {cnt:>6,}  ({cnt/len(orders)*100:.1f}%)")

    print("\n🏭  Stockout rate by region:")
    for region, rate in (orders.groupby("region")["stockout_flag"].mean()*100).sort_values(ascending=False).items():
        print(f"    {region:<20} {rate:4.1f}%  {'█' * int(rate/2)}")

    print("\n⏱️  Avg lead time by region:")
    for region, lt in orders.groupby("region")["lead_time_days"].mean().sort_values(ascending=False).items():
        print(f"    {region:<20} {lt:.1f} days")

    print("\n✅  Ready for the forecasting model!")
