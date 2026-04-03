"""
PharmaForesight — REAL-WORLD CALIBRATED Synthetic Data Generator v4
Team: MatriXplorers | University of Moratuwa

============================================================================
V4 IMPROVEMENTS OVER V2 (Based on REAL DATA Analysis):
============================================================================

📊 DISEASE SIGNAL CALIBRATION (from Sri Lanka Dengue/Flu data):
  V2: Dengue peak June-Sept (single peak), Flu peak Jan-Feb
  V4: Dengue TWO PEAKS: Weeks 1-7 (Jan-Feb) + Weeks 25-32 (Jun-Aug) ← REAL DATA!
      Flu PRIMARY peak: Weeks 17-21 (May) ← REAL WHO DATA SHOWED THIS!
      Regional dengue weights: Colombo 2.5x, Jaffna 1.5x, Northern areas lower

📈 DEMAND VARIABILITY (from Rossmann Pharmacy + Medicine TS data):
  V2: CV = 12% (too smooth, unrealistic)
  V4: CV = 45% (matches real pharmacy sales variability)
      December multiplier: +25% (real holiday effect)
      Category-specific seasonality (Paracetamol peaks Oct-Jan, Respiratory Dec-Jan)

📦 SUPPLY CHAIN (from USAID SCMS + Supplier Disruption data):
  V2: Stockout ~0% (unrealistic)
  V4: Stockout 1-3% baseline, 4-5% during monsoon weeks
      Late delivery rate: 12% (matches real data)
      Lead time CV: 25% (realistic variability)

🗺️ REGIONAL CALIBRATION (from Sri Lanka HDX data):
  V2: Uniform disease impact across regions
  V4: District-weighted dengue impact based on real case distribution
      Colombo/Western = 2.5x, Jaffna = 1.5x, Rural areas = 0.7-0.8x

🕐 TIME RANGE:
  V2: 1 year (2024 only)
  V4: 3 years (2023-2025) with year-over-year trends

============================================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
START_DATE = "2023-01-02"  # First Monday of 2023
END_DATE = "2025-12-29"    # Last Monday of 2025
OUTPUT_DIR = "v4-dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pharmacy_orders_v4.csv")
HEALTH_FILE = os.path.join(OUTPUT_DIR, "health_signals_v4.csv")

# ============================================================================
# REAL-WORLD CALIBRATED PARAMETERS (from downloaded datasets)
# ============================================================================

# 🦟 DENGUE SEASONALITY (from 02_DISEASE_001_SL_Dengue_2024.csv)
# REAL DATA: Two peaks - Jan-Feb (10,299 cases) and Jun-Aug (4,315 cases)
DENGUE_PRIMARY_PEAK_WEEKS = list(range(1, 8))      # Weeks 1-7 (Jan-Feb) - BIGGEST PEAK!
DENGUE_SECONDARY_PEAK_WEEKS = list(range(25, 33)) # Weeks 25-32 (Jun-Aug)
DENGUE_TERTIARY_PEAK_WEEKS = list(range(48, 53))  # Weeks 48-52 (Dec) - Year-end surge
DENGUE_LOW_WEEKS = list(range(13, 21))            # Weeks 13-20 (Apr-May) - LOWEST

# 🦠 FLU SEASONALITY (from 02_DISEASE_005_WHO_FluNet.csv - Sri Lanka filtered)
# REAL DATA: Primary peak Week 17-21 (May), secondary peaks Week 1-4 and 52
FLU_PRIMARY_PEAK_WEEKS = list(range(17, 22))      # Weeks 17-21 (May) - MAIN PEAK!
FLU_SECONDARY_PEAK_WEEKS = list(range(1, 5)) + [52] # Jan + year-end
FLU_LOW_WEEKS = list(range(32, 41))               # Weeks 32-40 (Aug-Oct) - LOWEST

# 🗺️ REGIONAL DENGUE WEIGHTS (from district case distribution)
# Based on: Colombo 22%, Jaffna 12%, Gampaha 10%, Kandy 9%
REGIONAL_DENGUE_WEIGHT = {
    "Western": 2.5,       # Colombo + Gampaha combined (highest burden)
    "Northern": 1.5,      # Jaffna has high dengue
    "Central": 1.2,       # Kandy moderate
    "North Western": 1.1, # Kurunegala moderate
    "Sabaragamuwa": 1.0,  # Baseline
    "Southern": 0.9,      # Lower than average
    "Eastern": 0.8,       # Lower
    "North Central": 0.7, # Rural, lower
    "Uva": 0.7,           # Rural, lower
}

# 📊 DEMAND VARIABILITY (from Rossmann: CV=44.6%)
DEMAND_CV_TARGET = 0.45  # Real pharmacy sales variability
DECEMBER_MULTIPLIER = 1.25  # Real holiday effect (+23%)
PROMO_PROBABILITY = 0.08  # 8% of weeks have promotional spikes
PROMO_LIFT = 0.35  # 35% lift during promos

# 📦 SUPPLY CHAIN (from USAID SCMS + Supplier Disruption data)
STOCKOUT_BASE_PROB = 0.015      # 1.5% baseline
STOCKOUT_MONSOON_PROB = 0.04    # 4% during monsoon
LATE_DELIVERY_PROB = 0.12       # 12% late deliveries
LEAD_TIME_CV = 0.25             # 25% variability

# ── Sri Lanka Provinces ────────────────────────────────────────────────────────
REGIONS = {
    "Western":       {"pharmacies": 650, "population": 5_800_000, "urban_factor": 1.40, "area_km2": 3_593,  "base_lead_days": 1, "monsoon_exposure": "sw"},
    "Central":       {"pharmacies": 280, "population": 2_600_000, "urban_factor": 0.90, "area_km2": 5_575,  "base_lead_days": 3, "monsoon_exposure": "sw"},
    "Southern":      {"pharmacies": 260, "population": 2_500_000, "urban_factor": 0.85, "area_km2": 5_559,  "base_lead_days": 2, "monsoon_exposure": "sw"},
    "Northern":      {"pharmacies": 180, "population": 1_100_000, "urban_factor": 0.75, "area_km2": 8_884,  "base_lead_days": 5, "monsoon_exposure": "ne"},
    "Eastern":       {"pharmacies": 200, "population": 1_700_000, "urban_factor": 0.70, "area_km2": 9_996,  "base_lead_days": 4, "monsoon_exposure": "ne"},
    "North Western": {"pharmacies": 290, "population": 2_500_000, "urban_factor": 0.95, "area_km2": 7_888,  "base_lead_days": 2, "monsoon_exposure": "sw"},
    "North Central": {"pharmacies": 160, "population": 1_200_000, "urban_factor": 0.70, "area_km2": 10_472, "base_lead_days": 4, "monsoon_exposure": "ne"},
    "Uva":           {"pharmacies": 130, "population": 1_300_000, "urban_factor": 0.65, "area_km2": 8_500,  "base_lead_days": 4, "monsoon_exposure": "both"},
    "Sabaragamuwa":  {"pharmacies": 150, "population": 1_900_000, "urban_factor": 0.75, "area_km2": 4_968,  "base_lead_days": 3, "monsoon_exposure": "sw"},
}

# ── Medicine SKUs (same as V2) ─────────────────────────────────────────────────
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

# ── V4: REAL-WORLD CALIBRATED Disease Sensitivities ────────────────────────────
# Based on: Medicine Demand Time Series analysis (Paracetamol peaks Oct-Jan, etc.)
SKU_DISEASE_SENSITIVITY = {
    # Antipyretics: Peak Oct-Jan (real data showed 42.0 in Oct vs 19.5 in Jul = 2.15x)
    "SKU001": {"dengue": 1.80, "flu": 1.50, "school": 0.30, "rain": 0.10, "seasonal_peak_month": [10, 11, 12, 1]},
    "SKU002": {"dengue": 1.50, "flu": 1.30, "school": 0.25, "rain": 0.08, "seasonal_peak_month": [10, 11, 12, 1]},
    
    # Antibiotics: Flu-driven (real data showed strong secondary infection correlation)
    "SKU003": {"dengue": 0.40, "flu": 3.00, "school": 0.40, "rain": 0.10, "seasonal_peak_month": [1, 2, 5]},
    "SKU004": {"dengue": 0.30, "flu": 3.50, "school": 0.35, "rain": 0.08, "seasonal_peak_month": [1, 2, 5]},
    
    # Diagnostics: PURE dengue signal (test kits spike during outbreaks)
    "SKU005": {"dengue": 4.00, "flu": 0.00, "school": 0.00, "rain": 0.15, "seasonal_peak_month": [1, 2, 6, 7, 8]},
    
    # Rehydration: Strong dengue + flu (dehydration from fever)
    "SKU006": {"dengue": 2.50, "flu": 1.20, "school": 0.20, "rain": 0.12, "seasonal_peak_month": [1, 2, 6, 7, 8]},
    
    # Paediatric: School term + disease season
    "SKU007": {"dengue": 1.00, "flu": 0.90, "school": 0.90, "rain": 0.10, "seasonal_peak_month": [1, 2, 6, 9, 10]},
    "SKU008": {"dengue": 0.00, "flu": 0.00, "school": 1.20, "rain": 0.00, "seasonal_peak_month": [1, 5, 9]},
    
    # Respiratory: Peak Dec-Jan (real data: 7.9 Dec vs 3.0 Jul = 2.6x swing!)
    "SKU009": {"dengue": 0.00, "flu": 2.50, "school": 0.25, "rain": 0.25, "seasonal_peak_month": [12, 1, 2, 10]},
    
    # Antihistamine: Peak Apr-May (allergy season), trough Dec
    "SKU010": {"dengue": 0.50, "flu": 1.50, "school": 0.25, "rain": 0.20, "seasonal_peak_month": [4, 5, 3]},
    
    # Chronic medications: Stable baseline (no disease sensitivity)
    "SKU011": {"dengue": 0.00, "flu": 0.00, "school": 0.00, "rain": 0.00, "seasonal_peak_month": []},
    "SKU012": {"dengue": 0.00, "flu": 0.00, "school": 0.00, "rain": 0.00, "seasonal_peak_month": []},
    
    # Supplements: Moderate disease correlation (immunity boosting)
    "SKU013": {"dengue": 0.50, "flu": 0.70, "school": 0.35, "rain": 0.05, "seasonal_peak_month": [1, 2, 10, 11, 12]},
    "SKU014": {"dengue": 0.60, "flu": 0.50, "school": 0.25, "rain": 0.05, "seasonal_peak_month": [1, 2, 6, 7]},
    
    # GI: Stable with mild seasonal variation
    "SKU015": {"dengue": 0.15, "flu": 0.15, "school": 0.10, "rain": 0.05, "seasonal_peak_month": []},
}

# ── Rainfall Model (same as V2, validated against real patterns) ───────────────
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
    "Central": 2.5, "Northern": 3.5, "Eastern": 3.0,
    "North Central": 3.0, "Uva": 3.5, "Sabaragamuwa": 2.5,
}

# ============================================================================
# V4 HELPER FUNCTIONS (Real-World Calibrated)
# ============================================================================

def dengue_index_v4(week, region):
    """
    V4: REAL-WORLD CALIBRATED dengue index
    Based on: 02_DISEASE_001_SL_Dengue_2024.csv analysis
    
    Key insight: TWO PEAKS per year!
    - Primary: Weeks 1-7 (Jan-Feb) - 10,299 cases in Jan!
    - Secondary: Weeks 25-32 (Jun-Aug) - 4,315 cases
    - Tertiary: Weeks 48-52 (Dec) - 7,191 cases
    """
    regional_weight = REGIONAL_DENGUE_WEIGHT.get(region, 1.0)
    
    # Primary peak: Jan-Feb (BIGGEST - based on real data!)
    if week in DENGUE_PRIMARY_PEAK_WEEKS:
        # Peak at week 3 (real data: 2,913 cases in week 3)
        amplitude = 1.5 * np.sin(np.pi * (week - 0.5) / 7)
        base = 1.0 + amplitude
    
    # Secondary peak: Jun-Aug
    elif week in DENGUE_SECONDARY_PEAK_WEEKS:
        # Peak at week 28 (real data: week 28 had 935 cases)
        amplitude = 0.8 * np.sin(np.pi * (week - 24) / 8)
        base = 1.0 + amplitude
    
    # Tertiary peak: Dec (year-end surge)
    elif week in DENGUE_TERTIARY_PEAK_WEEKS:
        amplitude = 0.6 * np.sin(np.pi * (week - 47) / 5)
        base = 1.0 + amplitude
    
    # Trough: Apr-May (lowest period)
    elif week in DENGUE_LOW_WEEKS:
        base = 0.7  # 30% below baseline
    
    else:
        base = 1.0
    
    # Apply regional weight
    weighted = base * regional_weight
    
    # Add realistic noise (CV ~15% from real data)
    noise = np.random.normal(1.0, 0.12)
    
    # Occasional outbreak spikes (2% probability)
    if np.random.random() < 0.02:
        spike = 1.0 + np.random.exponential(0.4)
    else:
        spike = 1.0
    
    return round(max(0.5, min(3.0, weighted * noise * spike)), 3)

def flu_index_v4(week):
    """
    V4: REAL-WORLD CALIBRATED flu index
    Based on: 02_DISEASE_005_WHO_FluNet.csv (Sri Lanka filtered)
    
    Key insight: PRIMARY PEAK IS MAY, not Jan!
    - Primary: Weeks 17-21 (May) - 92 cases in week 19
    - Secondary: Weeks 1-4 + 52 (Jan/Dec) - 115 cases in week 1
    - Trough: Weeks 32-40 (Aug-Oct) - only 13 cases in week 37
    """
    # Primary peak: May (MAIN PEAK - corrected from V2!)
    if week in FLU_PRIMARY_PEAK_WEEKS:
        amplitude = 0.70 * np.sin(np.pi * (week - 16) / 5)
        base = 1.0 + amplitude
    
    # Secondary peak: Jan + Dec
    elif week in FLU_SECONDARY_PEAK_WEEKS:
        if week <= 4:
            amplitude = 0.55 * np.sin(np.pi * week / 4)
        else:  # week 52
            amplitude = 0.50
        base = 1.0 + amplitude
    
    # Trough: Aug-Oct (lowest period)
    elif week in FLU_LOW_WEEKS:
        base = 0.75  # 25% below baseline
    
    else:
        base = 1.0
    
    # Add realistic noise
    noise = np.random.normal(1.0, 0.10)
    
    return round(max(0.6, min(2.0, base * noise)), 3)

def school_term_multiplier(week):
    """Sri Lankan school terms (same as V2)"""
    school_weeks = set(range(1, 17)) | set(range(19, 35)) | set(range(37, 52))
    return 1.15 if week in school_weeks else 0.85

def get_rainfall_v4(week, region):
    """V4: Rainfall with realistic variability"""
    p = REGION_RAIN_PROFILE[region]
    
    # SW Monsoon: May-Sept (weeks 18-39)
    sw = p["sw"] * max(0, np.sin(np.pi * (week - 18) / 21)) if 18 <= week <= 39 else 0
    
    # NE Monsoon: Nov-Jan (weeks 44-4)
    ne_w = week - 44 if week >= 44 else week + 8
    ne = p["ne"] * max(0, np.sin(np.pi * ne_w / 14)) if (week >= 44 or week <= 4) else 0
    
    # Inter-monsoon periods
    inter = 0.4 if week in range(12, 17) or week in range(40, 44) else 0
    
    # Base calculation
    raw = p["base"] * (1 + sw + ne + inter)
    
    # Add realistic variability (CV ~18%)
    noise = np.random.normal(1.0, 0.18)
    
    # Occasional extreme weather (2% flood, 1% drought)
    extreme = np.random.choice([0.3, 1.0, 1.8], p=[0.01, 0.97, 0.02])
    
    return round(max(5, raw * noise * extreme), 1)

def get_weather(rain):
    """Weather category from rainfall"""
    if rain < 20:  return "Drought"
    if rain < 50:  return "Sunny"
    if rain < 90:  return "Cloudy"
    if rain < 150: return "Rainy"
    return "Heavy Rain"

def get_category_seasonal_multiplier(sku_id, month):
    """
    V4: Category-specific seasonal multiplier
    Based on: Medicine Demand Time Series analysis
    
    Paracetamol peaks Oct-Jan (2.15x swing)
    Respiratory peaks Dec-Jan (2.6x swing)
    Antihistamines peak Apr-May (3.5x swing)
    """
    sens = SKU_DISEASE_SENSITIVITY.get(sku_id, {})
    peak_months = sens.get("seasonal_peak_month", [])
    
    if not peak_months:
        return 1.0  # Chronic meds: no seasonality
    
    if month in peak_months:
        return 1.3  # 30% boost in peak months
    else:
        return 0.9  # 10% reduction in off-peak

def calculate_demand_v4(sku, region_name, region_info, d_idx, f_idx, s_mult, rain, week, month, year):
    """
    V4: REAL-WORLD CALIBRATED demand calculation
    Incorporates: Disease signals, seasonality, regional factors, realistic variability
    """
    sid = sku["sku_id"]
    sens = SKU_DISEASE_SENSITIVITY[sid]
    
    # Base regional scale
    region_scale = (region_info["pharmacies"] / 280) * region_info["urban_factor"]
    
    # Disease-driven demand
    disease_effect = (
        1.0
        + sens["dengue"] * (d_idx - 1.0)
        + sens["flu"] * (f_idx - 1.0)
        + sens["school"] * (s_mult - 1.0)
        + sens["rain"] * (min(rain, 200) / 200)
    )
    
    # Category-specific seasonality
    category_seasonal = get_category_seasonal_multiplier(sid, month)
    
    # December holiday effect (real: +25%)
    december_effect = DECEMBER_MULTIPLIER if month == 12 else 1.0
    
    # Year-over-year growth (post-COVID recovery trajectory)
    if year == 2023:
        year_factor = 0.92  # Post-COVID recovery
    elif year == 2024:
        year_factor = 0.97  # Continued recovery
    else:  # 2025
        year_factor = 1.02  # Normal + slight growth
    
    # Random promotional spikes (8% of weeks, +35% lift)
    promo_effect = 1.0
    if np.random.random() < PROMO_PROBABILITY:
        promo_effect = 1.0 + PROMO_LIFT
    
    # Calculate base units
    base_units = (
        sku["base_demand"]
        * region_scale
        * disease_effect
        * category_seasonal
        * december_effect
        * year_factor
        * promo_effect
    )
    
    # Add realistic variability (CV = 45%, LogNormal distribution)
    # LogNormal prevents negative values and creates realistic right-skew
    # Use smaller sigma to control CV - lognormal CV = sqrt(exp(sigma^2) - 1)
    # For CV ~45%, sigma ~0.42
    noise = np.random.lognormal(0, 0.38)  # Controlled to achieve ~45% CV
    noise = min(noise, 2.5)  # Cap extreme outliers
    
    units = max(1, int(base_units * noise))
    
    return units

def simulate_stock_v4(units_ordered, region, lead_time, shelf_life, week, rain):
    """
    V4: REAL-WORLD CALIBRATED stock simulation
    Based on: Supplier Disruption data analysis
    
    Stockout: 1.5% baseline, 4% during monsoon
    Late delivery: 12% probability
    """
    safety = SAFETY_STOCK_WEEKS.get(region, 2.0)
    target_stock = units_ordered * safety
    
    # Stock variance (pharmacists sometimes over/under-stock)
    variance = np.random.normal(1.0, 0.30)
    stock = max(1, int(target_stock * variance))
    
    # Stockout probability
    # Base: 1.5%, increases during monsoon weeks and high rainfall
    is_monsoon_week = (18 <= week <= 39) or (week >= 44 or week <= 4)
    is_high_rain = rain > 150
    
    if is_monsoon_week and is_high_rain:
        stockout_prob = STOCKOUT_MONSOON_PROB + 0.02  # 6%
    elif is_monsoon_week or is_high_rain:
        stockout_prob = STOCKOUT_MONSOON_PROB  # 4%
    else:
        stockout_prob = STOCKOUT_BASE_PROB  # 1.5%
    
    # Increase stockout prob if demand spike exceeds stock
    if units_ordered > stock * 1.3:
        stockout_prob += 0.02
    
    stockout = int(np.random.random() < stockout_prob)
    
    return stock, stockout

def get_lead_time_v4(region, rain, week):
    """
    V4: REAL-WORLD CALIBRATED lead time
    Based on: USAID SCMS analysis (12% late, 20-day std dev)
    """
    base = REGIONS[region]["base_lead_days"]
    
    # Rainfall delays
    if rain > 200:
        delay = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
    elif rain > 150:
        delay = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
    elif rain > 100:
        delay = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
    else:
        delay = np.random.choice([0, 1], p=[0.85, 0.15])
    
    # Late delivery events (12% probability based on real data)
    if np.random.random() < LATE_DELIVERY_PROB:
        # Delay distribution: mostly 7-14 days, sometimes 30+
        extra_delay = np.random.choice([7, 14, 21, 30], p=[0.5, 0.3, 0.15, 0.05])
        delay += extra_delay // 3  # Convert to rough lead time impact
    
    # Occasional supply chain disruptions (1% of weeks)
    if np.random.random() < 0.01:
        delay += np.random.choice([5, 7, 10])
    
    return int(base + delay)

def get_expiry_risk_v4(sku, stock_units, units_ordered):
    """V4: Realistic expiry risk (target 2-5%)"""
    shelf = sku["shelf_life_weeks"]
    ratio = stock_units / max(1, units_ordered)
    
    # Short shelf life + high stock ratio = expiry risk
    if shelf < 60 and ratio > 2.0:
        return int(np.random.random() < 0.30)  # 30% chance
    if shelf < 104 and ratio > 2.5:
        return int(np.random.random() < 0.20)  # 20% chance
    if shelf < 130 and ratio > 3.0:
        return int(np.random.random() < 0.15)  # 15% chance
    if shelf < 260 and ratio > 4.0:
        return int(np.random.random() < 0.08)  # 8% chance
    
    return 0

# ============================================================================
# MAIN V4 GENERATOR
# ============================================================================

def generate_pharmacy_orders_v4():
    """Generate 3 years of REAL-WORLD CALIBRATED pharmacy order data"""
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
    records = []
    
    print(f"Generating V4 dataset: {len(dates)} weeks × {len(REGIONS)} regions × {len(SKUS)} SKUs")
    print(f"Expected records: {len(dates) * len(REGIONS) * len(SKUS):,}")
    
    for date in dates:
        week = int(date.isocalendar()[1])
        month = date.month
        year = date.year
        
        s_mult = school_term_multiplier(week)
        f_idx = flu_index_v4(week)
        
        for region_name, region_info in REGIONS.items():
            d_idx = dengue_index_v4(week, region_name)  # Regional dengue weighting
            pop_density = round(region_info["population"] / region_info["area_km2"], 1)
            rain = get_rainfall_v4(week, region_name)
            weather = get_weather(rain)
            
            for sku in SKUS:
                sid = sku["sku_id"]
                
                # V4: Real-world calibrated demand
                units = calculate_demand_v4(
                    sku, region_name, region_info,
                    d_idx, f_idx, s_mult, rain,
                    week, month, year
                )
                
                # V4: Real-world calibrated supply chain
                lead = get_lead_time_v4(region_name, rain, week)
                stock, stockout = simulate_stock_v4(units, region_name, lead, sku["shelf_life_weeks"], week, rain)
                expiry = get_expiry_risk_v4(sku, stock, units)
                
                records.append({
                    # Identifiers
                    "date": date.strftime("%Y-%m-%d"),
                    "week": week,
                    "year": year,
                    "region": region_name,
                    "sku_id": sid,
                    "medicine_name": sku["name"],
                    "category": sku["category"],
                    # Demand
                    "units_ordered": units,
                    "unit_price_lkr": round(sku["unit_price"] * 310, 2),
                    # Supply chain
                    "stock_level_units": stock,
                    "lead_time_days": lead,
                    "stockout_flag": stockout,
                    "expiry_risk_flag": expiry,
                    # Product metadata
                    "generic_vs_brand": "Generic" if sku["is_generic"] else "Brand",
                    "shelf_life_weeks": sku["shelf_life_weeks"],
                    # Regional context
                    "pharmacy_count_region": region_info["pharmacies"],
                    "population_density_per_km2": pop_density,
                    # Weather / environment
                    "rainfall_mm": rain,
                    "weather_condition": weather,
                    # Epidemiological signals
                    "dengue_index": d_idx,
                    "flu_index": f_idx,
                    "school_term_active": s_mult > 1.0,
                })
    
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df

def generate_health_signals_v4():
    """Generate health signals companion file"""
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
    records = []
    
    for date in dates:
        week = int(date.isocalendar()[1])
        d_idx = dengue_index_v4(week, "Western")  # Use Western as national proxy
        f_idx = flu_index_v4(week)
        
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "week": week,
            "year": date.year,
            "dengue_cases_national": int(np.random.normal(400 * d_idx, 60)),
            "flu_cases_national": int(np.random.normal(150 * f_idx, 25)),
            "dengue_alert_level": "HIGH" if d_idx > 1.8 else "MEDIUM" if d_idx > 1.3 else "LOW",
            "flu_alert_level": "HIGH" if f_idx > 1.5 else "MEDIUM" if f_idx > 1.2 else "LOW",
            "school_term_active": school_term_multiplier(week) > 1.0,
        })
    
    return pd.DataFrame(records)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("PharmaForesight V4 — REAL-WORLD CALIBRATED Data Generator")
    print("=" * 70)
    print("\n⏳ Generating enhanced pharmacy order data (V4)...")
    
    orders = generate_pharmacy_orders_v4()
    orders.to_csv(OUTPUT_FILE, index=False)
    
    print("⏳ Generating public health signals...")
    signals = generate_health_signals_v4()
    signals.to_csv(HEALTH_FILE, index=False)
    
    print("\n" + "=" * 70)
    print("✅ PharmaForesight V4 — COMPLETE")
    print("=" * 70)
    print(f"  Records   : {len(orders):,}")
    print(f"  Columns   : {len(orders.columns)}  →  {list(orders.columns)}")
    print(f"  Date range: {orders['date'].min().date()} → {orders['date'].max().date()}")
    
    # V4 Validation Metrics
    print("\n📊 V4 VALIDATION METRICS:")
    print("-" * 50)
    
    # Disease correlations
    dengue_corr = orders['units_ordered'].corr(orders['dengue_index'])
    flu_corr = orders['units_ordered'].corr(orders['flu_index'])
    print(f"  Dengue-Demand Correlation: {dengue_corr:.3f} (target: 0.20-0.35)")
    print(f"  Flu-Demand Correlation:    {flu_corr:.3f} (target: 0.10-0.20)")
    
    # Supply chain metrics
    stockout_rate = orders['stockout_flag'].mean() * 100
    expiry_rate = orders['expiry_risk_flag'].mean() * 100
    print(f"  Stockout Rate:             {stockout_rate:.2f}% (target: 1.5-4%)")
    print(f"  Expiry Risk Rate:          {expiry_rate:.2f}% (target: 2-5%)")
    
    # Demand variability
    demand_cv = orders['units_ordered'].std() / orders['units_ordered'].mean()
    print(f"  Demand CV:                 {demand_cv:.1%} (target: ~45%)")
    
    # Missing values
    missing = orders.isnull().sum().sum()
    print(f"  Missing Values:            {missing} (target: 0)")
    
    print("\n📦 Supply Chain Stats:")
    print(f"  Stockout flags    : {orders['stockout_flag'].sum():,}")
    print(f"  Expiry risk flags : {orders['expiry_risk_flag'].sum():,}")
    
    print("\n🌧️ Weather distribution:")
    for cond, cnt in orders["weather_condition"].value_counts().items():
        print(f"    {cond:<12} {cnt:>6,}  ({cnt/len(orders)*100:.1f}%)")
    
    print("\n🏭 Stockout rate by region:")
    for region, rate in (orders.groupby("region")["stockout_flag"].mean()*100).sort_values(ascending=False).items():
        print(f"    {region:<20} {rate:4.1f}%  {'█' * int(rate)}")
    
    print("\n🦟 Dengue index range by region:")
    for region in orders["region"].unique():
        region_data = orders[orders["region"] == region]["dengue_index"]
        print(f"    {region:<20} min={region_data.min():.2f}  max={region_data.max():.2f}  mean={region_data.mean():.2f}")
    
    print("\n✅ V4 Dataset ready for forecasting!")
