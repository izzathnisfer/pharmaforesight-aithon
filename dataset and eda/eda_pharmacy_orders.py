import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = "pharmacy_orders_v2.csv"   # change path if needed
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 6)


# =========================================================
# HELPERS
# =========================================================
def safe_print(title: str, obj):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(obj)


def save_bar(series, title, xlabel, ylabel, filename, top_n=None, rotate=45):
    plot_series = series.copy()
    if top_n is not None:
        plot_series = plot_series.head(top_n)

    plt.figure(figsize=(12, 6))
    plot_series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


def save_line(series, title, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 6))
    series.plot(kind="line", marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


def save_scatter(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


def plot_category_scatter(df, x_col, y_col, category_col, categories, title_prefix, filename_prefix):
    for cat in categories:
        sub = df[df[category_col] == cat].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(10, 6))
        plt.scatter(sub[x_col], sub[y_col], alpha=0.6)
        plt.title(f"{title_prefix} - {cat}")
        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel("Units Ordered")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{filename_prefix}_{cat.lower().replace(' ', '_')}.png"),
            dpi=200
        )
        plt.close()


def plot_category_avg_trend(df, x_col, y_col, category_col, categories, title_prefix, filename_prefix):
    for cat in categories:
        sub = df[df[category_col] == cat].copy()
        if sub.empty:
            continue

        avg_df = sub.groupby(x_col)[y_col].mean().reset_index().sort_values(x_col)

        plt.figure(figsize=(10, 6))
        plt.plot(avg_df[x_col], avg_df[y_col], marker="o")
        plt.title(f"{title_prefix} - {cat}")
        plt.xlabel(x_col.replace("_", " ").title())
        plt.ylabel("Average Units Ordered")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"{filename_prefix}_{cat.lower().replace(' ', '_')}.png"),
            dpi=200
        )
        plt.close()


def plot_multi_category_avg(df, x_col, y_col, category_col, categories, title, filename):
    plt.figure(figsize=(12, 7))

    for cat in categories:
        sub = df[df[category_col] == cat].copy()
        if sub.empty:
            continue

        avg_df = sub.groupby(x_col)[y_col].mean().reset_index().sort_values(x_col)
        plt.plot(avg_df[x_col], avg_df[y_col], marker="o", label=cat)

    plt.title(title)
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Average Units Ordered")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


def plot_correlation_matrix(corr_df, filename="correlation_matrix.png"):
    plt.figure(figsize=(14, 10))
    matrix = corr_df.values
    im = plt.imshow(matrix, aspect="auto")

    plt.colorbar(im)
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
    plt.yticks(range(len(corr_df.index)), corr_df.index)

    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            plt.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)

safe_print("Dataset Shape", df.shape)
safe_print("Columns", df.columns.tolist())
safe_print("First 5 Rows", df.head())

# =========================================================
# BASIC CLEANING
# =========================================================
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

if "date" in df.columns:
    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)
    df["iso_year_week"] = df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)

# =========================================================
# COLUMN DETECTION
# =========================================================
region_col = "region" if "region" in df.columns else None
sku_col = "sku_name" if "sku_name" in df.columns else ("sku_id" if "sku_id" in df.columns else None)
category_col = "category" if "category" in df.columns else None
demand_col = "units_ordered" if "units_ordered" in df.columns else None
stockout_col = "stockout_flag" if "stockout_flag" in df.columns else None
expiry_col = "expiry_risk_flag" if "expiry_risk_flag" in df.columns else None
dengue_col = "dengue_index" if "dengue_index" in df.columns else None
flu_col = "flu_index" if "flu_index" in df.columns else None
rain_col = "rainfall_mm" if "rainfall_mm" in df.columns else None
lead_time_col = "lead_time_days" if "lead_time_days" in df.columns else None
stock_col = "stock_level_units" if "stock_level_units" in df.columns else None
pharmacy_count_col = "pharmacy_count_region" if "pharmacy_count_region" in df.columns else None
population_density_col = "population_density_per_km2" if "population_density_per_km2" in df.columns else None
school_term_col = "school_term_flag" if "school_term_flag" in df.columns else None

required_core = [region_col, sku_col, category_col, demand_col]
if any(col is None for col in required_core):
    raise ValueError(
        "Missing expected columns. Need at least: region, sku_name or sku_id, category, units_ordered"
    )

# =========================================================
# DATA QUALITY CHECKS
# =========================================================
safe_print("Data Types", df.dtypes)
safe_print("Missing Values", df.isnull().sum().sort_values(ascending=False))
safe_print("Duplicate Rows", df.duplicated().sum())

if {"date", "week"}.issubset(df.columns):
    week_issue = df[["date", "week"]].drop_duplicates().sort_values("date")
    safe_print("Date vs Original Week Sample", week_issue.head(10))
    safe_print("Date vs Original Week Tail", week_issue.tail(10))

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
safe_print("Numeric Summary", df[numeric_cols].describe().T)

overview = {
    "rows": len(df),
    "columns": len(df.columns),
    "date_min": df["date"].min() if "date" in df.columns else None,
    "date_max": df["date"].max() if "date" in df.columns else None,
    "n_regions": df[region_col].nunique(),
    "n_skus": df[sku_col].nunique(),
    "n_categories": df[category_col].nunique(),
}
safe_print("Overview", overview)

# =========================================================
# DEMAND ANALYSIS
# =========================================================
safe_print("Demand Summary", df[demand_col].describe())

demand_by_region = df.groupby(region_col)[demand_col].sum().sort_values(ascending=False)
demand_by_sku = df.groupby(sku_col)[demand_col].sum().sort_values(ascending=False)
demand_by_category = df.groupby(category_col)[demand_col].sum().sort_values(ascending=False)

safe_print("Demand by Region", demand_by_region)
safe_print("Top 15 SKUs by Demand", demand_by_sku.head(15))
safe_print("Demand by Category", demand_by_category)

save_bar(demand_by_region, "Total Demand by Region", "Region", "Units Ordered", "demand_by_region.png")
save_bar(demand_by_sku, "Top 10 SKUs by Total Demand", "SKU", "Units Ordered", "top_10_skus_by_demand.png", top_n=10, rotate=60)
save_bar(demand_by_category, "Demand by Category", "Category", "Units Ordered", "demand_by_category.png")

if "date" in df.columns:
    weekly_demand = df.groupby("date")[demand_col].sum().sort_index()
    safe_print("Weekly Demand Trend", weekly_demand.head(10))
    save_line(weekly_demand, "Total Weekly Demand Over Time", "Date", "Units Ordered", "weekly_demand_trend.png")

# =========================================================
# STOCKOUT / EXPIRY ANALYSIS
# =========================================================
if stockout_col:
    stockout_rate = df[stockout_col].mean()
    safe_print("Overall Stockout Rate", stockout_rate)

    stockout_by_region = df.groupby(region_col)[stockout_col].sum().sort_values(ascending=False)
    stockout_by_sku = df.groupby(sku_col)[stockout_col].sum().sort_values(ascending=False)

    safe_print("Stockouts by Region", stockout_by_region)
    safe_print("Top Stockout SKUs", stockout_by_sku.head(10))

    save_bar(stockout_by_region, "Stockouts by Region", "Region", "Count", "stockouts_by_region.png")
    save_bar(stockout_by_sku, "Top 10 SKUs by Stockouts", "SKU", "Count", "stockouts_by_sku.png", top_n=10, rotate=60)

if expiry_col:
    expiry_rate = df[expiry_col].mean()
    safe_print("Overall Expiry Risk Rate", expiry_rate)

    expiry_by_region = df.groupby(region_col)[expiry_col].sum().sort_values(ascending=False)
    expiry_by_sku = df.groupby(sku_col)[expiry_col].sum().sort_values(ascending=False)

    safe_print("Expiry Risk by Region", expiry_by_region)
    safe_print("Top Expiry Risk SKUs", expiry_by_sku.head(10))

    save_bar(expiry_by_region, "Expiry Risk by Region", "Region", "Count", "expiry_risk_by_region.png")
    save_bar(expiry_by_sku, "Top 10 SKUs by Expiry Risk", "SKU", "Count", "expiry_risk_by_sku.png", top_n=10, rotate=60)

# =========================================================
# CORRELATION ANALYSIS
# =========================================================
corr_candidates = [demand_col]
for c in [
    dengue_col, flu_col, rain_col, lead_time_col, stock_col,
    pharmacy_count_col, population_density_col, school_term_col
]:
    if c:
        corr_candidates.append(c)

corr_candidates = list(dict.fromkeys(corr_candidates))  # remove duplicates
corr_df = df[corr_candidates].corr(numeric_only=True)

safe_print("Correlation Matrix", corr_df)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
plot_correlation_matrix(corr_df, "correlation_matrix.png")

# =========================================================
# BASIC DISEASE SCATTERS
# =========================================================
if dengue_col:
    save_scatter(
        df[dengue_col],
        df[demand_col],
        "Demand vs Dengue Index",
        "Dengue Index",
        "Units Ordered",
        "demand_vs_dengue_all.png",
    )

if flu_col:
    save_scatter(
        df[flu_col],
        df[demand_col],
        "Demand vs Flu Index",
        "Flu Index",
        "Units Ordered",
        "demand_vs_flu_all.png",
    )

# =========================================================
# CATEGORY-SPECIFIC DISEASE ANALYSIS
# =========================================================
dengue_sensitive_categories = ["Antipyretic", "Rehydration", "Paediatric", "Diagnostics"]
flu_sensitive_categories = ["Respiratory", "Antibiotic"]

if dengue_col:
    dengue_corr_by_cat = (
        df.groupby(category_col)
        .apply(lambda x: x[[demand_col, dengue_col]].corr().iloc[0, 1] if len(x) > 1 else np.nan)
        .sort_values(ascending=False)
    )
    safe_print("Demand vs Dengue Correlation by Category", dengue_corr_by_cat)
    dengue_corr_by_cat.to_csv(os.path.join(OUTPUT_DIR, "dengue_corr_by_category.csv"))

    plot_category_scatter(
        df, dengue_col, demand_col, category_col,
        dengue_sensitive_categories,
        "Demand vs Dengue Index",
        "scatter_dengue"
    )

    plot_category_avg_trend(
        df, dengue_col, demand_col, category_col,
        dengue_sensitive_categories,
        "Average Demand by Dengue Index",
        "avg_demand_dengue"
    )

    plot_multi_category_avg(
        df,
        x_col=dengue_col,
        y_col=demand_col,
        category_col=category_col,
        categories=dengue_sensitive_categories,
        title="Average Demand by Dengue Index Across Dengue-Sensitive Categories",
        filename="multi_category_avg_dengue.png"
    )

if flu_col:
    flu_corr_by_cat = (
        df.groupby(category_col)
        .apply(lambda x: x[[demand_col, flu_col]].corr().iloc[0, 1] if len(x) > 1 else np.nan)
        .sort_values(ascending=False)
    )
    safe_print("Demand vs Flu Correlation by Category", flu_corr_by_cat)
    flu_corr_by_cat.to_csv(os.path.join(OUTPUT_DIR, "flu_corr_by_category.csv"))

    plot_category_scatter(
        df, flu_col, demand_col, category_col,
        flu_sensitive_categories,
        "Demand vs Flu Index",
        "scatter_flu"
    )

    plot_category_avg_trend(
        df, flu_col, demand_col, category_col,
        flu_sensitive_categories,
        "Average Demand by Flu Index",
        "avg_demand_flu"
    )

    plot_multi_category_avg(
        df,
        x_col=flu_col,
        y_col=demand_col,
        category_col=category_col,
        categories=flu_sensitive_categories,
        title="Average Demand by Flu Index Across Flu-Sensitive Categories",
        filename="multi_category_avg_flu.png"
    )

# =========================================================
# TOP WEEKS / SEASONAL SPIKES
# =========================================================
if "date" in df.columns:
    top_weeks = (
        df.groupby("date")[demand_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    safe_print("Top 10 Highest Demand Weeks", top_weeks)
    top_weeks.to_csv(os.path.join(OUTPUT_DIR, "top_10_demand_weeks.csv"))

# =========================================================
# PIVOTS
# =========================================================
pivot_region_category = pd.pivot_table(
    df,
    index=region_col,
    columns=category_col,
    values=demand_col,
    aggfunc="sum",
    fill_value=0,
)
pivot_region_category.to_csv(os.path.join(OUTPUT_DIR, "pivot_region_category_demand.csv"))
safe_print("Region x Category Demand Pivot", pivot_region_category)

# =========================================================
# MODELING READINESS CHECKS
# =========================================================
if "date" in df.columns:
    pair_counts = (
        df.groupby([region_col, sku_col])["date"]
        .nunique()
        .reset_index(name="n_weeks")
        .sort_values("n_weeks")
    )
    safe_print("SKU-Region Weekly Coverage", pair_counts.head(20))
    pair_counts.to_csv(os.path.join(OUTPUT_DIR, "sku_region_week_coverage.csv"), index=False)

critical_numeric = [demand_col, stock_col, lead_time_col, dengue_col, flu_col, rain_col]
critical_numeric = [c for c in critical_numeric if c]

negative_check = {}
for col in critical_numeric:
    negative_check[col] = int((df[col] < 0).sum())

safe_print("Negative Value Check", negative_check)

# =========================================================
# SAVE SUMMARY TABLES
# =========================================================
demand_by_region.to_csv(os.path.join(OUTPUT_DIR, "demand_by_region.csv"))
demand_by_sku.to_csv(os.path.join(OUTPUT_DIR, "demand_by_sku.csv"))
demand_by_category.to_csv(os.path.join(OUTPUT_DIR, "demand_by_category.csv"))

# =========================================================
# FINAL
# =========================================================
print("\nEDA completed successfully.")
print(f"Outputs saved to: {OUTPUT_DIR}")

print("\nRecommended next steps:")
print("1. Use 'date' or 'iso_year_week' instead of raw 'week' for forecasting.")
print("2. Create lag features: lag_1, lag_2, lag_4.")
print("3. Create rolling features: rolling_mean_4w, rolling_std_4w.")
print("4. Train a baseline forecast model first.")
print("5. Compare results by SKU, region, and category.")