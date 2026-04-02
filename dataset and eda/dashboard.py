"""
PharmaForesight — Streamlit Dashboard
Team: MatriXplorers | University of Moratuwa

Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaForesight | Hemas Pharmaceuticals",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2342 0%, #1a3a5c 100%);
    }
    section[data-testid="stSidebar"] * { color: #e8f0fe !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label { color: #a8c4e0 !important; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1a3a5c;
        margin-bottom: 8px;
    }
    .metric-card.red   { border-left-color: #e53e3e; }
    .metric-card.amber { border-left-color: #dd6b20; }
    .metric-card.green { border-left-color: #276749; }
    .metric-card.blue  { border-left-color: #2b6cb0; }

    .metric-val  { font-size: 2rem; font-weight: 700; color: #1a202c; }
    .metric-lbl  { font-size: 0.82rem; color: #718096; margin-top: 2px; font-weight: 500; }
    .metric-sub  { font-size: 0.75rem; color: #a0aec0; margin-top: 4px; }

    /* Alert badges */
    .badge-critical { background:#fff5f5; color:#c53030; border:1px solid #fc8181;
                      border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:700; }
    .badge-high     { background:#fffaf0; color:#c05621; border:1px solid #f6ad55;
                      border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:700; }
    .badge-medium   { background:#fffff0; color:#b7791f; border:1px solid #f6e05e;
                      border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:700; }

    /* Section headers */
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1a3a5c;
        border-bottom: 2px solid #bee3f8; padding-bottom: 6px; margin-bottom: 16px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600; color: #4a5568;
    }
    .stTabs [aria-selected="true"] {
        color: #1a3a5c !important;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #0a2342 0%, #1a3a5c 60%, #2b6cb0 100%);
        padding: 24px 32px; border-radius: 14px; margin-bottom: 24px;
        display: flex; align-items: center; justify-content: space-between;
    }
    .header-title { color: white; font-size: 1.8rem; font-weight: 800; margin: 0; }
    .header-sub   { color: #a8c4e0; font-size: 0.9rem; margin-top: 4px; }
    .header-badge {
        background: rgba(255,255,255,0.15); color: white;
        border-radius: 20px; padding: 6px 18px; font-size: 0.8rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_orders():
    return pd.read_csv("data/pharmacy_orders.csv", parse_dates=["date"])

@st.cache_data
def load_forecasts():
    return pd.read_csv("data/forecasts.csv", parse_dates=["forecast_date"])

@st.cache_data
def load_alerts():
    df = pd.read_csv("data/procurement_alerts.csv", parse_dates=["date","reorder_by_date"])
    return df

@st.cache_data
def load_signals():
    return pd.read_csv("data/health_signals.csv", parse_dates=["date"])

orders    = load_orders()
forecasts = load_forecasts()
alerts    = load_alerts()
signals   = load_signals()

REGIONS  = sorted(orders["region"].unique())
SKUS     = sorted(orders["medicine_name"].unique())
SKU_MAP  = orders[["sku_id","medicine_name"]].drop_duplicates().set_index("medicine_name")["sku_id"].to_dict()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 PharmaForesight")
    st.markdown("*AI Demand Forecasting*")
    st.markdown("---")

    st.markdown("### 🎛️ Filters")
    sel_region = st.selectbox("Region", ["All Regions"] + REGIONS)
    sel_sku    = st.selectbox("Medicine", ["All Medicines"] + SKUS)

    st.markdown("---")
    st.markdown("### 🦟 Scenario Simulation")
    st.markdown("Simulate a disease outbreak to see demand impact:")
    dengue_boost = st.slider("Dengue Outbreak Intensity", 0, 100, 0,
                             help="0 = normal, 100 = severe outbreak")
    flu_boost    = st.slider("Flu Season Intensity", 0, 100, 0)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
**Team:** MatriXplorers  
**University:** University of Moratuwa  
**SBU:** Hemas Pharmaceuticals  
**Track:** Analytics & Consumer Intelligence  
    """)


# ── Apply Filters ──────────────────────────────────────────────────────────────
def filter_df(df, date_col="date"):
    d = df.copy()
    if sel_region != "All Regions":
        d = d[d["region"] == sel_region]
    if sel_sku != "All Medicines":
        med_col = "medicine_name" if "medicine_name" in d.columns else None
        if med_col:
            d = d[d[med_col] == sel_sku]
    return d

orders_f    = filter_df(orders)
forecasts_f = filter_df(forecasts, "forecast_date").copy()
alerts_f    = filter_df(alerts)


# ── Apply Scenario Simulation ──────────────────────────────────────────────────
if dengue_boost > 0 or flu_boost > 0:
    dengue_sensitive = ["Dengue Rapid Test Kit","ORS Sachets","Paracetamol 500mg",
                        "Ibuprofen 400mg","Paediatric Paracetamol Syrup"]
    flu_sensitive    = ["Amoxicillin 500mg","Azithromycin 250mg","Cetirizine 10mg",
                        "Paracetamol 500mg","Salbutamol Inhaler"]
    db = dengue_boost / 100 * 0.85
    fb = flu_boost    / 100 * 0.50
    def apply_boost(row):
        mult = 1.0
        if row["medicine_name"] in dengue_sensitive: mult += db
        if row["medicine_name"] in flu_sensitive:    mult += fb
        return round(row["ensemble_forecast"] * mult)
    forecasts_f["ensemble_forecast"] = forecasts_f.apply(apply_boost, axis=1)


# ── Header ─────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%d %B %Y, %H:%M")
st.markdown(f"""
<div class="header-banner">
  <div>
    <div class="header-title">💊 PharmaForesight</div>
    <div class="header-sub">AI-Powered Demand Forecasting · Hemas Pharmaceuticals · Sri Lanka</div>
  </div>
  <div class="header-badge">🕐 {now_str}</div>
</div>
""", unsafe_allow_html=True)


# ── KPI Cards ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total_forecast  = int(forecasts_f["ensemble_forecast"].sum())
critical_count  = int((alerts_f["severity"] == "CRITICAL").sum())
high_count      = int((alerts_f["severity"] == "HIGH").sum())
regions_active  = orders_f["region"].nunique()
skus_active     = orders_f["sku_id"].nunique()
reorder_total   = int(alerts_f[alerts_f["severity"].isin(["CRITICAL","HIGH"])]["recommended_reorder_qty"].sum())

with col1:
    st.markdown(f"""
    <div class="metric-card blue">
      <div class="metric-val">{total_forecast:,}</div>
      <div class="metric-lbl">📦 Units Forecasted (4 Weeks)</div>
      <div class="metric-sub">{skus_active} SKUs · {regions_active} regions</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card red">
      <div class="metric-val">{critical_count}</div>
      <div class="metric-lbl">🔴 Critical Alerts</div>
      <div class="metric-sub">Immediate procurement action needed</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card amber">
      <div class="metric-val">{high_count}</div>
      <div class="metric-lbl">🟠 High Alerts</div>
      <div class="metric-sub">Action within 2 weeks recommended</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card green">
      <div class="metric-val">{reorder_total:,}</div>
      <div class="metric-lbl">🛒 Urgent Reorder Units</div>
      <div class="metric-sub">Across critical + high alerts</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecast",
    "🗺️ Regional Heatmap",
    "🚨 Anomaly Alerts",
    "🛒 Procurement Plan",
    "🎮 Scenario Simulation",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📈 Demand Forecast — Next 4 Weeks</div>',
                unsafe_allow_html=True)

    # Historical + Forecast combined chart
    hist = orders_f.groupby("date")["units_ordered"].sum().reset_index()
    hist.columns = ["date","units"]

    fore = forecasts_f.groupby("forecast_date")["ensemble_forecast"].sum().reset_index()
    fore.columns = ["date","units"]
    fore_lower = forecasts_f.groupby("forecast_date")["lower_bound"].sum().reset_index()
    fore_upper = forecasts_f.groupby("forecast_date")["upper_bound"].sum().reset_index()

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["units"],
        mode="lines", name="Historical Demand",
        line=dict(color="#2b6cb0", width=2),
    ))

    # Confidence interval band
    fig.add_trace(go.Scatter(
        x=pd.concat([fore["date"], fore["date"][::-1]]),
        y=pd.concat([fore_upper["upper_bound"], fore_lower["lower_bound"][::-1]]),
        fill="toself", fillcolor="rgba(237,100,166,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval", showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fore["date"], y=fore["units"],
        mode="lines+markers", name="AI Forecast",
        line=dict(color="#e53e3e", width=3, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))

    # Divider line
    split = orders_f["date"].max()
    fig.add_vline(x=split, line_dash="dot", line_color="#718096",
                  annotation_text="Forecast →", annotation_position="top right")

    fig.update_layout(
        height=380, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#edf2f7"),
        yaxis=dict(showgrid=True, gridcolor="#edf2f7", title="Units Ordered"),
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-SKU breakdown
    st.markdown('<div class="section-title">💊 Forecast by Medicine (Next 4 Weeks)</div>',
                unsafe_allow_html=True)
    sku_fore = (forecasts_f.groupby(["medicine_name","category"])["ensemble_forecast"]
                .sum().reset_index().sort_values("ensemble_forecast", ascending=False))

    fig2 = px.bar(sku_fore, x="ensemble_forecast", y="medicine_name",
                  color="category", orientation="h",
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  labels={"ensemble_forecast":"Forecasted Units","medicine_name":"Medicine"})
    fig2.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0),
                       plot_bgcolor="white", paper_bgcolor="white",
                       yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — REGIONAL HEATMAP
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🗺️ Regional Demand Heatmap</div>',
                unsafe_allow_html=True)

    # Province coordinates (centroid approx)
    COORDS = {
        "Western":       (6.921,  79.858),
        "Central":       (7.291,  80.636),
        "Southern":      (6.035,  80.700),
        "Northern":      (9.665,  80.007),
        "Eastern":       (7.867,  81.600),
        "North Western": (7.953,  80.018),
        "North Central": (8.335,  80.405),
        "Uva":           (6.993,  81.055),
        "Sabaragamuwa":  (6.747,  80.365),
    }

    region_demand = (orders_f.groupby("region")["units_ordered"].sum().reset_index())
    region_demand["lat"] = region_demand["region"].map(lambda r: COORDS[r][0])
    region_demand["lon"] = region_demand["region"].map(lambda r: COORDS[r][1])
    region_demand["alert_count"] = region_demand["region"].map(
        alerts_f.groupby("region").size().to_dict()).fillna(0).astype(int)

    fig3 = px.scatter_mapbox(
        region_demand, lat="lat", lon="lon",
        size="units_ordered", color="units_ordered",
        hover_name="region",
        hover_data={"units_ordered":True, "alert_count":True, "lat":False, "lon":False},
        color_continuous_scale=["#bee3f8","#2b6cb0","#1a3a5c"],
        size_max=55, zoom=6.5,
        mapbox_style="carto-positron",
        center={"lat":7.8731,"lon":80.7718},
        labels={"units_ordered":"Total Units","alert_count":"Alerts"},
    )
    fig3.update_layout(height=480, margin=dict(l=0,r=0,t=0,b=0),
                       coloraxis_showscale=True)
    st.plotly_chart(fig3, use_container_width=True)

    # Bar comparison
    st.markdown('<div class="section-title">📊 Region Comparison</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig4 = px.bar(region_demand.sort_values("units_ordered", ascending=True),
                      x="units_ordered", y="region", orientation="h",
                      color="units_ordered",
                      color_continuous_scale=["#bee3f8","#1a3a5c"],
                      labels={"units_ordered":"Total Units Ordered","region":"Region"},
                      title="Total Demand by Region (Historical)")
        fig4.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        reg_alerts = (alerts_f.groupby(["region","severity"]).size()
                      .reset_index(name="count"))
        fig5 = px.bar(reg_alerts, x="region", y="count", color="severity",
                      color_discrete_map={"CRITICAL":"#e53e3e","HIGH":"#dd6b20","MEDIUM":"#d69e2e"},
                      title="Alerts by Region & Severity",
                      labels={"count":"Alert Count","region":"Region","severity":"Severity"})
        fig5.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0,r=0,t=40,b=0),
                           xaxis_tickangle=-30)
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY ALERTS
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🚨 Procurement Alerts — Demand Anomalies</div>',
                unsafe_allow_html=True)

    # Severity filter
    sev_filter = st.multiselect("Filter by Severity",
                                ["CRITICAL","HIGH","MEDIUM"],
                                default=["CRITICAL","HIGH"])
    alerts_show = alerts_f[alerts_f["severity"].isin(sev_filter)].copy()

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Critical", int((alerts_show["severity"]=="CRITICAL").sum()))
    c2.metric("🟠 High",     int((alerts_show["severity"]=="HIGH").sum()))
    c3.metric("🟡 Medium",   int((alerts_show["severity"]=="MEDIUM").sum()))

    # Alert table with color
    def color_severity(val):
        colors = {"CRITICAL":"background-color:#fff5f5;color:#c53030;font-weight:700",
                  "HIGH":    "background-color:#fffaf0;color:#c05621;font-weight:700",
                  "MEDIUM":  "background-color:#fffff0;color:#b7791f;font-weight:700"}
        return colors.get(val, "")

    display_cols = ["date","region","medicine_name","actual_units","baseline_units",
                    "deviation_pct","z_score","severity","alert_reason","recommended_reorder_qty"]
    display_labels = {
        "date":"Date","region":"Region","medicine_name":"Medicine",
        "actual_units":"Actual","baseline_units":"Baseline",
        "deviation_pct":"Deviation %","z_score":"Z-Score",
        "severity":"Severity","alert_reason":"Reason","recommended_reorder_qty":"Reorder Qty"
    }

    tbl = (alerts_show[display_cols]
           .rename(columns=display_labels)
           .sort_values("Date", ascending=False)
           .head(50))

    styled = tbl.style.applymap(color_severity, subset=["Severity"])
    st.dataframe(styled, use_container_width=True, height=400)

    # Z-score over time
    st.markdown('<div class="section-title">📉 Demand Anomaly Timeline</div>',
                unsafe_allow_html=True)
    top_sku = alerts_f["medicine_name"].value_counts().index[0]
    anomaly_ts = orders_f[orders_f["medicine_name"]==top_sku].groupby("date")["units_ordered"].sum().reset_index()
    alert_pts  = alerts_f[alerts_f["medicine_name"]==top_sku]

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=anomaly_ts["date"], y=anomaly_ts["units_ordered"],
                              mode="lines", name="Demand", line=dict(color="#2b6cb0", width=2)))
    for sev, color, sym in [("CRITICAL","#e53e3e","x"),("HIGH","#dd6b20","triangle-up"),("MEDIUM","#d69e2e","circle")]:
        pts = alert_pts[alert_pts["severity"]==sev]
        fig6.add_trace(go.Scatter(
            x=pts["date"], y=pts["actual_units"],
            mode="markers", name=f"{sev} Alert",
            marker=dict(color=color, size=10, symbol=sym),
        ))
    fig6.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=0,r=0,t=10,b=0),
                       xaxis=dict(showgrid=True, gridcolor="#edf2f7"),
                       yaxis=dict(title=f"Units — {top_sku}", showgrid=True, gridcolor="#edf2f7"),
                       hovermode="x unified")
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — PROCUREMENT PLAN
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🛒 Procurement Recommendation Plan</div>',
                unsafe_allow_html=True)

    urgent = alerts_f[alerts_f["severity"].isin(["CRITICAL","HIGH"])].copy()
    plan = (urgent.groupby(["medicine_name","category"])
            .agg(
                total_reorder=("recommended_reorder_qty","sum"),
                regions_affected=("region","nunique"),
                avg_deviation=("deviation_pct","mean"),
                latest_alert=("date","max"),
            )
            .reset_index()
            .sort_values("total_reorder", ascending=False))
    plan["avg_deviation"]   = plan["avg_deviation"].round(1).astype(str) + "%"
    plan["reorder_by"]      = (datetime.now() + timedelta(weeks=2)).strftime("%Y-%m-%d")
    plan["priority"]        = ["🔴 URGENT"] * min(3, len(plan)) + \
                              ["🟠 HIGH"] * (len(plan) - min(3, len(plan)))

    plan.columns = ["Medicine","Category","Total Reorder Qty","Regions Affected",
                    "Avg Deviation","Latest Alert","Reorder By","Priority"]
    st.dataframe(plan, use_container_width=True, height=380)

    # Reorder quantity chart
    fig7 = px.treemap(
        alerts_f[alerts_f["severity"].isin(["CRITICAL","HIGH"])],
        path=["category","medicine_name"],
        values="recommended_reorder_qty",
        color="severity",
        color_discrete_map={"CRITICAL":"#fc8181","HIGH":"#f6ad55"},
        title="Reorder Quantities by Category & Medicine",
    )
    fig7.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig7, use_container_width=True)

    # Forecast table
    st.markdown('<div class="section-title">📋 4-Week Forecast Detail</div>',
                unsafe_allow_html=True)
    fore_tbl = (forecasts_f[["forecast_date","region","medicine_name","category",
                              "ensemble_forecast","lower_bound","upper_bound"]]
                .rename(columns={
                    "forecast_date":"Week","region":"Region","medicine_name":"Medicine",
                    "category":"Category","ensemble_forecast":"Forecast Units",
                    "lower_bound":"Lower Bound","upper_bound":"Upper Bound"})
                .sort_values(["Week","Forecast Units"], ascending=[True, False]))
    st.dataframe(fore_tbl, use_container_width=True, height=340)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — SCENARIO SIMULATION
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🎮 What-If Scenario Simulation</div>',
                unsafe_allow_html=True)

    st.info("👈 Use the **sidebar sliders** to simulate a dengue outbreak or flu season. "
            "The forecasts below update in real time.")

    if dengue_boost == 0 and flu_boost == 0:
        st.warning("🟡 No scenario active. Move the sliders in the sidebar to simulate an outbreak.")

    else:
        if dengue_boost > 0:
            severity_label = "SEVERE 🔴" if dengue_boost > 70 else "MODERATE 🟠" if dengue_boost > 30 else "MILD 🟡"
            st.error(f"🦟 **Dengue Outbreak Simulated** — Intensity: {dengue_boost}% ({severity_label})")
        if flu_boost > 0:
            severity_label = "SEVERE 🔴" if flu_boost > 70 else "MODERATE 🟠" if flu_boost > 30 else "MILD 🟡"
            st.warning(f"🤧 **Flu Season Simulated** — Intensity: {flu_boost}% ({severity_label})")

    # Compare baseline vs scenario
    base_fore = load_forecasts()
    if sel_region != "All Regions":
        base_fore = base_fore[base_fore["region"] == sel_region]
    if sel_sku != "All Medicines":
        base_fore = base_fore[base_fore["medicine_name"] == sel_sku]

    base_total = int(base_fore["ensemble_forecast"].sum())
    scen_total = int(forecasts_f["ensemble_forecast"].sum())
    delta      = scen_total - base_total
    delta_pct  = delta / base_total * 100 if base_total > 0 else 0

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("📦 Baseline Forecast", f"{base_total:,} units")
    col_s2.metric("🔮 Scenario Forecast", f"{scen_total:,} units",
                  delta=f"+{delta:,} units (+{delta_pct:.1f}%)" if delta > 0 else f"{delta:,} units")
    col_s3.metric("⚠️ Extra Units Needed", f"{max(0,delta):,}")

    # Side-by-side bar: baseline vs scenario per medicine
    base_sku = base_fore.groupby("medicine_name")["ensemble_forecast"].sum().reset_index()
    base_sku.columns = ["medicine_name","baseline"]
    scen_sku = forecasts_f.groupby("medicine_name")["ensemble_forecast"].sum().reset_index()
    scen_sku.columns = ["medicine_name","scenario"]
    comp = base_sku.merge(scen_sku, on="medicine_name")
    comp["extra"] = comp["scenario"] - comp["baseline"]
    comp = comp.sort_values("extra", ascending=False)

    fig8 = go.Figure()
    fig8.add_trace(go.Bar(name="Baseline", x=comp["medicine_name"], y=comp["baseline"],
                          marker_color="#a0aec0"))
    fig8.add_trace(go.Bar(name="Scenario", x=comp["medicine_name"], y=comp["scenario"],
                          marker_color="#e53e3e"))
    fig8.update_layout(
        barmode="group", height=380,
        title="Baseline vs Scenario Demand per Medicine",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0,r=0,t=40,b=0),
        xaxis_tickangle=-35,
        legend=dict(orientation="h", y=1.08),
        yaxis=dict(title="Forecasted Units"),
    )
    st.plotly_chart(fig8, use_container_width=True)

    # Which medicines are most affected
    st.markdown('<div class="section-title">💊 Most Impacted Medicines</div>',
                unsafe_allow_html=True)
    top_impact = comp[comp["extra"] > 0].sort_values("extra", ascending=False).head(8)
    top_impact["impact_pct"] = ((top_impact["extra"] / top_impact["baseline"]) * 100).round(1)
    top_impact = top_impact[["medicine_name","baseline","scenario","extra","impact_pct"]]
    top_impact.columns = ["Medicine","Baseline Units","Scenario Units","Extra Demand","Impact %"]
    st.dataframe(top_impact, use_container_width=True, height=280)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#a0aec0; font-size:0.8rem;'>"
    "PharmaForesight · MatriXplorers · University of Moratuwa · Hemas Pharmaceuticals · 2026"
    "</div>",
    unsafe_allow_html=True,
)
