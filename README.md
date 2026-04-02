# PharmaForesight

PharmaForesight is an AI-driven demand forecasting and procurement intelligence project for pharmacy supply chains in Sri Lanka.

It combines:
- Synthetic pharmacy order generation with epidemiological and weather signals
- Exploratory data analysis (EDA)
- Multi-model forecasting (Prophet + XGBoost ensemble)
- Anomaly detection and procurement alerting
- Interactive decision dashboard built with Streamlit

## Project Highlights

- Weekly demand simulation for 15 medicine SKUs across 9 Sri Lankan provinces
- Disease-aware features (dengue and flu signals)
- Weather-aware supply features (rainfall, lead-time effects)
- Forecast horizon: 4 weeks ahead at SKU x Region granularity
- Rolling z-score anomaly detection for demand spikes and drops
- Action-oriented procurement alerts with severity levels and reorder quantities

## Folder Structure

```text
.
|-- README.md
`-- dataset and eda/
    |-- data_generator.py
    |-- data_generator_v2.py
    |-- eda_pharmacy_orders.py
    |-- forecasting_model.py
    |-- anomaly_detector.py
    |-- dashboard.py
    |-- code.ipynb
    |-- data/
    |   |-- pharmacy_orders.csv
    |   |-- health_signals.csv
    |   |-- forecasts.csv
    |   |-- anomalies.csv
    |   `-- procurement_alerts.csv
    |-- eda_outputs/
    |   |-- *.png
    |   `-- *.csv
    |-- models/
    |   |-- xgb_model.pkl
    |   |-- prophet_models.pkl
    |   `-- encoders.pkl
    `-- test/
```

## Requirements

- Python 3.10+
- pip

Recommended Python packages:
- pandas
- numpy
- matplotlib
- scikit-learn
- prophet
- xgboost
- streamlit
- plotly

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn prophet xgboost streamlit plotly
```

## Quick Start

Run commands from the project root folder.

### 1. Generate synthetic datasets

```bash
python "dataset and eda/data_generator_v2.py"
```

This creates:
- dataset and eda/data/pharmacy_orders.csv
- dataset and eda/data/health_signals.csv

### 2. Train forecasting models and create forecasts

```bash
python "dataset and eda/forecasting_model.py"
```

This creates:
- dataset and eda/data/forecasts.csv
- dataset and eda/models/xgb_model.pkl
- dataset and eda/models/prophet_models.pkl
- dataset and eda/models/encoders.pkl

### 3. Run anomaly detection and generate procurement alerts

```bash
python "dataset and eda/anomaly_detector.py"
```

This creates:
- dataset and eda/data/anomalies.csv
- dataset and eda/data/procurement_alerts.csv

### 4. Launch dashboard

```bash
streamlit run "dataset and eda/dashboard.py"
```

The dashboard includes:
- Demand forecast trends and confidence bounds
- Regional demand heatmap
- Alert severity monitoring
- Procurement planning table
- Outbreak scenario simulation (dengue and flu)

## EDA Outputs

Optional EDA script:

```bash
python "dataset and eda/eda_pharmacy_orders.py"
```

This populates:
- dataset and eda/eda_outputs/*.png
- dataset and eda/eda_outputs/*.csv

## Data and Modeling Notes

- Forecasting engine uses a hybrid ensemble:
  - Prophet per SKU x Region time series
  - Global XGBoost regressor with lag and seasonal features
- Ensemble output uses weighted blending of Prophet and XGBoost predictions
- Anomaly detector uses rolling mean and rolling standard deviation with z-score thresholding
- Procurement alerts prioritize demand spikes and map to severity levels: MEDIUM, HIGH, CRITICAL

## Typical Workflow

1. Generate or refresh synthetic source data
2. Train forecasting models and export forecasts
3. Detect anomalies and create procurement alerts
4. Review outcomes in the Streamlit dashboard

## Team

MatriXplorers
University of Moratuwa

## License

No license file is currently included. Add a LICENSE file if you plan to distribute this project.
