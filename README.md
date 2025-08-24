# Sales Forecasting & Analysis

**Goal:** Identify seasonality and forecast short‑term sales for a retail SKU using simple, transparent features.

**Tech stack:** Python (pandas, numpy, scikit‑learn), SQL (schema & queries), BI‑ready CSVs.

## Approach
- Create lag/rolling features (t‑1, t‑7, 7‑day mean).
- Train `LinearRegression` on train split; validate on a recent holdout.
- Evaluate with MAE/MAPE; export a next‑30‑day forecast and a plot.

## How to run
```bash
python src/train.py
```
Outputs in `reports/`:
- `forecast.csv`
- `forecast_plot.png`
- `metrics.json`
