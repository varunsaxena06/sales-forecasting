import json, os
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

here = os.path.dirname(__file__)
proj = os.path.abspath(os.path.join(here, ".."))
data_path = os.path.join(proj, "data", "sales_daily.csv")
reports = os.path.join(proj, "reports")
os.makedirs(reports, exist_ok=True)

# Load data
df = pd.read_csv(data_path, parse_dates=["date"])
df = df.sort_values("date")
df = df[["date", "quantity"]].copy()

# Feature engineering
for lag in [1, 7]:
    df[f"lag_{lag}"] = df["quantity"].shift(lag)
df["roll7_mean"] = df["quantity"].rolling(7).mean()
df = df.dropna().reset_index(drop=True)

# Train/validation split (last 60 days as validation)
train = df.iloc[:-60]
valid = df.iloc[-60:]

X_train = train[["lag_1", "lag_7", "roll7_mean"]].values
y_train = train["quantity"].values
X_valid = valid[["lag_1", "lag_7", "roll7_mean"]].values
y_valid = valid["quantity"].values

model = LinearRegression()
model.fit(X_train, y_train)
pred_valid = model.predict(X_valid).clip(min=0)

mae = float(mean_absolute_error(y_valid, pred_valid))
mape = float(np.mean(np.abs((y_valid - pred_valid) / np.maximum(1, y_valid))) * 100)

metrics = {"MAE": mae, "MAPE": mape, "n_valid": int(len(valid))}
with open(os.path.join(reports, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Rolling forecast for next 30 days
hist = df.copy()
future_days = 30
preds = []
last_date = hist["date"].iloc[-1]
for i in range(future_days):
    row = {}
    row["date"] = last_date + pd.Timedelta(days=i+1)
    lag_1 = hist["quantity"].iloc[-1] if i == 0 else preds[-1]["forecast"]
    lag_7_source = hist["quantity"].iloc[-7+i] if i < 7 else preds[i-7]["forecast"]
    roll7_vals = list(hist["quantity"].iloc[-6:]) + [lag_1] if i == 0 else [p["forecast"] for p in preds[-6:]] + [lag_1]
    roll7_mean = float(np.mean(roll7_vals))
    X = np.array([[lag_1, lag_7_source, roll7_mean]])
    yhat = float(model.predict(X)[0])
    row["forecast"] = max(0.0, yhat)
    preds.append(row)

df_pred = pd.DataFrame(preds)
df_pred.to_csv(os.path.join(reports, "forecast.csv"), index=False)

# Plot history + forecast
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(df["date"], df["quantity"], label="history")
plt.plot(df_pred["date"], df_pred["forecast"], label="forecast")
plt.title("Sales – History & 30‑day Forecast")
plt.xlabel("Date")
plt.ylabel("Units")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(reports, "forecast_plot.png"))
print("Done. Metrics:", metrics)
