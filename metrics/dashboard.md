# Metrics & Dashboard Ideas

This repo now emits per-call metrics to `metrics/cost_log.csv` (and Supabase when configured). A lightweight way to visualize cost and latency is to load that CSV into a notebook and chart it.

## Quick Jupyter sketch

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics/cost_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Total cost by component
cost_by_component = df.groupby("component")["cost_usd"].sum().sort_values(ascending=False)
cost_by_component.plot(kind="barh", title="Total cost (USD) by component")
plt.tight_layout()

# Latency distribution
df.boxplot(column="latency_ms", by="component", rot=45)
plt.suptitle("")
plt.title("Latency (ms) by component")
plt.tight_layout()

# Cost over time
df.set_index("timestamp").resample("5min")["cost_usd"].sum().plot(title="Spend per 5 minutes")
plt.tight_layout()
plt.show()
```

## Grafana-style mockup

- **Panel 1:** Stat for `sum(cost_usd)` grouped by component with a threshold line at your daily budget.
- **Panel 2:** Bar chart of average `latency_ms` per component (System1 vs System2 vs Apify vs Google Maps vs Scheduler).
- **Panel 3:** Time series of `cost_usd` over time with an annotation channel for deployment markers.
- **Panel 4:** Table of recent requests (timestamp, component, model/tool, tokens_in/out, latency_ms, cost_usd, conversation_id) filtered to the last hour.

These panels can be powered either from Supabase (table `metrics`) or by tailing the CSV via a simple Prometheus exporter if you prefer a pull-based setup.
