# Eval Analysis

- Run eval: `python3 eval/run_eval.py --baseline full-agent` (or `--baseline llm-only` and optionally `--system us|lb`). Results land in `eval/results.json` by default.
- Shortcut script: `bash scripts/run_eval.sh` (honors `BASELINE`, `SYSTEM`, and `RESULTS_PATH` env vars).
- Metrics stored per task in `results[].metrics`:
  - `accuracy.score` (heuristic checklist: listings mention, links/prices, lease/viewing cues, location echo).
  - `reliability` (non-empty reply and no exception).
  - `latency_seconds` (wall clock per task).
  - `cost` (telemetry passthrough if available; falls back to empty).
- Aggregates are precomputed in `summary`: `overall`, `by_system`, and `by_baseline` with averages, p95 latency, and totals.
- Bias probes are stored separately under `bias_results` with an optional `bias_summary`; see `eval/bias_checks.json` for prompts and `eval/human_guidelines.md` for manual review steps.

## Randomness & seeds

- The harness itself is deterministic (no sampling or shuffling); `PYTHONHASHSEED` is pinned to `0` in `scripts/run_eval.sh` to avoid hash-order drift.
- LLM calls remain stochastic (temperature set inside the agents); no explicit seed is provided to OpenAI. Expect minor variability between runs even with identical inputs.

## Quick table (from saved results)

```python
import json
from pathlib import Path

data = json.loads(Path("eval/results.json").read_text())
by_system = data["summary"]["by_system"]
rows = [
    ("US", by_system.get("us", {})),
    ("Lebanon", by_system.get("lb", {})),
]
print("| System | Acc | Reliability | Avg Lat (s) | p95 Lat (s) | Cost ($) |")
print("| --- | --- | --- | --- | --- | --- |")
for label, metrics in rows:
    print(
        f"| {label} | {metrics.get('accuracy_avg', 0):.2f} | {metrics.get('reliability', 0):.2f} | "
        f"{metrics.get('latency_avg_seconds', 0):.2f} | {metrics.get('latency_p95_seconds', 0):.2f} | "
        f"{metrics.get('cost_total_usd', 0):.4f} |"
    )
```

Example format (values will update after a run):

| System | Acc | Reliability | Avg Lat (s) | p95 Lat (s) | Cost ($) |
| --- | --- | --- | --- | --- | --- |
| US | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| Lebanon | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |

## Baseline vs. Full Agent

Use the `--baseline` flag to tag a run, then compare `summary["by_baseline"]` entries. Example snippet:

```python
by_baseline = data["summary"]["by_baseline"]
for name, metrics in by_baseline.items():
    print(name, {k: v for k, v in metrics.items() if k != "count"})
```

## Plotting sketch (optional)

```python
# Requires matplotlib (optional).
import matplotlib.pyplot as plt
labels = list(by_system.keys())
acc = [by_system[k]["accuracy_avg"] for k in labels]
plt.bar(labels, acc)
plt.title("Accuracy by system")
plt.xlabel("System")
plt.ylabel("Accuracy (heuristic)")
plt.tight_layout()
plt.show()
```
