"""Build results_table.csv / results_table.md from results.jsonl (all runs)."""
import json, csv
rs = [json.loads(l) for l in open("results.jsonl")]
cols = ["name", "width", "depth", "act", "opt", "lr", "steps", "win", "win_final",
        "skip", "phys_init", "seed", "n_params", "train_rmse", "test_rmse"]
rows = [[r.get(c) for c in cols] for r in rs]
with open("results_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols); w.writerows(rows)
fmt = lambda c, v: ("" if v is None else f"{v:.5f}" if c.endswith("rmse") else str(v))
with open("results_table.md", "w") as f:
    f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for row in rows:
        f.write("| " + " | ".join(fmt(c, v) for c, v in zip(cols, row)) + " |\n")
print(len(rows), "runs")
