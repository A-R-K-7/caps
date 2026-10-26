import csv
from collections import Counter

rows = []
with open("research/results/research_f5f93acf_episodes.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

ep_counter = Counter(r["episode_id"] for r in rows)
dup_eps = [(ep, n) for ep, n in ep_counter.items() if n > 1]
print(f"Duplicate episode IDs: {len(dup_eps)}")

ex_ep = dup_eps[0][0]
ex_rows = [r for r in rows if r["episode_id"] == ex_ep]
print(f"\nDuplicate: {ex_ep}")
for r in ex_rows:
    m = r["method"]
    b = r["budget_fraction"]
    s = r["seed"]
    t = r["task_id"]
    sp = r["split"]
    print(f"  method={m}, budget={b}, seed={s}, task={t}, split={sp}")

print()
ex_ep2 = dup_eps[100][0]
ex_rows2 = [r for r in rows if r["episode_id"] == ex_ep2]
print(f"Duplicate: {ex_ep2}")
for r in ex_rows2:
    m = r["method"]
    b = r["budget_fraction"]
    s = r["seed"]
    t = r["task_id"]
    print(f"  method={m}, budget={b}, seed={s}, task={t}")
