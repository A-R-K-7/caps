import sys, csv, json
from collections import defaultdict
sys.path.insert(0, "src")

path = "research/results/research_5d19b3f8_episodes.csv"
rows = list(csv.DictReader(open(path)))
cols = list(rows[0].keys())
print(f"Run 5d19b3f8: {len(rows)} episodes, {len(cols)} cols")
print(f"Columns: {cols}")

# Sample
r = rows[0]
for k, v in r.items():
    if v and v != "0" and v != "0.0":
        print(f"  {k}: {v}")

# Success
any_success = sum(1 for r in rows if r.get("task_success") == "True")
print(f"\nSuccess: {any_success}/{len(rows)} = {any_success/len(rows)*100:.1f}%")

# API tokens
api_vals = [(r.get("api_prompt_tokens",""), r.get("api_completion_tokens","")) for r in rows[:5]]
print(f"API tokens (first 5): {api_vals}")

# Latencies
lats = [float(r.get("latency_ms", 0)) for r in rows[:10]]
print(f"Latencies (first 10): {[f'{l:.0f}' for l in lats]}")

# Method breakdown
method_success = defaultdict(list)
method_tokens = defaultdict(list)
method_prec = defaultdict(list)
method_recall = defaultdict(list)
method_api = defaultdict(list)
for r in rows:
    m = r["method"]
    method_success[m].append(r.get("task_success") == "True")
    try: method_tokens[m].append(int(r.get("selected_tokens", 0)))
    except: pass
    try: method_prec[m].append(float(r.get("precision", 0)))
    except: pass
    try: method_recall[m].append(float(r.get("recall", 0)))
    except: pass
    try: method_api[m].append(int(r.get("api_prompt_tokens", 0)))
    except: pass

print(f"\nResults by Method:")
print(f"{'Method':30} {'N':5} {'Success':10} {'Tokens':8} {'Prec':8} {'Recall':8} {'APITok':8}")
for m in sorted(method_success.keys()):
    s = method_success[m]
    t = method_tokens[m]
    p = method_prec[m]
    rc = method_recall[m]
    ap = method_api[m]
    sr = sum(s)/len(s) if s else 0
    mt = sum(t)/len(t) if t else 0
    mp = sum(p)/len(p) if p else 0
    mr = sum(rc)/len(rc) if rc else 0
    ma = sum(ap)/len(ap) if ap else 0
    print(f"  {m:28} {len(s):5} {sr:.3f} ({sum(s):3}/{len(s):3})  {mt:6.0f}  {mp:.3f}    {mr:.3f}    {ma:.0f}")

# Split breakdown
splits = defaultdict(int)
for r in rows:
    splits[r.get("split", "?")] += 1
print(f"\nSplits: {dict(splits)}")
