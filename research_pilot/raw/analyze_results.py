import sys, csv, json
from collections import defaultdict
sys.path.insert(0, "src")

for run_id in ["research_f5f93acf", "research_cda97b59", "research_8179e54a"]:
    path = f"research/results/{run_id}_episodes.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        continue
    
    cols = list(rows[0].keys())
    
    # Check for API tokens
    has_api = sum(1 for r in rows if r.get("api_prompt_tokens", "0") not in ("0", "", "NONE"))
    any_success = sum(1 for r in rows if r.get("task_success") == "True")
    
    print(f"\n{'='*60}")
    print(f"RUN: {run_id}")
    print(f"Episodes: {len(rows)}")
    print(f"Columns: {len(cols)}")
    print(f"Episodes with API tokens > 0: {has_api}")
    print(f"Episodes with task_success=True: {any_success} ({any_success/len(rows)*100:.1f}%)")
    
    # Test results by method x budget
    test_rows = [r for r in rows if r.get("split") == "test"]
    print(f"Test episodes: {len(test_rows)}")
    
    method_budget_success = defaultdict(lambda: defaultdict(list))
    method_budget_tokens = defaultdict(lambda: defaultdict(list))
    method_budget_precision = defaultdict(lambda: defaultdict(list))
    method_budget_recall = defaultdict(lambda: defaultdict(list))
    
    for r in test_rows:
        m = r["method"]
        b = r.get("budget_fraction", "?")
        method_budget_success[m][b].append(r.get("task_success") == "True")
        try:
            method_budget_tokens[m][b].append(int(r.get("selected_tokens", 0)))
        except:
            pass
        try:
            method_budget_precision[m][b].append(float(r.get("precision", 0)))
        except:
            pass
        try:
            method_budget_recall[m][b].append(float(r.get("recall", 0)))
        except:
            pass
    
    print(f"\nTest Results by Method x Budget:")
    print(f"{'Method':30} {'Budget':8} {'Success':10} {'Tokens':10} {'Precision':10} {'Recall':10}")
    for m in sorted(method_budget_success.keys()):
        for b in sorted(method_budget_success[m].keys()):
            s = method_budget_success[m][b]
            t = method_budget_tokens[m][b]
            p = method_budget_precision[m][b]
            rc = method_budget_recall[m][b]
            sr = sum(s)/len(s) if s else 0
            mt = sum(t)/len(t) if t else 0
            mp = sum(p)/len(p) if p else 0
            mr = sum(rc)/len(rc) if rc else 0
            print(f"  {m:28} {b:8} {sr:.3f} ({sum(s):2}/{len(s):2}) {mt:8.0f}   {mp:.3f}      {mr:.3f}")

    # Train results for learning curve
    train_rows = [r for r in rows if r.get("split") == "train"]
    if train_rows:
        train_success = sum(1 for r in train_rows if r.get("task_success") == "True")
        print(f"\nTrain: {train_success}/{len(train_rows)} success ({train_success/len(train_rows)*100:.1f}%)")
    
    # Check for real LLM indicators
    latencies = [float(r.get("latency_ms", 0)) for r in rows[:10]]
    print(f"First 10 latencies: {[f'{l:.0f}' for l in latencies]}")

print("\n\nAlso checking for newer runs...")
import os
for f in sorted(os.listdir("research/results")):
    if f.endswith("_episodes.csv"):
        size = os.path.getsize(f"research/results/{f}")
        print(f"  {f}: {size:,} bytes")
