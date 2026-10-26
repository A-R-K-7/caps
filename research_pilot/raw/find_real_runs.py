import sys, csv, os
sys.path.insert(0, "src")

# Check ALL episode CSVs for real-LLM indicators
for d in ["research/results", "research_pilot"]:
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith("_episodes.csv"):
            continue
        path = os.path.join(d, f)
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        
        # Real LLM indicators: api_prompt_tokens > 0, latency > 1000ms
        has_api = sum(1 for r in rows if r.get("api_prompt_tokens", "0") not in ("0", "", "NONE", None))
        high_latency = sum(1 for r in rows if float(r.get("latency_ms", 0)) > 1000)
        any_success = sum(1 for r in rows if r.get("task_success") == "True")
        
        unique_methods = set(r["method"] for r in rows)
        unique_splits = set(r.get("split", "?") for r in rows)
        
        is_real = has_api > 0 or high_latency > len(rows) * 0.5
        
        marker = "*** REAL LLM ***" if is_real else "mock"
        print(f"\n{marker}: {f}")
        print(f"  Episodes: {len(rows)}, Methods: {unique_methods}, Splits: {unique_splits}")
        print(f"  API tokens>0: {has_api}, High latency: {high_latency}, Success: {any_success}/{len(rows)}")
        if is_real:
            # Show method breakdown
            from collections import defaultdict
            meth_stats = defaultdict(lambda: {"n":0, "ok":0, "api":0})
            for r in rows:
                m = r["method"]
                meth_stats[m]["n"] += 1
                meth_stats[m]["ok"] += r.get("task_success") == "True"
                try: meth_stats[m]["api"] += int(r.get("api_prompt_tokens", 0))
                except: pass
            for m, s in sorted(meth_stats.items()):
                print(f"    {m:30} n={s['n']:3} success={s['ok']:3} api_tokens={s['api']}")
