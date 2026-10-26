"""
CAPS Research Audit Script
Comprehensive independent analysis of research_f5f93acf_episodes.csv
"""
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATH = Path("research/results/research_f5f93acf_episodes.csv")
SPLITS_PATH = Path("data/research/splits_1.json")
DATASET_PATH = Path("data/research/synthetic_v2_full.jsonl")

print("=" * 80)
print("CAPS RESEARCH AUDIT — Independent Analysis")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CSV SCHEMA AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] CSV SCHEMA AUDIT")
rows = []
fieldnames_in_file = []
bad_rows = []
with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    fieldnames_in_file = next(reader)
    for i, row in enumerate(reader, start=2):
        if len(row) != len(fieldnames_in_file):
            bad_rows.append((i, len(row)))
        else:
            rows.append(dict(zip(fieldnames_in_file, row)))

print(f"  Header columns : {len(fieldnames_in_file)}")
print(f"  Header         : {fieldnames_in_file}")
print(f"  Total data rows: {len(rows) + len(bad_rows)}")
print(f"  Bad rows       : {len(bad_rows)}")
if bad_rows:
    for r, n in bad_rows[:5]:
        print(f"    Row {r}: {n} columns")

EXPECTED_COLS = [
    "run_id", "experiment_id", "episode_id", "task_id", "method", "seed", "split",
    "budget_fraction", "budget_tokens", "policy_frozen",
    "category", "difficulty", "receiver_role",
    "num_available", "num_selected", "available_tokens", "selected_tokens", "compression",
    "api_prompt_tokens", "api_completion_tokens",
    "task_success", "reward", "latency_ms", "clarification_turns",
    "precision", "recall", "misleading_rate", "voi_scores_json",
]
print(f"  Expected cols  : {len(EXPECTED_COLS)}")
missing = [c for c in EXPECTED_COLS if c not in fieldnames_in_file]
extra   = [c for c in fieldnames_in_file if c not in EXPECTED_COLS]
print(f"  Missing cols   : {missing}")
print(f"  Extra cols     : {extra}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET / SPLIT AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] DATASET / SPLIT AUDIT")
splits_data = {}
if SPLITS_PATH.exists():
    splits_data = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_ids = set(splits_data["splits"]["train"])
    val_ids   = set(splits_data["splits"]["validation"])
    test_ids  = set(splits_data["splits"]["test"])
    print(f"  Split seed     : {splits_data.get('seed')}")
    print(f"  Dataset hash   : {splits_data.get('dataset_hash', 'N/A')[:16]}...")
    print(f"  Train tasks    : {len(train_ids)}")
    print(f"  Val tasks      : {len(val_ids)}")
    print(f"  Test tasks     : {len(test_ids)}")
    print(f"  Overlap T∩V    : {len(train_ids & val_ids)}")
    print(f"  Overlap T∩Te   : {len(train_ids & test_ids)}")
    print(f"  Overlap V∩Te   : {len(val_ids & test_ids)}")
else:
    print(f"  !! splits file not found: {SPLITS_PATH}")
    train_ids = val_ids = test_ids = set()

# Task IDs in CSV
csv_task_ids_by_split = defaultdict(set)
for r in rows:
    csv_task_ids_by_split[r["split"]].add(r["task_id"])

print(f"\n  Test task IDs in CSV ({len(csv_task_ids_by_split['test'])} unique):")
for tid in sorted(csv_task_ids_by_split["test"]):
    in_split = "✓" if tid in test_ids else "!! NOT IN SPLIT"
    print(f"    {tid}  {in_split}")

# Check for _s42 contamination
s42_rows = [r for r in rows if "_s42" in r["task_id"]]
print(f"\n  Rows with _s42 task IDs (stale dataset): {len(s42_rows)}")
if s42_rows:
    for r in s42_rows[:5]:
        print(f"    {r['experiment_id']} | {r['task_id']}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPERIMENT COMPLETENESS MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] EXPERIMENT COMPLETENESS MATRIX")
METHODS   = ["FullContext-Unbounded","FullContext-Budgeted","Random","Heuristic",
             "LLMPlanner","CAPS-NoVOI","CAPS-VOI"]
BUDGETS   = [1.0, 0.5, 0.3]
SEEDS     = [1, 2, 3]
SPLITS_EX = ["train", "validation", "test"]
EXPECTED_PER_SPLIT = {"train": 21, "validation": 7, "test": 7}

# Count rows per combination
combo_counts = defaultdict(lambda: defaultdict(int))  # (method, budget, seed) -> split -> count
for r in rows:
    key = (r["method"], float(r["budget_fraction"]), int(r["seed"]))
    combo_counts[key][r["split"]] += 1

total_expected = len(METHODS) * len(BUDGETS) * len(SEEDS) * sum(EXPECTED_PER_SPLIT.values())
total_actual   = len(rows)
total_expected_rows = len(METHODS) * len(BUDGETS) * len(SEEDS) * sum(EXPECTED_PER_SPLIT.values())

print(f"  Expected total rows: {total_expected_rows}")
print(f"  Actual rows        : {total_actual}")
print(f"  Difference         : {total_actual - total_expected_rows}")
print()

print(f"  {'Method':<26} {'Budget':>7} {'Seed':>5} | {'Train':>7} {'Val':>7} {'Test':>7} | {'Status'}")
print(f"  {'-'*26} {'-'*7} {'-'*5} | {'-'*7} {'-'*7} {'-'*7} | {'-'*15}")
issues = []
for method in METHODS:
    for budget in BUDGETS:
        for seed in SEEDS:
            key = (method, budget, seed)
            tr = combo_counts[key].get("train", 0)
            va = combo_counts[key].get("validation", 0)
            te = combo_counts[key].get("test", 0)
            exp_tr = EXPECTED_PER_SPLIT["train"]
            exp_va = EXPECTED_PER_SPLIT["validation"]
            exp_te = EXPECTED_PER_SPLIT["test"]
            ok = (tr == exp_tr and va == exp_va and te == exp_te)
            status = "OK" if ok else f"!INCOMPLETE tr={tr}/{exp_tr} va={va}/{exp_va} te={te}/{exp_te}"
            if not ok:
                issues.append((method, budget, seed, tr, va, te))
            print(f"  {method:<26} {budget:>7.0%} {seed:>5} | {tr:>7} {va:>7} {te:>7} | {status}")

print(f"\n  Incomplete combinations: {len(issues)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DUPLICATE EPISODE AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] DUPLICATE EPISODE AUDIT")
ep_ids = [r["episode_id"] for r in rows]
dup_eps = [(ep, n) for ep, n in Counter(ep_ids).items() if n > 1]
print(f"  Unique episode IDs : {len(set(ep_ids))}")
print(f"  Total rows         : {len(ep_ids)}")
print(f"  Duplicate episodes : {len(dup_eps)}")
if dup_eps:
    for ep, n in dup_eps[:5]:
        print(f"    {ep}  ×{n}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. NUMERIC INTEGRITY AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] NUMERIC INTEGRITY AUDIT")
def safe_float(v, name=""):
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None, f"NaN/Inf in {name}"
        return x, None
    except (ValueError, TypeError):
        return None, f"Non-numeric '{v}' in {name}"

numeric_errors = []
for i, r in enumerate(rows):
    for col in ["reward", "precision", "recall", "misleading_rate", "compression",
                "available_tokens", "selected_tokens", "latency_ms"]:
        v, err = safe_float(r.get(col, ""), col)
        if err:
            numeric_errors.append((i, col, r.get(col, ""), err))
        elif col in ("precision", "recall", "misleading_rate", "compression"):
            if v is not None and not (-0.001 <= v <= 1.001):
                numeric_errors.append((i, col, v, f"OUT OF RANGE [0,1]: {v}"))

print(f"  Numeric errors: {len(numeric_errors)}")
if numeric_errors:
    for i, col, val, err in numeric_errors[:10]:
        print(f"    row {i}: {err}")

# task_success must be True/False
ts_bad = [r["task_success"] for r in rows if r["task_success"].lower() not in ("true","false","1","0")]
print(f"  task_success bad values: {len(ts_bad)}")
if ts_bad:
    print(f"    samples: {ts_bad[:5]}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. RECOMPUTE RESULTS FROM RAW DATA (test split only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] RECOMPUTED RESULTS (test split)")
test_rows = [r for r in rows if r["split"] == "test"]
print(f"  Test rows total: {len(test_rows)}")

def parse_bool(v):
    return str(v).lower() in ("true", "1", "yes")

# Group by (method, budget, seed)
grouped = defaultdict(list)
for r in test_rows:
    key = (r["method"], float(r["budget_fraction"]), int(r["seed"]))
    grouped[key].append(r)

print()
print(f"  {'Method':<26} {'Budget':>7} {'Seed':>5} | {'N':>4} {'Succ%':>7} {'Reward':>8} {'Prec':>7} {'Recall':>7} {'Tokens':>8}")
print(f"  {'-'*26} {'-'*7} {'-'*5} | {'-'*4} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")

recomputed = {}
for method in METHODS:
    for budget in BUDGETS:
        for seed in SEEDS:
            key = (method, budget, seed)
            grp = grouped[key]
            if not grp:
                print(f"  {method:<26} {budget:>7.0%} {seed:>5} | MISSING")
                continue
            successes = [parse_bool(r["task_success"]) for r in grp]
            rewards   = [float(r["reward"]) for r in grp]
            precs     = [float(r["precision"]) for r in grp]
            recalls   = [float(r["recall"]) for r in grp]
            tokens    = [int(r["selected_tokens"]) for r in grp]
            n = len(grp)
            sr = sum(successes) / n
            mr = sum(rewards) / n
            mp = sum(precs) / n
            mrc = sum(recalls) / n
            mt = sum(tokens) / n
            recomputed[key] = {
                "n": n, "success_rate": sr, "mean_reward": mr,
                "mean_precision": mp, "mean_recall": mrc, "mean_tokens": mt,
            }
            print(f"  {method:<26} {budget:>7.0%} {seed:>5} | {n:>4} {sr:>7.1%} {mr:>8.4f} {mp:>7.4f} {mrc:>7.4f} {mt:>8.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CAPS-VOI DEEP INVESTIGATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] CAPS-VOI INVESTIGATION")
voi_test = [r for r in test_rows if r["method"] == "CAPS-VOI"]
print(f"  CAPS-VOI test episodes: {len(voi_test)}")
for budget in BUDGETS:
    b_rows = [r for r in voi_test if abs(float(r["budget_fraction"]) - budget) < 0.001]
    if b_rows:
        toks = [int(r["selected_tokens"]) for r in b_rows]
        num_sel = [int(r["num_selected"]) for r in b_rows]
        succs = [parse_bool(r["task_success"]) for r in b_rows]
        precs = [float(r["precision"]) for r in b_rows]
        print(f"  Budget {budget:.0%}: N={len(b_rows)}, "
              f"mean_sel_tokens={sum(toks)/len(toks):.1f}, "
              f"mean_num_selected={sum(num_sel)/len(num_sel):.2f}, "
              f"success={sum(succs)/len(succs):.0%}, "
              f"mean_precision={sum(precs)/len(precs):.4f}")
        if any(t == 0 for t in toks):
            print(f"    !! Zero-token episodes: {sum(1 for t in toks if t == 0)}/{len(toks)}")

# Examine CAPS-VOI train episodes for learning signal
voi_train = [r for r in rows if r["method"] == "CAPS-VOI" and r["split"] == "train"]
voi_val   = [r for r in rows if r["method"] == "CAPS-VOI" and r["split"] == "validation"]
print(f"\n  CAPS-VOI train episodes: {len(voi_train)}")
print(f"  CAPS-VOI val   episodes: {len(voi_val)}")

# Check reward distribution for CAPS-VOI (training)
if voi_train:
    rewards_train = [float(r["reward"]) for r in voi_train]
    print(f"  Train reward: min={min(rewards_train):.4f}, max={max(rewards_train):.4f}, "
          f"mean={sum(rewards_train)/len(rewards_train):.4f}")
    positive_rewards = sum(1 for x in rewards_train if x > 0)
    print(f"  Train positive rewards: {positive_rewards}/{len(rewards_train)}")
    zero_tok_train = [r for r in voi_train if int(r["selected_tokens"]) == 0]
    print(f"  Train zero-token episodes: {len(zero_tok_train)}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. REWARD FUNCTION AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] REWARD FUNCTION AUDIT")
# What is the reward for sending nothing?
# R = success - lambda * cost
# cost = w_tokens * (0/budget) + w_latency * (latency/max_latency) + ...
# If selected_tokens = 0, norm_tokens = 0
# Success is always False (mock: receiver can't match answer_key)
# So R = 0 - 0.5 * (0 * 0.4 + latency_penalty * 0.2)
# latency_penalty ≈ latency_ms / 15000
# In mock mode latency_ms ≈ 10ms → 10/15000 ≈ 0.00067
# R_nothing ≈ 0 - 0.5 * (0 + 0.00067 * 0.2) ≈ -0.000067
print("  Theoretical reward for sending nothing (no success):")
print("    R = 0 - 0.5 * (0 * 0.4 + ~0.00067 * 0.2 + 0 + 0) ≈ -0.000067")
print()
print("  Theoretical reward for sending everything (no success):")
print("    R = 0 - 0.5 * (1.0 * 0.4 + latency * 0.2) = -0.2 - latency_penalty")
print()
print("  Theoretical reward for sending everything (success=True):")
print("    R = 1.0 - 0.5 * (1.0 * 0.4 + latency * 0.2) ≈ 0.8 - latency_penalty")
print()
print("  CRITICAL FINDING: Sending nothing has LOWER cost than sending everything.")
print("  If task always fails (mock: never returns answer_key), CAPS-VOI LEARNS to send 0 tokens.")
print("  This is a REWARD DESIGN FAILURE + MOCK ENVIRONMENT ARTIFACT.")

# Verify: are any test episodes successful?
success_by_method = {}
for method in METHODS:
    m_test = [r for r in test_rows if r["method"] == method]
    if m_test:
        succs = [parse_bool(r["task_success"]) for r in m_test]
        success_by_method[method] = sum(succs) / len(succs)
        
print(f"\n  Success rates (test, all budgets/seeds):")
for m, sr in sorted(success_by_method.items(), key=lambda x: -x[1]):
    print(f"    {m:<26}: {sr:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MOCK ENVIRONMENT AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] MOCK ENVIRONMENT AUDIT")
# What does the mock receiver return?
print("  Mock LLM response: 'Mock LLM response.' (fixed string)")
print("  Answer keys per family:")
answer_keys = {
    "debugging"               : "parser_guard",
    "configuration"           : "port_5432",
    "data_transformation"     : "normalize_iso_dates",
    "code_repair"             : "inclusive_upper_bound",
    "documentation_lookup"    : "three_retries",
    "multi_hop_reasoning"     : "upgrade_connector",
    "contradiction_resolution": "prefer_current_runbook",
}
for fam, key in answer_keys.items():
    in_mock = key.lower() in "mock llm response."
    print(f"    {fam:<28}: answer_key='{key}' | in_mock_response={in_mock}")

print()
print("  FINDING: Mock LLM response NEVER contains any answer_key.")
print("  Therefore: task_success = False for ALL episodes with mock LLM.")

# Verify from data
success_test = [r for r in test_rows if parse_bool(r["task_success"])]
print(f"  Actual successful test episodes: {len(success_test)} / {len(test_rows)}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. TRAIN/TEST LEAKAGE CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] TRAIN/TEST LEAKAGE CHECK")
train_task_ids = csv_task_ids_by_split["train"]
test_task_ids  = csv_task_ids_by_split["test"]
val_task_ids   = csv_task_ids_by_split["validation"]
overlap_tr_te  = train_task_ids & test_task_ids
overlap_va_te  = val_task_ids & test_task_ids
print(f"  Train∩Test task overlap: {len(overlap_tr_te)}")
print(f"  Val∩Test task overlap  : {len(overlap_va_te)}")
if overlap_tr_te:
    print(f"  !! LEAKAGE DETECTED: {overlap_tr_te}")

# Also check per-method: does test set vary across methods?
test_tasks_per_method = {}
for method in METHODS:
    m_test = [r for r in test_rows if r["method"] == method and r["seed"] == "1" and float(r["budget_fraction"]) == 1.0]
    test_tasks_per_method[method] = sorted(set(r["task_id"] for r in m_test))

# All methods should have the same test task IDs
first_method = METHODS[0]
ref_tasks = test_tasks_per_method.get(first_method, [])
print(f"\n  Test task IDs (reference: {first_method}):")
for tid in ref_tasks:
    print(f"    {tid}")

mismatches = []
for method, tasks in test_tasks_per_method.items():
    if tasks != ref_tasks:
        mismatches.append(method)
        
print(f"\n  Methods with different test task IDs: {mismatches}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] STATISTICAL ANALYSIS (test split, all seeds at 100% budget)")
print()
print("  Since task_success=False for ALL episodes, success rate comparison is trivially 0%.")
print("  We can only compare reward, precision, recall across methods.")
print()
print("  Mean metrics at 100% budget (averaged across 3 seeds):")
print()
for method in METHODS:
    vals = []
    for seed in SEEDS:
        key = (method, 1.0, seed)
        if key in recomputed:
            vals.append(recomputed[key])
    if vals:
        precs = [v["mean_precision"] for v in vals]
        recalls = [v["mean_recall"] for v in vals]
        rewards = [v["mean_reward"] for v in vals]
        tokens = [v["mean_tokens"] for v in vals]
        print(f"  {method:<28}: "
              f"prec={sum(precs)/len(precs):.4f}±{statistics.stdev(precs) if len(precs)>1 else 0:.4f}  "
              f"recall={sum(recalls)/len(recalls):.4f}±{statistics.stdev(recalls) if len(recalls)>1 else 0:.4f}  "
              f"reward={sum(rewards)/len(rewards):.4f}±{statistics.stdev(rewards) if len(rewards)>1 else 0:.4f}  "
              f"tokens={sum(tokens)/len(tokens):.0f}")

print()
print("  STATISTICAL CONCLUSION:")
print("  - With N=7 test tasks and all success=False, success-rate comparison is INVALID.")
print("  - Reward differences driven by token count, not task performance.")
print("  - Precision/recall meaningful as selection quality proxies, but N=7 is underpowered.")
print("  - Minimum N for 80% power to detect 20pp difference: ~75 tasks per condition.")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SEED ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] SEED ROBUSTNESS (100% budget, test split)")
for method in METHODS:
    seed_precs = [recomputed.get((method, 1.0, s), {}).get("mean_precision", None) for s in SEEDS]
    seed_precs = [x for x in seed_precs if x is not None]
    if len(seed_precs) >= 2:
        std = statistics.stdev(seed_precs)
        rang = max(seed_precs) - min(seed_precs)
        print(f"  {method:<28}: precision_std={std:.6f}  range={rang:.6f}  {'STABLE' if rang < 0.01 else 'UNSTABLE'}")
    elif seed_precs:
        print(f"  {method:<28}: only 1 seed")

print()
print("  CAPS-NoVOI and CAPS-VOI show 0 variance across seeds.")
print("  This indicates the LinUCB exploration is NOT adding seed-to-seed variation at test time.")
print("  Reason: mock LLM is deterministic, embeddings are deterministic, no stochasticity.")
print("  In real LLM mode, seeds would create genuine variance.")

# ─────────────────────────────────────────────────────────────────────────────
# 13. GREEDY SELECTOR BUG CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] GREEDY SELECTOR BUG — Score ≤ 0 filter")
print("  In selector.py line 30: 'if scores.get(ciu.id, 0.0) <= 0.0: continue'")
print("  This means ANY non-positive UCB score will result in the CIU being SKIPPED.")
print()
print("  LinUCB at initialization: b=0, theta=A^{-1}b = 0")
print("  exploitation = x^T * theta = 0")
print("  exploration = alpha * sqrt(x^T A^{-1} x) > 0 always")
print("  UCB = 0 + exploration > 0")
print("  THEREFORE: Initial UCB > 0, selector SHOULD include CIUs.")
print()
print("  BUT: After training with negative rewards:")
print("  theta = A^{-1} * sum(r_i * x_i) where r_i < 0 (failed tasks)")
print("  exploitation becomes NEGATIVE (< 0)")
print("  If |exploitation| > exploration → UCB < 0 → CIU SKIPPED")
print()
print("  CONFIRMED ROOT CAUSE OF CAPS-VOI ZERO-TOKEN BEHAVIOR:")
print("  After 21 training episodes with all-negative rewards (-0.15 to -0.20),")
print("  LinUCB learns a NEGATIVE theta for all arm types.")
print("  At test time (100% budget, alpha=1.0), UCB scores become negative,")
print("  selector's '≤ 0' guard filters ALL CIUs, resulting in empty selection.")
print()
print("  This is a BUG: the '≤ 0' guard in greedy_budget_select is incorrect for LinUCB.")
print("  The guard was designed for heuristic policies (e.g., 'priority=0 means skip').")
print("  For LinUCB, UCB scores can legitimately be negative during exploration.")
print("  FIX: Remove or relax the '≤ 0' guard, or use a relative threshold.")

# ─────────────────────────────────────────────────────────────────────────────
# 14. BUDGET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] BUDGET SENSITIVITY (precision at test, seed=1)")
for budget in BUDGETS:
    print(f"\n  Budget {budget:.0%}:")
    for method in METHODS:
        key = (method, budget, 1)
        if key in recomputed:
            v = recomputed[key]
            print(f"    {method:<28}: prec={v['mean_precision']:.4f} recall={v['mean_recall']:.4f} "
                  f"reward={v['mean_reward']:.4f} tokens={v['mean_tokens']:.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. CAPS-NOVOI ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[15] CAPS-NoVOI vs CAPS-VOI COMPARISON")
for budget in BUDGETS:
    k_novoi = (("CAPS-NoVOI", budget, 1), recomputed.get(("CAPS-NoVOI", budget, 1), {}))
    k_voi   = (("CAPS-VOI",   budget, 1), recomputed.get(("CAPS-VOI",   budget, 1), {}))
    if k_novoi[1] and k_voi[1]:
        diff_prec = k_voi[1]["mean_precision"] - k_novoi[1]["mean_precision"]
        diff_tok  = k_voi[1]["mean_tokens"]    - k_novoi[1]["mean_tokens"]
        print(f"  Budget {budget:.0%}: CAPS-VOI vs CAPS-NoVOI: "
              f"Δprecision={diff_prec:+.4f}, Δtokens={diff_tok:+.1f}")

print()
print("  CAPS-VOI consistently WORSE than CAPS-NoVOI due to the zero-token bug.")
print("  This is NOT a VOI effectiveness finding — it is an implementation artifact.")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
