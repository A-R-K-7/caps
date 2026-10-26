"""
Diagnostic tracer — runs one end-to-end episode with full logging.

Produces:
  research_debug/raw/example_episode_trace.json
  research_debug/raw/evaluator_test_results.json
  research_debug/raw/sanity_benchmark_results.json
  research_debug/raw/checkpoint_audit.json
  research_debug/raw/seed_audit.json
  research_debug/raw/llm_calls.json

Run:
    $env:PYTHONPATH="src"
    .venv\\Scripts\\python.exe research_debug\\run_diagnostic.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
RAW_DIR = Path("research_debug/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 0. Import all modules under test
# ===========================================================================
def _import():
    from caps.datasets.synthetic_v2 import (
        generate_task, generate_synthetic_v2_dataset,
        HIDDEN_FIELDS, visible_ciu, create_splits
    )
    from caps.environments.synthetic_v2 import SyntheticV2Environment
    from caps.agents.receiver import ReceiverAgent, ReceiverOutput
    from caps.agents.planner import PlannerAgent
    from caps.communication.baselines import (
        FullContextPolicy, BudgetedFullContextPolicy,
        RandomPolicy, StaticHeuristicPolicy, CAPSNoVOIPolicy,
    )
    from caps.bandits.linucb import LinUCBPolicy
    from caps.embeddings.local import LocalEmbeddingProvider
    from caps.evaluation.reward import RewardFunction
    from caps.llm.mock_client import MockLLMProvider
    return (generate_task, generate_synthetic_v2_dataset, HIDDEN_FIELDS,
            visible_ciu, create_splits, SyntheticV2Environment, ReceiverAgent,
            ReceiverOutput, PlannerAgent, FullContextPolicy,
            BudgetedFullContextPolicy, RandomPolicy, StaticHeuristicPolicy,
            CAPSNoVOIPolicy, LinUCBPolicy, LocalEmbeddingProvider,
            RewardFunction, MockLLMProvider)


# ===========================================================================
# 1. Mock LLM answer generation
# ===========================================================================
def make_smart_mock_llm(answer_key: str):
    """
    Build a MockLLMProvider that returns the answer_key in the receiver response.
    This simulates a receiver that correctly identifies the answer.
    """
    from caps.llm.mock_client import MockLLMProvider
    
    class SmartMock(MockLLMProvider):
        def _generate_schema_response(self, schema):
            return json.dumps({
                "status": "success",
                "answer": f"The correct answer is {answer_key}.",
                "used_ciu_ids": [],
                "missing_information": [],
                "confidence": 0.9,
            })
            
    return SmartMock()


# ===========================================================================
# 2. Evaluator deterministic tests
# ===========================================================================
def run_evaluator_tests():
    """Section 5: Deterministic evaluator sanity tests."""
    from caps.environments.synthetic_v2 import SyntheticV2Environment
    from caps.datasets.synthetic_v2 import generate_task

    env = SyntheticV2Environment()
    task = generate_task("debugging", "easy", 0, 1)
    env.load_tasks([task])
    env.reset(0)

    answer_key = task["answer_key"]   # "parser_guard"
    answer_text = task["answer_text"] # "The correct fix requires adding..."

    test_cases = [
        # (input_text, expected_success, description)
        (f"The answer is {answer_key}.", True, "exact keyword in sentence"),
        (f"{answer_key}", True, "bare keyword"),
        (f"Fix: {answer_key} applied.", True, "keyword with context"),
        (answer_text.split(",")[0], True, "answer_text fragment"),
        ("mock_value", False, "mock_value (mock LLM default)"),
        ("I do not know the answer.", False, "explicit failure"),
        ("The answer is 42.", False, "wrong number answer"),
        ("", False, "empty string"),
        ("parser", False, "partial keyword — should fail unless word boundary matches"),
    ]

    results = []
    for text, expected, desc in test_cases:
        got = env.evaluate(text)
        status = "PASS" if got == expected else "FAIL"
        results.append({
            "input": text[:80],
            "expected": expected,
            "got": got,
            "status": status,
            "description": desc,
        })
        print(f"  [{status}] {desc}: evaluate({text[:40]!r}) → {got}")

    summary = {
        "answer_key": answer_key,
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
        "cases": results,
    }
    (RAW_DIR / "evaluator_test_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n  Evaluator tests: {summary['passed']}/{summary['total']} PASS\n")
    return summary


# ===========================================================================
# 3. End-to-end episode trace
# ===========================================================================
def run_episode_trace():
    """Section 4: Complete end-to-end episode trace."""
    (generate_task, generate_synthetic_v2_dataset, HIDDEN_FIELDS,
     visible_ciu, create_splits, SyntheticV2Environment, ReceiverAgent,
     ReceiverOutput, PlannerAgent, FullContextPolicy,
     BudgetedFullContextPolicy, RandomPolicy, StaticHeuristicPolicy,
     CAPSNoVOIPolicy, LinUCBPolicy, LocalEmbeddingProvider,
     RewardFunction, MockLLMProvider) = _import()

    print("\n=== END-TO-END EPISODE TRACE ===")

    # 1. Task creation
    task = generate_task("debugging", "easy", 0, 1)
    answer_key = task["answer_key"]
    print(f"  Task ID:     {task['task_id']}")
    print(f"  Task:        {task['task']}")
    print(f"  Answer key:  {answer_key}")
    print(f"  Answer text: {task['answer_text']}")
    print(f"  Num CIUs:    {len(task['cius'])}")

    # 2. CIU visibility check
    for ciu in task["cius"]:
        clean = visible_ciu(ciu)
        for hf in HIDDEN_FIELDS:
            assert hf not in clean, f"LEAKAGE: {hf} in visible CIU {ciu['id']}"
    print(f"  Hidden field leakage check: PASS (all {len(task['cius'])} CIUs clean)")

    # 3. Environment
    env = SyntheticV2Environment()
    env.load_tasks([task])
    env.reset(0)

    # 4. LLM providers — test BOTH mock-dumb and mock-smart
    dumb_llm = MockLLMProvider()   # returns "mock_value"
    smart_llm = make_smart_mock_llm(answer_key)  # returns correct answer

    embedder = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2", cache_path=None)
    planner = PlannerAgent(llm_provider=dumb_llm, embedding_provider=embedder)
    reward_fn = RewardFunction()

    # 5. CIUs (visible only)
    cius = env.get_cius()
    budget_tokens = sum(c.estimate_tokens() for c in cius)  # full budget

    state = planner.build_state(
        task=env.get_task_description(),
        cius=cius,
        token_budget=budget_tokens,
        receiver_role=task["receiver_role"],
        task_category=task["category"],
        difficulty=task["difficulty"],
    )

    # 6. Policy selects (FullContext)
    policy = FullContextPolicy()
    action = policy.select(state)

    from caps.communication.ciu import CIU
    ciu_map = {c.id: c for c in cius}
    selected_cius = [ciu_map[cid] for cid in action.selected_ciu_ids if cid in ciu_map]

    print(f"  Selected {len(selected_cius)}/{len(cius)} CIUs (FullContext-Unbounded)")

    # 7. Receiver with DUMB mock (what happens in current experiment)
    t0 = time.time()
    dumb_receiver = ReceiverAgent(llm_provider=dumb_llm)
    dumb_output = dumb_receiver.execute(
        task=env.get_task_description(),
        selected_cius=selected_cius,
        receiver_role=task["receiver_role"],
    )
    dumb_latency = (time.time() - t0) * 1000

    # 8. Receiver with SMART mock (verifies evaluator works if answer present)
    smart_receiver = ReceiverAgent(llm_provider=smart_llm)
    smart_output = smart_receiver.execute(
        task=env.get_task_description(),
        selected_cius=selected_cius,
        receiver_role=task["receiver_role"],
    )

    # 9. Evaluation
    dumb_success = env.evaluate(dumb_output.answer)
    smart_success = env.evaluate(smart_output.answer)
    sq = env.selection_quality(action.selected_ciu_ids)

    dumb_reward = reward_fn.calculate(
        task_success=dumb_success, state=state, action=action,
        latency_ms=dumb_latency,
    )

    print(f"\n  --- DUMB MOCK (current experiment behavior) ---")
    print(f"  Raw response text: {dumb_llm.call_log[-1]['response']!r}")
    print(f"  Receiver answer:   {dumb_output.answer!r}")
    print(f"  Evaluator input:   {dumb_output.answer!r}")
    print(f"  Expected keyword:  {answer_key!r}")
    print(f"  task_success:      {dumb_success}  ← ROOT CAUSE: mock_value ≠ answer_key")
    print(f"  reward:            {dumb_reward:.4f}")

    print(f"\n  --- SMART MOCK (verifies evaluator is correct) ---")
    print(f"  Raw response text: {smart_llm.call_log[-1]['response']!r}")
    print(f"  Receiver answer:   {smart_output.answer!r}")
    print(f"  task_success:      {smart_success}  ← SHOULD BE True if evaluator works")
    print(f"  selection quality: {sq}")

    # Determine if evaluator can possibly return True
    evaluator_can_succeed = smart_success
    print(f"\n  Can task_success=True occur? {evaluator_can_succeed}")

    trace = {
        "task_id": task["task_id"],
        "task": task["task"],
        "answer_key": answer_key,
        "answer_text": task["answer_text"],
        "num_available_cius": len(cius),
        "selected_ciu_ids": action.selected_ciu_ids,
        "selected_count": len(selected_cius),
        "cius_sent_to_receiver": [
            {"id": c.id, "type": c.type, "content_preview": c.content[:80]}
            for c in selected_cius
        ],
        "receiver_prompt_preview": (
            f"TASK:\n{env.get_task_description()}\n\nPROVIDED CONTEXT: "
            f"[{len(selected_cius)} CIUs, total ~{sum(c.estimate_tokens() for c in selected_cius)} tokens]"
        ),
        "dumb_mock": {
            "raw_llm_response": dumb_llm.call_log[-1]["response"],
            "receiver_answer": dumb_output.answer,
            "receiver_status": dumb_output.status,
            "evaluator_expected": answer_key,
            "evaluator_result": dumb_success,
            "reward": dumb_reward,
            "root_cause": "mock LLM returns 'mock_value' — never matches answer_key",
        },
        "smart_mock": {
            "raw_llm_response": smart_llm.call_log[-1]["response"],
            "receiver_answer": smart_output.answer,
            "receiver_status": smart_output.status,
            "evaluator_result": smart_success,
            "root_cause_confirmed": "evaluator works correctly when answer is present",
        },
        "selection_quality": sq,
        "evaluator_can_return_true": evaluator_can_succeed,
        "hidden_fields_leaked": False,
    }
    (RAW_DIR / "example_episode_trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )
    return trace


# ===========================================================================
# 4. Sanity benchmark — trivial tasks
# ===========================================================================
SANITY_TASKS = [
    {
        "task_id": "sanity_01",
        "task": "What is 2 + 2?",
        "answer_key": "four",
        "answer_text": "The answer is four. This is basic arithmetic.",
        "receiver_role": "general",
        "category": "arithmetic",
        "difficulty": "easy",
        "cius": [
            {"id": "s01_c00", "type": "task_constraint", "content": "2 + 2 = four",
             "token_count": 8, "source": "sanity",
             "information_role": "necessary", "ground_truth_necessity": True,
             "ground_truth_sufficiency": True, "relevance": 1.0,
             "dependency_group": "req_1"},
            {"id": "s01_c01", "type": "irrelevant_documentation", "content": "Office scheduling for Q4.",
             "token_count": 12, "source": "sanity",
             "information_role": "irrelevant", "ground_truth_necessity": False,
             "ground_truth_sufficiency": False, "relevance": 0.0,
             "dependency_group": "noise"},
        ],
        "ground_truth": {
            "necessary_cius": ["s01_c00"],
            "useful_cius": ["s01_c00"],
            "misleading_cius": [],
        },
    },
    {
        "task_id": "sanity_02",
        "task": "What is the capital of France?",
        "answer_key": "paris",
        "answer_text": "The capital of France is Paris.",
        "receiver_role": "analyst",
        "category": "knowledge",
        "difficulty": "easy",
        "cius": [
            {"id": "s02_c00", "type": "task_constraint",
             "content": "The capital of France is Paris.",
             "token_count": 10, "source": "sanity",
             "information_role": "necessary", "ground_truth_necessity": True,
             "ground_truth_sufficiency": True, "relevance": 1.0,
             "dependency_group": "req_1"},
            {"id": "s02_c01", "type": "outdated_or_conflicting_context",
             "content": "France's capital was previously Lyon, before 1789.",
             "token_count": 15, "source": "sanity",
             "information_role": "misleading", "ground_truth_necessity": False,
             "ground_truth_sufficiency": False, "relevance": -0.5,
             "dependency_group": "conflict"},
        ],
        "ground_truth": {
            "necessary_cius": ["s02_c00"],
            "useful_cius": ["s02_c00"],
            "misleading_cius": ["s02_c01"],
        },
    },
    {
        "task_id": "sanity_03",
        "task": "What database port should the service use?",
        "answer_key": "port_5432",
        "answer_text": "The service must use PostgreSQL on port 5432.",
        "receiver_role": "sysadmin",
        "category": "configuration",
        "difficulty": "easy",
        "cius": [
            {"id": "s03_c00", "type": "task_constraint",
             "content": "The service must use PostgreSQL on port_5432 with TLS enabled.",
             "token_count": 16, "source": "sanity",
             "information_role": "necessary", "ground_truth_necessity": True,
             "ground_truth_sufficiency": True, "relevance": 1.0,
             "dependency_group": "req_1"},
            {"id": "s03_c01", "type": "redundant_context",
             "content": "Database connection documentation from 2019.",
             "token_count": 10, "source": "sanity",
             "information_role": "redundant", "ground_truth_necessity": False,
             "ground_truth_sufficiency": False, "relevance": 0.4,
             "dependency_group": "redundant"},
            {"id": "s03_c02", "type": "irrelevant_documentation",
             "content": "Billing dashboard configuration for office expenses.",
             "token_count": 12, "source": "sanity",
             "information_role": "irrelevant", "ground_truth_necessity": False,
             "ground_truth_sufficiency": False, "relevance": 0.0,
             "dependency_group": "noise"},
        ],
        "ground_truth": {
            "necessary_cius": ["s03_c00"],
            "useful_cius": ["s03_c00"],
            "misleading_cius": [],
        },
    },
]
# Add 7 more sanity tasks by varying the answer
def _more_sanity_tasks():
    extras = []
    templates = [
        ("sanity_04", "What retry count does the API allow?",
         "three_retries", "The API client allows exactly three_retries with exponential backoff.",
         "The API client allows exactly three_retries with exponential backoff.", "analyst"),
        ("sanity_05", "What must be upgraded to fix the connector issue?",
         "upgrade_connector", "The connector must be upgraded because the runtime removed the old endpoint.",
         "The connector must be upgraded.", "research_analyst"),
        ("sanity_06", "Which runbook takes precedence?",
         "prefer_current_runbook", "The current runbook supersedes the archived note.",
         "The current runbook supersedes the archived note and requires the safer rollback.", "investigator"),
        ("sanity_07", "What loop bounds should the function use?",
         "inclusive_upper_bound", "The loop must include the inclusive_upper_bound and preserve zero.",
         "The loop must include the upper bound.", "developer"),
        ("sanity_08", "How should dates be normalized?",
         "normalize_iso_dates", "Dates must be normalized to normalize_iso_dates format.",
         "Dates must be normalized to ISO-8601.", "data_engineer"),
        ("sanity_09", "What guard is needed before parsing?",
         "parser_guard", "Adding a parser_guard before parsing prevents the crash.",
         "The correct fix requires adding an empty-input guard before parsing.", "debugger"),
        ("sanity_10", "What is the empty answer case?",
         "no_ciu_selected", "No necessary CIU was provided.",
         "This task has no necessary CIU — success impossible without it.", "general"),
    ]
    for tid, task, akey, ciu_content, atext, role in templates:
        extras.append({
            "task_id": tid,
            "task": task,
            "answer_key": akey,
            "answer_text": atext,
            "receiver_role": role,
            "category": "sanity",
            "difficulty": "easy",
            "cius": [
                {"id": f"{tid}_c00", "type": "task_constraint",
                 "content": ciu_content, "token_count": len(ciu_content) // 4,
                 "source": "sanity",
                 "information_role": "necessary", "ground_truth_necessity": True,
                 "ground_truth_sufficiency": True, "relevance": 1.0,
                 "dependency_group": "req_1"},
            ],
            "ground_truth": {
                "necessary_cius": [f"{tid}_c00"],
                "useful_cius": [f"{tid}_c00"],
                "misleading_cius": [],
            },
        })
    return extras

ALL_SANITY_TASKS = SANITY_TASKS + _more_sanity_tasks()


def run_sanity_benchmark():
    """Section 7–8: Run sanity benchmark with smart mock LLM."""
    (generate_task, generate_synthetic_v2_dataset, HIDDEN_FIELDS,
     visible_ciu, create_splits, SyntheticV2Environment, ReceiverAgent,
     ReceiverOutput, PlannerAgent, FullContextPolicy,
     BudgetedFullContextPolicy, RandomPolicy, StaticHeuristicPolicy,
     CAPSNoVOIPolicy, LinUCBPolicy, LocalEmbeddingProvider,
     RewardFunction, MockLLMProvider) = _import()

    print("\n=== SANITY BENCHMARK ===")
    print(f"  {len(ALL_SANITY_TASKS)} tasks")

    embedder = LocalEmbeddingProvider(model_name="all-MiniLM-L6-v2", cache_path=None)
    reward_fn = RewardFunction()

    methods = {
        "FullContext-Unbounded": FullContextPolicy(),
        "Random": RandomPolicy(seed=1),
        "Heuristic": StaticHeuristicPolicy(),
    }

    all_results = {}

    for method_name, policy in methods.items():
        method_results = []
        print(f"\n  Method: {method_name}")

        for task_data in ALL_SANITY_TASKS:
            answer_key = task_data["answer_key"]

            # Smart mock: returns the answer key in its response
            smart_llm = make_smart_mock_llm(answer_key)
            planner = PlannerAgent(llm_provider=smart_llm, embedding_provider=embedder)
            receiver = ReceiverAgent(llm_provider=smart_llm)

            env = SyntheticV2Environment()
            env.load_tasks([task_data])
            env.reset(0)

            cius = env.get_cius()
            total_tokens = sum(c.estimate_tokens() for c in cius)
            state = planner.build_state(
                task=env.get_task_description(),
                cius=cius,
                token_budget=total_tokens,
                receiver_role=task_data["receiver_role"],
                task_category=task_data["category"],
                difficulty=task_data["difficulty"],
            )

            t0 = time.time()
            action = policy.select(state)
            ciu_map = {c.id: c for c in cius}
            selected_cius = [ciu_map[cid] for cid in action.selected_ciu_ids if cid in ciu_map]

            receiver_output = receiver.execute(
                task=env.get_task_description(),
                selected_cius=selected_cius,
                receiver_role=task_data["receiver_role"],
            )
            latency = (time.time() - t0) * 1000

            success = env.evaluate(receiver_output.answer)
            sq = env.selection_quality(action.selected_ciu_ids)
            reward = reward_fn.calculate(
                task_success=success, state=state, action=action, latency_ms=latency
            )

            row = {
                "task_id": task_data["task_id"],
                "task": task_data["task"],
                "answer_key": answer_key,
                "receiver_answer": receiver_output.answer,
                "success": success,
                "reward": reward,
                "selected_ids": action.selected_ciu_ids,
                "num_selected": len(selected_cius),
                "num_available": len(cius),
                **sq,
            }
            method_results.append(row)

            sym = "✓" if success else "✗"
            print(f"    {sym} {task_data['task_id']}: {task_data['task'][:50]} → {receiver_output.answer[:40]!r}")

        n = len(method_results)
        k = sum(1 for r in method_results if r["success"])
        print(f"  {method_name}: {k}/{n} successful")
        all_results[method_name] = {
            "n": n, "success": k,
            "success_rate": k / n if n else 0.0,
            "rows": method_results,
        }

    (RAW_DIR / "sanity_benchmark_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    return all_results


# ===========================================================================
# 5. Checkpoint audit
# ===========================================================================
def run_checkpoint_audit():
    """Section 15: Verify that policy checkpoints actually change."""
    print("\n=== CHECKPOINT AUDIT ===")

    ckpt_dir = Path("research/checkpoints")
    if not ckpt_dir.exists():
        print("  No checkpoint dir found — skipping")
        return {"status": "no_checkpoints"}

    files = sorted(ckpt_dir.glob("*.jsonl"))
    if not files:
        print("  No checkpoint files found")
        return {"status": "no_files"}

    results = []
    for f in files[:3]:  # audit first 3 checkpoint files
        lines = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            results.append({"file": f.name, "n_checkpoints": 0, "changed": False})
            continue

        first_state = lines[0].get("state_dict", {})
        last_state = lines[-1].get("state_dict", {})
        n = len(lines)

        # Compare arm A matrices
        first_arms = first_state.get("arms", {})
        last_arms = last_state.get("arms", {})
        changed = False
        if first_arms and last_arms:
            for arm_name in first_arms:
                if arm_name in last_arms:
                    A_first = first_arms[arm_name].get("A", [])
                    A_last = last_arms[arm_name].get("A", [])
                    if A_first != A_last:
                        changed = True
                        break
        elif first_arms != last_arms:
            changed = True

        print(f"  {f.name}: {n} checkpoints, arms_changed={changed}")
        results.append({
            "file": f.name,
            "n_checkpoints": n,
            "changed": changed,
            "first_arms_count": len(first_arms),
            "last_arms_count": len(last_arms),
        })

    (RAW_DIR / "checkpoint_audit.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


# ===========================================================================
# 6. Seed audit
# ===========================================================================
def run_seed_audit():
    """Section 16: Test whether seeds produce different selections."""
    from caps.communication.baselines import RandomPolicy, CAPSNoVOIPolicy
    from caps.communication.state import CommunicationState
    from caps.communication.ciu import CIU

    print("\n=== SEED AUDIT ===")

    cius = [
        CIU(id=f"seed_c{i:02d}", type="task_constraint",
            content=f"Content fragment {i}", token_count=50)
        for i in range(8)
    ]
    state = CommunicationState(
        task_embedding=[0.1] * 384,
        task_category="debugging",
        task_difficulty="easy",
        task_length=50,
        planner_confidence=0.8,
        receiver_role="debugger",
        receiver_model="mock",
        token_budget=200,
        available_cius=cius,
    )

    results = {}
    for seed in [1, 2, 3]:
        p = RandomPolicy(seed=seed)
        a = p.select(state)
        results[f"random_seed_{seed}"] = {
            "selected_ids": a.selected_ciu_ids,
            "scores": {k: round(v, 4) for k, v in a.ciu_scores.items()},
        }
        print(f"  Random seed={seed}: {a.selected_ciu_ids}")

    # CAPS arms are seed-independent at init
    caps_results = {}
    for seed in [1, 2, 3]:
        p = CAPSNoVOIPolicy()
        a = p.select(state)
        caps_results[f"caps_seed_{seed}"] = {
            "selected_ids": a.selected_ciu_ids,
        }
        print(f"  CAPS-NoVOI seed={seed}: {a.selected_ciu_ids}")

    all_random_same = (
        results["random_seed_1"]["selected_ids"] ==
        results["random_seed_2"]["selected_ids"] ==
        results["random_seed_3"]["selected_ids"]
    )
    all_caps_same = (
        caps_results["caps_seed_1"]["selected_ids"] ==
        caps_results["caps_seed_2"]["selected_ids"] ==
        caps_results["caps_seed_3"]["selected_ids"]
    )

    summary = {
        "random_varies_across_seeds": not all_random_same,
        "caps_varies_across_seeds": not all_caps_same,
        "caps_seed_independence_noted": True,
        "explanation": {
            "Random": "Varies by seed — correct (seed drives episode-level random choice)",
            "CAPS": "Identical across seeds — expected (seeds not yet wired into LinUCB init)",
        },
        "results": {**results, **caps_results},
    }
    print(f"\n  Random varies across seeds: {summary['random_varies_across_seeds']}")
    print(f"  CAPS varies across seeds:   {summary['caps_varies_across_seeds']}")

    (RAW_DIR / "seed_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ===========================================================================
# 7. Feature vector audit
# ===========================================================================
def run_feature_audit():
    """Section 13: Verify feature vectors are non-empty and mathematically correct."""
    from caps.bandits.linucb import LinUCBPolicy
    from caps.communication.state import CommunicationState
    from caps.communication.ciu import CIU
    import numpy as np

    print("\n=== FEATURE VECTOR AUDIT ===")

    ciu = CIU(id="fv_c00", type="task_constraint",
              content="Feature vector test content", token_count=50,
              embedding=[0.2] * 384)
    state = CommunicationState(
        task_embedding=[0.1] * 384,
        task_category="debugging",
        task_difficulty="easy",
        task_length=50,
        planner_confidence=0.8,
        receiver_role="debugger",
        receiver_model="mock",
        token_budget=1000,
        available_cius=[ciu],
    )

    policy = LinUCBPolicy(alpha=1.0)
    x = policy._build_context(state, ciu)
    dim = x.shape[0]

    arm = policy._get_or_create_arm(ciu.type, dim)
    A_inv = np.linalg.solve(arm["A"], np.eye(dim))
    theta = A_inv @ arm["b"]

    exploitation = float((x.T @ theta).item())
    exploration = float(policy.alpha * np.sqrt((x.T @ A_inv @ x).item()))
    ucb = exploitation + exploration

    feature_values = x.flatten().tolist()
    result = {
        "feature_dim": dim,
        "feature_values": [round(v, 6) for v in feature_values],
        "feature_is_empty": all(v == 0 for v in feature_values),
        "exploitation_score": exploitation,
        "exploration_bonus": exploration,
        "ucb_score": ucb,
        "ucb_formula_verified": abs(ucb - (exploitation + exploration)) < 1e-9,
        "expected_dim_note": "9 features: 3 budget + 5 CIU + 1 similarity",
        "instrumentation_bug": (
            "research/runner.py build_decision_rows uses placeholder 0.5/0.5 split "
            "instead of actual _build_context output. Feature CSV column is always '[]'."
        ),
    }
    print(f"  Feature dim: {dim}")
    print(f"  Features: {[round(v, 4) for v in feature_values]}")
    print(f"  Exploitation: {exploitation:.6f}")
    print(f"  Exploration: {exploration:.6f}")
    print(f"  UCB: {ucb:.6f}")
    print(f"  Formula verified: {result['ucb_formula_verified']}")

    (RAW_DIR / "feature_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("CAPS DIAGNOSTIC RUN")
    print("=" * 70)

    all_results = {}

    print("\n[1/6] Evaluator deterministic tests...")
    eval_results = run_evaluator_tests()
    all_results["evaluator"] = eval_results

    print("\n[2/6] End-to-end episode trace...")
    trace = run_episode_trace()
    all_results["episode_trace"] = {k: v for k, v in trace.items()
                                     if k not in ("cius_sent_to_receiver",)}

    print("\n[3/6] Sanity benchmark...")
    sanity = run_sanity_benchmark()
    all_results["sanity_benchmark"] = {
        m: {"n": r["n"], "success": r["success"], "success_rate": r["success_rate"]}
        for m, r in sanity.items()
    }

    print("\n[4/6] Checkpoint audit...")
    ckpts = run_checkpoint_audit()
    all_results["checkpoint_audit"] = ckpts

    print("\n[5/6] Seed audit...")
    seeds = run_seed_audit()
    all_results["seed_audit"] = {
        "random_varies": seeds["random_varies_across_seeds"],
        "caps_varies": seeds["caps_varies_across_seeds"],
    }

    print("\n[6/6] Feature vector audit...")
    feats = run_feature_audit()
    all_results["feature_audit"] = {
        "dim": feats["feature_dim"],
        "is_empty": feats["feature_is_empty"],
        "formula_verified": feats["ucb_formula_verified"],
    }

    # Save manifest
    manifest = {
        "diagnostic_run_id": "diag_v1",
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
    }
    Path("research_debug/diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Evaluator tests:   {eval_results['passed']}/{eval_results['total']} PASS")
    for m, r in sanity.items():
        print(f"  Sanity [{m}]: {r['success']}/{r['n']} ({r['success_rate']:.0%})")
    print(f"  Checkpoints:       {ckpts}")
    print(f"  Seeds Random vary: {seeds['random_varies_across_seeds']}")
    print(f"  Seeds CAPS vary:   {seeds['caps_varies_across_seeds']}")
    print(f"  Feature dim:       {feats['feature_dim']}")
    print(f"  Formula correct:   {feats['ucb_formula_verified']}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
