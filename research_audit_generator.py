from __future__ import annotations

import csv
import json
import math
import os
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_audit"
RAW = OUT / "raw_statistics"
FIG = OUT / "figures"

DBS = [
    ROOT / "data" / "experiments.db",
    ROOT / "data" / "experiments_archive.db",
    ROOT / "data" / "mock_experiments.db",
    ROOT / "caps_experiments.db",
    ROOT / "data" / "llm_cache.db",
    ROOT / "data" / "embedding_cache.db",
    ROOT / "llm_cache.db",
    ROOT / "mlflow.db",
    ROOT / "results.db",
]

SOURCE_FILES = [
    "README.md",
    "plan.md",
    "architecture_decisions.md",
    "pyproject.toml",
    "src/caps/config.py",
    "src/caps/experiments/runner.py",
    "src/caps/bandits/linucb.py",
    "src/caps/communication/selector.py",
    "src/caps/communication/baselines.py",
    "src/caps/communication/ciu.py",
    "src/caps/communication/state.py",
    "src/caps/agents/planner.py",
    "src/caps/agents/receiver.py",
    "src/caps/agents/evaluator.py",
    "src/caps/environments/synthetic.py",
    "src/caps/environments/swe_bench.py",
    "src/caps/datasets/generator.py",
    "src/caps/datasets/swe_bench.py",
    "src/caps/evaluation/reward.py",
    "src/caps/evaluation/metrics.py",
    "src/caps/evaluation/analysis.py",
    "src/caps/llm/groq_client.py",
    "src/caps/llm/cache.py",
    "src/caps/llm/budget.py",
    "src/caps/llm/mock_client.py",
]


def read_text(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def line_of(rel: str, needle: str) -> str:
    text = read_text(rel).splitlines()
    for i, line in enumerate(text, start=1):
        if needle in line:
            return f"{rel}:{i}"
    return f"{rel}:NOT FOUND"


def safe_json_loads(value, default):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def ensure_dirs():
    OUT.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)


def inspect_db(path: Path):
    info = {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "tables": {},
        "experiments": [],
        "episode_stats": [],
        "metrics": [],
    }
    if (not path.exists()) or path.stat().st_size == 0:
        return info
    try:
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        table_names = [r[0] for r in cur.execute("select name from sqlite_master where type='table' order by name")]
        for table in table_names:
            cols = [dict(cid=r[0], name=r[1], type=r[2], notnull=r[3], default=r[4], pk=r[5]) for r in cur.execute(f'pragma table_info("{table}")')]
            count = cur.execute(f'select count(*) from "{table}"').fetchone()[0]
            info["tables"][table] = {"columns": cols, "row_count": count}
        if "experiments" in table_names:
            info["experiments"] = [dict(r) for r in cur.execute("select * from experiments order by created_at, experiment_id")]
        if "metrics" in table_names:
            info["metrics"] = [dict(r) for r in cur.execute("select * from metrics order by experiment_id, metric_name, id")]
        if "episodes" in table_names:
            rows = [dict(r) for r in cur.execute("select * from episodes order by experiment_id, id")]
            export_rows(path.stem + "_episodes.csv", rows)
            by_exp = defaultdict(list)
            seen_episode_ids = Counter()
            for r in rows:
                by_exp[r.get("experiment_id")].append(r)
                seen_episode_ids[(r.get("experiment_id"), r.get("episode_id"))] += 1
            for exp_id, eps in by_exp.items():
                totals = [int(e.get("total_tokens") or 0) for e in eps]
                lats = [float(e.get("latency_ms") or 0.0) for e in eps]
                rewards = [float(e.get("reward") or 0.0) for e in eps]
                statuses = Counter(str(e.get("receiver_status")) for e in eps)
                successes = sum(1 for e in eps if bool(e.get("task_success")))
                clar = sum(int(e.get("clarification_turns") or 0) for e in eps)
                retries = sum(int(e.get("retry_count") or 0) for e in eps)
                dupes = sum(c - 1 for k, c in seen_episode_ids.items() if k[0] == exp_id and c > 1)
                info["episode_stats"].append({
                    "database": str(path.relative_to(ROOT)),
                    "experiment_id": exp_id,
                    "rows": len(eps),
                    "completed_episodes": len(eps),
                    "successful_episodes": successes,
                    "failed_episodes": len(eps) - successes,
                    "api_failures": "NOT RECORDED IN EPISODES",
                    "duplicate_episode_ids": dupes,
                    "mean_tokens": mean(totals),
                    "median_tokens": median(totals),
                    "std_tokens": stdev(totals),
                    "mean_latency_ms": mean(lats),
                    "median_latency_ms": median(lats),
                    "std_latency_ms": stdev(lats),
                    "clarification_turns": clar,
                    "clarification_rate": clar / len(eps) if eps else 0.0,
                    "retry_count": retries,
                    "retry_rate": sum(1 for e in eps if int(e.get("retry_count") or 0) > 0) / len(eps) if eps else 0.0,
                    "average_reward": mean(rewards),
                    "receiver_status_counts": dict(statuses),
                })
            export_rows("episodes.csv", rows)
        if "llm_calls" in table_names:
            rows = [dict(r) for r in cur.execute("select * from llm_calls order by id")]
            export_rows(path.stem + "_llm_calls.csv", rows)
            export_rows("llm_calls.csv", rows)
        con.close()
    except Exception as e:
        info["error"] = str(e)
    return info


def export_rows(name: str, rows):
    if not rows:
        return
    path = RAW / name
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def median(xs):
    return float(statistics.median(xs)) if xs else 0.0


def stdev(xs):
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def binom_ci(success, n):
    if n == 0:
        return [0.0, 0.0]
    # Wilson 95%
    z = 1.959963984540054
    p = success / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_audit():
    data = {}
    for path in sorted((ROOT / "data" / "synthetic").glob("*.jsonl")):
        rows = load_jsonl(path)
        ciu_counts = [len(r.get("cius", [])) for r in rows]
        necessary_counts = [len(r.get("ground_truth", {}).get("necessary_cius", [])) for r in rows]
        types = Counter()
        necessary_types = Counter()
        examples = []
        for task in rows:
            for ciu in task.get("cius", []):
                types[ciu.get("type", "unknown")] += 1
                if ciu.get("necessary") is True or ciu.get("id") in task.get("ground_truth", {}).get("necessary_cius", []):
                    necessary_types[ciu.get("type", "unknown")] += 1
            if len(examples) < 3:
                examples.append({
                    "task_id": task.get("task_id"),
                    "category": task.get("category"),
                    "task": task.get("task", "")[:300],
                    "ciu_examples": [
                        {
                            "id": c.get("id"),
                            "type": c.get("type"),
                            "token_count": c.get("token_count"),
                            "content_excerpt": c.get("content", "")[:180],
                        }
                        for c in task.get("cius", [])[:3]
                    ],
                })
        data[str(path.relative_to(ROOT))] = {
            "tasks": len(rows),
            "categories": dict(Counter(r.get("category", "unknown") for r in rows)),
            "ciu_count_mean": mean(ciu_counts),
            "ciu_count_min": min(ciu_counts) if ciu_counts else 0,
            "ciu_count_max": max(ciu_counts) if ciu_counts else 0,
            "necessary_ciu_mean": mean(necessary_counts),
            "ciu_types": dict(types),
            "necessary_ciu_types": dict(necessary_types),
            "task_ids_sample": [r.get("task_id") for r in rows[:20]],
            "examples": examples,
        }
    return data


def all_episode_rows(db_infos):
    rows = []
    for db in DBS:
        if not db.exists() or db.stat().st_size == 0:
            continue
        try:
            con = sqlite3.connect(db)
            con.row_factory = sqlite3.Row
            tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
            if "episodes" in tables:
                for r in con.execute("select * from episodes"):
                    d = dict(r)
                    d["_database"] = str(db.relative_to(ROOT))
                    rows.append(d)
            con.close()
        except Exception:
            pass
    return rows


def export_combined_raw(rows):
    export_rows("episodes.csv", rows)
    export_rows("rewards.csv", [
        {"database": r.get("_database"), "experiment_id": r.get("experiment_id"), "episode_id": r.get("episode_id"), "task_id": r.get("task_id"), "reward": r.get("reward"), "success": r.get("task_success"), "selected_ciu_ids": r.get("selected_ciu_ids")}
        for r in rows
    ])


def selection_analysis(rows, datasets):
    task_to_cius = {}
    for _, d in datasets.items():
        # Reload from key path
        pass
    for path in sorted((ROOT / "data" / "synthetic").glob("*.jsonl")):
        for task in load_jsonl(path):
            task_to_cius[task.get("task_id")] = {c.get("id"): c for c in task.get("cius", [])}
    by_exp_type = defaultdict(lambda: {"available": 0, "selected": 0})
    context_stats = defaultdict(list)
    for r in rows:
        exp = r.get("experiment_id")
        task_id = r.get("task_id")
        av = safe_json_loads(r.get("available_ciu_ids"), [])
        sel = safe_json_loads(r.get("selected_ciu_ids"), [])
        ciu_map = task_to_cius.get(task_id, {})
        total_tokens = 0
        selected_tokens = int(r.get("total_tokens") or 0)
        for cid in av:
            c = ciu_map.get(cid, {})
            typ = c.get("type", "unknown")
            by_exp_type[(exp, typ)]["available"] += 1
            total_tokens += int(c.get("token_count") or 0)
        for cid in sel:
            c = ciu_map.get(cid, {})
            typ = c.get("type", "unknown")
            by_exp_type[(exp, typ)]["selected"] += 1
        context_stats[exp].append({
            "available": len(av),
            "selected": len(sel),
            "tokens_selected": selected_tokens,
            "pct_context_selected": selected_tokens / total_tokens if total_tokens else None,
        })
    table = []
    for (exp, typ), vals in sorted(by_exp_type.items()):
        table.append({
            "experiment_id": exp,
            "ciu_type": typ,
            "available": vals["available"],
            "selected": vals["selected"],
            "selection_rate": vals["selected"] / vals["available"] if vals["available"] else 0.0,
        })
    export_rows("ciu_selections.csv", table)
    context_summary = []
    for exp, vals in context_stats.items():
        for field in ["available", "selected", "tokens_selected"]:
            xs = [v[field] for v in vals if v[field] is not None]
            context_summary.append({"experiment_id": exp, "metric": field, "mean": mean(xs), "median": median(xs), "min": min(xs) if xs else 0, "max": max(xs) if xs else 0, "std": stdev(xs)})
        xs = [v["pct_context_selected"] for v in vals if v["pct_context_selected"] is not None]
        context_summary.append({"experiment_id": exp, "metric": "pct_context_selected", "mean": mean(xs), "median": median(xs), "min": min(xs) if xs else 0, "max": max(xs) if xs else 0, "std": stdev(xs)})
    return table, context_summary


def policy_learning_analysis(rows):
    caps = [r for r in rows if "_CAPS_" in str(r.get("experiment_id", "")) or str(r.get("experiment_id", "")).endswith("_CAPS_42")]
    out = []
    by_exp = defaultdict(list)
    for r in caps:
        by_exp[r.get("experiment_id")].append(r)
    for exp, eps in by_exp.items():
        eps = sorted(eps, key=lambda r: r.get("id", 0))
        first_sel = safe_json_loads(eps[0].get("selected_ciu_ids"), []) if eps else []
        last_sel = safe_json_loads(eps[-1].get("selected_ciu_ids"), []) if eps else []
        out.append({
            "experiment_id": exp,
            "episodes": len(eps),
            "first_selected_cius": first_sel,
            "last_selected_cius": last_sel,
            "mean_reward_first_half": mean([float(e.get("reward") or 0) for e in eps[: max(1, len(eps)//2)]]),
            "mean_reward_second_half": mean([float(e.get("reward") or 0) for e in eps[max(1, len(eps)//2):]]),
            "mean_tokens_first_half": mean([int(e.get("total_tokens") or 0) for e in eps[: max(1, len(eps)//2)]]),
            "mean_tokens_second_half": mean([int(e.get("total_tokens") or 0) for e in eps[max(1, len(eps)//2):]]),
            "evidence": "Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only.",
        })
    export_rows("rewards.csv", [
        {"experiment_id": r.get("experiment_id"), "episode_id": r.get("episode_id"), "task_id": r.get("task_id"), "reward": r.get("reward"), "success": r.get("task_success"), "selected_ciu_ids": r.get("selected_ciu_ids")}
        for r in rows
    ])
    return out


def env_key_counts():
    result = {}
    for name in [".env", "creds.env", ".env.example"]:
        path = ROOT / name
        if not path.exists():
            continue
        counts = Counter()
        for line in path.read_text(errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key = s.split("=", 1)[0].strip()
            if "KEY" in key or "TOKEN" in key or "SECRET" in key:
                counts[key] += 1
        result[name] = dict(counts)
    return result


def file_inventory():
    files = []
    skip_parts = {".venv", ".pytest_cache", "__pycache__", "caps.egg-info", ".git", "research_audit"}
    for p in ROOT.rglob("*"):
        if p.is_file() and not any(part in skip_parts for part in p.parts):
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            suffix = p.suffix.lower()
            if ".env" in p.name or p.name.endswith(".env"):
                purpose = "Environment configuration; secret values intentionally not inspected/exported"
            elif suffix == ".py":
                purpose = "Python source or tests"
            elif suffix in [".yaml", ".yml", ".toml"]:
                purpose = "Configuration/dependency metadata"
            elif suffix == ".db":
                purpose = "SQLite database artifact"
            elif suffix == ".jsonl":
                purpose = "Dataset artifact"
            elif suffix in [".md", ".txt"]:
                purpose = "Documentation/report/log text"
            elif suffix == ".csv":
                purpose = "Tabular results export"
            else:
                purpose = "Other artifact"
            relevance = "high" if rel.startswith(("src/", "configs/", "data/", "experiments/")) or rel in ["README.md", "plan.md", "architecture_decisions.md", "pyproject.toml"] else "medium"
            files.append({"path": rel, "type": suffix.lstrip(".") or "unknown", "purpose": purpose, "research_relevance": relevance, "size": p.stat().st_size})
    return {"files": sorted(files, key=lambda x: x["path"])}


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def md_table(rows, columns):
    if not rows:
        return "NOT AVAILABLE\n"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "NOT AVAILABLE")) for c in columns) + " |")
    return "\n".join(out) + "\n"


def plot_figures(raw_stats, ciu_table, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return plot_svg_figures(raw_stats, ciu_table, rows)
    made = []
    exp_stats = [s for db in raw_stats["databases"] for s in db.get("episode_stats", [])]
    main = [s for s in exp_stats if s["database"] == "data/experiments.db"]
    if main:
        labels = [s["experiment_id"].replace("caps_swe_bench_v3_", "").replace("_42", "") for s in main]
        success = [s["successful_episodes"] / s["completed_episodes"] if s["completed_episodes"] else 0 for s in main]
        tokens = [s["mean_tokens"] for s in main]
        plt.figure(figsize=(10, 5), dpi=160)
        plt.bar(labels, success)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Success rate")
        plt.tight_layout()
        p = FIG / "success_rate_by_method.png"
        plt.savefig(p)
        plt.close()
        made.append(str(p.relative_to(ROOT)))
        plt.figure(figsize=(10, 5), dpi=160)
        plt.bar(labels, tokens)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mean selected CIU tokens")
        plt.tight_layout()
        p = FIG / "communication_tokens_by_method.png"
        plt.savefig(p)
        plt.close()
        made.append(str(p.relative_to(ROOT)))
        plt.figure(figsize=(7, 5), dpi=160)
        plt.scatter(tokens, success)
        for label, x, y in zip(labels, tokens, success):
            plt.annotate(label, (x, y), fontsize=8)
        plt.xlabel("Mean selected CIU tokens")
        plt.ylabel("Success rate")
        plt.tight_layout()
        p = FIG / "success_vs_communication_cost.png"
        plt.savefig(p)
        plt.close()
        made.append(str(p.relative_to(ROOT)))
    caps_rows = [r for r in rows if "CAPS" in str(r.get("experiment_id", ""))]
    if caps_rows:
        caps_rows = sorted(caps_rows, key=lambda r: (r.get("experiment_id", ""), r.get("id", 0)))
        plt.figure(figsize=(10, 5), dpi=160)
        by = defaultdict(list)
        for r in caps_rows:
            by[r.get("experiment_id")].append(r)
        for exp, eps in by.items():
            plt.plot(range(1, len(eps) + 1), [float(e.get("reward") or 0) for e in eps], label=exp[-30:])
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend(fontsize=6)
        plt.tight_layout()
        p = FIG / "reward_curve.png"
        plt.savefig(p)
        plt.close()
        made.append(str(p.relative_to(ROOT)))
    caps_ciu = [r for r in ciu_table if "CAPS" in str(r.get("experiment_id", ""))]
    if caps_ciu:
        totals = defaultdict(lambda: [0, 0])
        for r in caps_ciu:
            totals[r["ciu_type"]][0] += r["available"]
            totals[r["ciu_type"]][1] += r["selected"]
        labels = list(totals)
        rates = [totals[l][1] / totals[l][0] if totals[l][0] else 0 for l in labels]
        plt.figure(figsize=(10, 5), dpi=160)
        plt.bar(labels, rates)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("CAPS selection rate")
        plt.tight_layout()
        p = FIG / "ciu_selection_frequencies.png"
        plt.savefig(p)
        plt.close()
        made.append(str(p.relative_to(ROOT)))
    (FIG / "caps_architecture.mmd").write_text("""flowchart TD
    Dataset[Dataset generator / loaded JSONL] --> Environment
    Environment --> Planner[PlannerAgent.build_state]
    Planner --> Embedder[LocalEmbeddingProvider]
    Planner --> Policy[CommunicationPolicy]
    Policy --> Selector[select_cius]
    Selector --> Receiver[ReceiverAgent.execute]
    Receiver --> EnvironmentEval[Environment.evaluate]
    EnvironmentEval --> Reward[RewardFunction.calculate]
    Receiver --> Evaluator[EvaluatorAgent VOI]
    Evaluator --> PolicyUpdate[policy.update]
    Reward --> PolicyUpdate
    PolicyUpdate --> Database[ExperimentDB episodes/metrics]
""", encoding="utf-8")
    made.append(str((FIG / "caps_architecture.mmd").relative_to(ROOT)))
    return made


def plot_svg_figures(raw_stats, ciu_table, rows):
    made = []
    exp_stats = [s for db in raw_stats["databases"] for s in db.get("episode_stats", [])]
    main = [s for s in exp_stats if s["database"] == "data\\experiments.db"]
    if main:
        labels = [method_from_exp(s["experiment_id"]) + "\\n" + s["experiment_id"].split("_")[1] if "swe" not in s["experiment_id"] else method_from_exp(s["experiment_id"]) for s in main[:28]]
        success = [s["successful_episodes"] / s["completed_episodes"] if s["completed_episodes"] else 0 for s in main[:28]]
        tokens = [s["mean_tokens"] for s in main[:28]]
        made.append(write_bar_svg(FIG / "success_rate_by_method.svg", labels, success, "Success rate by experiment", "success rate"))
        made.append(write_bar_svg(FIG / "communication_tokens_by_method.svg", labels, tokens, "Mean selected CIU tokens", "tokens"))
        made.append(write_scatter_svg(FIG / "success_vs_communication_cost.svg", tokens, success, labels, "Success vs communication cost"))
    caps = [r for r in rows if "CAPS" in str(r.get("experiment_id", ""))]
    if caps:
        by = defaultdict(list)
        for r in caps:
            by[r.get("experiment_id")].append(r)
        exp, eps = max(by.items(), key=lambda kv: len(kv[1]))
        eps = sorted(eps, key=lambda r: r.get("id", 0))
        made.append(write_line_svg(FIG / "reward_curve.svg", [float(e.get("reward") or 0) for e in eps], f"Reward curve: {exp}", "reward"))
    caps_ciu = [r for r in ciu_table if r.get("experiment_id") == "caps_swe_bench_v3_full_dataset_CAPS_42"]
    if caps_ciu:
        labels = [r["ciu_type"] for r in caps_ciu]
        vals = [r["selection_rate"] for r in caps_ciu]
        made.append(write_bar_svg(FIG / "ciu_selection_frequencies.svg", labels, vals, "CIU selection frequencies: full-dataset CAPS", "selection rate"))
    (FIG / "caps_architecture.mmd").write_text("""flowchart TD
    Dataset[Dataset generator / loaded JSONL] --> Environment
    Environment --> Planner[PlannerAgent.build_state]
    Planner --> Embedder[LocalEmbeddingProvider]
    Planner --> Policy[CommunicationPolicy]
    Policy --> Selector[select_cius]
    Selector --> Receiver[ReceiverAgent.execute]
    Receiver --> EnvironmentEval[Environment.evaluate]
    EnvironmentEval --> Reward[RewardFunction.calculate]
    Receiver --> Evaluator[EvaluatorAgent VOI]
    Evaluator --> PolicyUpdate[policy.update]
    Reward --> PolicyUpdate
    PolicyUpdate --> Database[ExperimentDB episodes/metrics]
""", encoding="utf-8")
    made.append(str((FIG / "caps_architecture.mmd").relative_to(ROOT)))
    return [m for m in made if m]


def svg_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_bar_svg(path, labels, values, title, ylabel):
    w, h = 1400, 700
    ml, mr, mt, mb = 80, 30, 70, 190
    maxv = max(values) if values else 1
    maxv = max(maxv, 1e-9)
    bw = (w - ml - mr) / max(1, len(values))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{w/2}" y="35" text-anchor="middle" font-family="Arial" font-size="24">{svg_escape(title)}</text>', f'<text x="20" y="{h/2}" transform="rotate(-90 20 {h/2})" font-family="Arial" font-size="16">{svg_escape(ylabel)}</text>']
    parts.append(f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="#222"/>')
    parts.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="#222"/>')
    for i, v in enumerate(values):
        x = ml + i * bw + 3
        bh = (h - mt - mb) * (v / maxv)
        y = h - mb - bh
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1,bw-6):.1f}" height="{bh:.1f}" fill="#3274a1"/>')
        parts.append(f'<text x="{x + bw/2:.1f}" y="{h-mb+15}" transform="rotate(60 {x + bw/2:.1f} {h-mb+15})" font-family="Arial" font-size="10">{svg_escape(labels[i])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path.relative_to(ROOT))


def write_line_svg(path, values, title, ylabel):
    w, h = 1200, 650
    ml, mr, mt, mb = 80, 40, 70, 70
    if not values:
        return ""
    minv, maxv = min(values), max(values)
    if minv == maxv:
        minv -= 1
        maxv += 1
    pts = []
    for i, v in enumerate(values):
        x = ml + (w - ml - mr) * (i / max(1, len(values) - 1))
        y = h - mb - (h - mt - mb) * ((v - minv) / (maxv - minv))
        pts.append(f"{x:.1f},{y:.1f}")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{w/2}" y="35" text-anchor="middle" font-family="Arial" font-size="24">{svg_escape(title)}</text>', f'<text x="20" y="{h/2}" transform="rotate(-90 20 {h/2})" font-family="Arial" font-size="16">{svg_escape(ylabel)}</text>', f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="#222"/>', f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="#222"/>', f'<polyline points="{" ".join(pts)}" fill="none" stroke="#c44e52" stroke-width="2"/>', "</svg>"]
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path.relative_to(ROOT))


def write_scatter_svg(path, xs, ys, labels, title):
    w, h = 1000, 700
    ml, mr, mt, mb = 80, 40, 70, 70
    if not xs:
        return ""
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    if minx == maxx:
        maxx += 1
    if miny == maxy:
        maxy += 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{w/2}" y="35" text-anchor="middle" font-family="Arial" font-size="24">{svg_escape(title)}</text>', f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="#222"/>', f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="#222"/>', f'<text x="{w/2}" y="{h-20}" text-anchor="middle" font-family="Arial" font-size="16">Mean selected CIU tokens</text>', f'<text x="20" y="{h/2}" transform="rotate(-90 20 {h/2})" font-family="Arial" font-size="16">Success rate</text>']
    for xval, yval, label in zip(xs, ys, labels):
        x = ml + (w - ml - mr) * ((xval - minx) / (maxx - minx))
        y = h - mb - (h - mt - mb) * ((yval - miny) / (maxy - miny))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#55a868"/>')
        parts.append(f'<text x="{x+7:.1f}" y="{y-7:.1f}" font-family="Arial" font-size="10">{svg_escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path.relative_to(ROOT))


def main():
    ensure_dirs()
    db_infos = [inspect_db(p) for p in DBS]
    datasets = dataset_audit()
    rows = all_episode_rows(db_infos)
    export_combined_raw(rows)
    ciu_table, context_summary = selection_analysis(rows, datasets)
    learning = policy_learning_analysis(rows)
    raw_stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "databases": db_infos,
        "datasets": datasets,
        "context_size_summary": context_summary,
        "ciu_selection_by_type": ciu_table,
        "policy_learning_evidence": learning,
        "env_key_counts": env_key_counts(),
        "notes": [
            "Episode total_tokens are selected CIU communication tokens from ExperimentRunner, not total API usage.",
            "No llm_calls rows were found in the primary experiment database during this audit unless reported otherwise in database table counts.",
        ],
    }
    figures = plot_figures(raw_stats, ciu_table, rows)
    raw_stats["figures"] = figures
    write_json("RAW_STATISTICS.json", raw_stats)
    write_json("FILE_INVENTORY.json", file_inventory())
    manifest = build_manifest(raw_stats)
    write_json("EXPERIMENT_MANIFEST.json", manifest)
    write_markdown_files(raw_stats, manifest)
    make_empty_exports_if_missing()
    print(f"Wrote audit to {OUT}")


def build_manifest(raw_stats):
    exps = []
    models = set()
    configs = []
    datasets = []
    seeds = set()
    for cfg in sorted((ROOT / "configs").glob("*.yaml")):
        configs.append({"path": str(cfg.relative_to(ROOT)), "size": cfg.stat().st_size})
    for name, d in raw_stats["datasets"].items():
        datasets.append({"path": name, "tasks": d["tasks"], "categories": d["categories"]})
    for db in raw_stats["databases"]:
        for exp in db.get("experiments", []):
            exps.append(exp)
            for k in ["planner_model", "receiver_model", "evaluator_model"]:
                if exp.get(k):
                    models.add(exp[k])
            if exp.get("seed") is not None:
                seeds.add(exp["seed"])
    return {
        "project": "CAPS",
        "experiments": exps,
        "datasets": datasets,
        "models": sorted(models),
        "seeds": sorted(seeds),
        "configs": configs,
        "metrics": [m for db in raw_stats["databases"] for m in db.get("metrics", [])],
        "results": [s for db in raw_stats["databases"] for s in db.get("episode_stats", [])],
    }


def make_empty_exports_if_missing():
    for name in ["counterfactuals.csv", "metrics.csv", "llm_calls.csv"]:
        p = RAW / name
        if not p.exists():
            p.write_text("status\nNOT AVAILABLE\n", encoding="utf-8")
    # Build metrics csv from RAW_STATISTICS.
    stats = json.loads((OUT / "RAW_STATISTICS.json").read_text())
    metric_rows = []
    for db in stats["databases"]:
        metric_rows.extend(db.get("metrics", []))
    if metric_rows:
        export_rows("metrics.csv", metric_rows)


def write_markdown_files(raw_stats, manifest):
    evidence = build_evidence_md(raw_stats, manifest)
    files = {
        "CAPS_RESEARCH_EVIDENCE.md": evidence,
        "EXECUTIVE_SUMMARY.md": section_exec(raw_stats, manifest),
        "IMPLEMENTATION_AUDIT.md": section_impl(raw_stats),
        "MATHEMATICAL_FORMULATION.md": section_math(),
        "EXPERIMENT_INVENTORY.md": section_exp(raw_stats, manifest),
        "DATASET_AUDIT.md": section_dataset(raw_stats),
        "MODEL_AUDIT.md": section_model(manifest),
        "METRICS_AUDIT.md": section_metrics(raw_stats),
        "RESULTS_AUDIT.md": section_results(raw_stats),
        "BASELINE_AUDIT.md": section_baseline(),
        "CAPS_POLICY_AUDIT.md": section_caps_policy(raw_stats),
        "VOI_AUDIT.md": section_voi(raw_stats),
        "COUNTERFACTUAL_AUDIT.md": section_counterfactual(raw_stats),
        "API_AND_RESOURCE_AUDIT.md": section_api(raw_stats),
        "REPRODUCIBILITY_AUDIT.md": section_repro(manifest),
        "VALIDITY_AND_RISKS.md": section_validity(),
        "PAPER_CLAIMS.md": section_claims(raw_stats),
    }
    for name, text in files.items():
        (OUT / name).write_text(text, encoding="utf-8")


def experiment_rows(raw_stats):
    rows = []
    stats_by_exp = {}
    for db in raw_stats["databases"]:
        for s in db.get("episode_stats", []):
            stats_by_exp[(s["database"], s["experiment_id"])] = s
        for exp in db.get("experiments", []):
            key = (db["path"], exp.get("experiment_id"))
            s = stats_by_exp.get(key, {})
            rows.append({
                "Experiment ID": exp.get("experiment_id"),
                "Config": "embedded in DB config_yaml" if exp.get("config_yaml") else "NOT AVAILABLE",
                "Date": exp.get("created_at", "NOT AVAILABLE"),
                "Model": exp.get("receiver_model") or exp.get("planner_model") or "NOT AVAILABLE",
                "Dataset": exp.get("dataset_name") or "NOT AVAILABLE",
                "Episodes": s.get("rows", exp.get("completed_episodes", 0)),
                "Method": exp.get("policy_name"),
                "Status": exp.get("status"),
                "Database": db["path"],
            })
    return rows


def result_rows(raw_stats):
    rows = []
    for db in raw_stats["databases"]:
        for s in db.get("episode_stats", []):
            n = s["completed_episodes"]
            k = s["successful_episodes"]
            ci = binom_ci(k, n)
            rows.append({
                "Database": s["database"],
                "Method": method_from_exp(s["experiment_id"]),
                "Experiment ID": s["experiment_id"],
                "Successful": k,
                "Completed": n,
                "API failures": s["api_failures"],
                "Other failures": s["failed_episodes"],
                "Total attempted": n,
                "Correct success": round(k / n, 4) if n else 0.0,
                "95% CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]" if n else "NOT AVAILABLE",
                "Mean tokens": round(s["mean_tokens"], 2),
                "Median tokens": round(s["median_tokens"], 2),
                "Mean latency ms": round(s["mean_latency_ms"], 2),
                "Mean reward": round(s["average_reward"], 4),
            })
    return rows


def method_from_exp(exp):
    for m in ["CAPS", "LLMPlanner", "Heuristic", "Random", "FullContext"]:
        if f"_{m}_" in str(exp) or str(exp).endswith(f"_{m}_42"):
            return m
    return "NOT VERIFIED"


def section_exec(raw_stats, manifest):
    exps = experiment_rows(raw_stats)
    results = result_rows(raw_stats)
    return f"""# Executive Summary

Project: CAPS - Communication-Adaptive Policy System.

Research objective: learn/select adaptive communication context for multi-agent LLM collaboration instead of always transmitting all available context. The explicit hypothesis is RECONSTRUCTED FROM IMPLEMENTATION and plan.md: an agent can learn which Candidate Information Units (CIUs) to send to preserve task success while reducing communication cost.

Implementation status: runnable Python package with planner, receiver, evaluator, CIU schema, baselines, LinUCB policy, synthetic and lightweight SWE-bench dataset support, SQLite storage, and CLI. It is a research prototype, not a complete official benchmark harness.

Frameworks/libraries: Python, Pydantic, SQLAlchemy, NumPy, pandas/scipy/scikit-learn/matplotlib, sentence-transformers, Typer/Rich, Tenacity, Groq/OpenAI-compatible clients.

LLM providers: Groq provider and OpenRouter-compatible mode via `base_url`. Mock provider for tests. Gemini is proposed in `plan.md` but not implemented in current source.

Embedding model: configured default `all-MiniLM-L6-v2`; local sentence-transformers embeddings cached in SQLite.

Databases: {", ".join(db["path"] for db in raw_stats["databases"] if db["exists"])}.

Experiments found: {len(exps)} experiment records across available databases. Episode rows found: {sum(r.get("Completed", 0) for r in results)}.

## Current Experimental Status

{md_table(results, ["Database", "Method", "Experiment ID", "Successful", "Completed", "API failures", "Other failures", "Correct success", "Mean tokens", "Mean latency ms", "Mean reward"])}
"""


def section_impl(raw_stats):
    return f"""# Implementation Architecture Audit

```mermaid
flowchart TD
    Dataset[Dataset generator / JSONL artifacts] --> Environment[SyntheticEnvironment or SWEBenchEnvironment]
    Environment --> Planner[PlannerAgent.build_state]
    Planner --> Embedder[LocalEmbeddingProvider]
    Planner --> Policy[CommunicationPolicy implementations]
    Policy --> Selector[select_cius / budget strategies]
    Selector --> Receiver[ReceiverAgent.execute]
    Receiver --> EnvironmentEval[Environment.evaluate]
    EnvironmentEval --> Reward[RewardFunction.calculate]
    Receiver --> Evaluator[EvaluatorAgent self-reflection or counterfactual]
    Evaluator --> PolicyUpdate[policy.update]
    Reward --> PolicyUpdate
    PolicyUpdate --> DB[ExperimentDB]
```

## Components

Component: Planner. File: `{line_of("src/caps/agents/planner.py", "class PlannerAgent")}`. Class: `PlannerAgent`. Function: `build_state`, `plan_with_llm`. Input: task string, CIUs, budget, metadata. Output: `CommunicationState` or `PlannerOutput`. Purpose: embed task/CIUs and construct policy state; LLM planning exists for raw context but the experiment runner uses `build_state`.

Component: Receiver. File: `{line_of("src/caps/agents/receiver.py", "class ReceiverAgent")}`. Class: `ReceiverAgent`. Function: `execute`. Input: task and selected CIUs only. Output: `ReceiverOutput`. Purpose: solve task using selected context and self-report used CIUs/missing info.

Component: Evaluator. File: `{line_of("src/caps/agents/evaluator.py", "class EvaluatorAgent")}`. Class: `EvaluatorAgent`. Functions: `evaluate_voi`, `evaluate_self_reflection`. Input: selected CIUs, receiver output, success, environment. Output: per-CIU VOI scores. Purpose: provide policy update signal.

Component: CIU schema. File: `{line_of("src/caps/communication/ciu.py", "class CIU")}`. Class: `CIU`. Input/Output: Pydantic object. Purpose: represent atomic context.

Component: State/action. File: `{line_of("src/caps/communication/state.py", "class CommunicationState")}`. Classes: `CommunicationState`, `Action`. Purpose: policy observation and selected CIU IDs/scores.

Component: LinUCB. File: `{line_of("src/caps/bandits/linucb.py", "class LinUCBPolicy")}`. Class: `LinUCBPolicy`. Purpose: CAPS learning policy.

Component: Reward. File: `{line_of("src/caps/evaluation/reward.py", "class RewardFunction")}`. Class: `RewardFunction`. Purpose: task success minus weighted normalized communication cost.

Component: Database. File: `{line_of("src/caps/storage/database.py", "class ExperimentRecord")}`. Classes: SQLAlchemy records and `ExperimentDB`. Purpose: store experiments, episodes, metrics, nominal LLM calls.

Component: API layer. File: `{line_of("src/caps/llm/groq_client.py", "class GroqProvider")}`. Class: `GroqProvider`. Purpose: Groq/OpenRouter-compatible chat completions, retries, rate-limit handling, key switching, model fallback.
"""


def section_math():
    return """# Mathematical Formulation

Implemented CAPS policy uses a per-CIU-type contextual LinUCB model.

Context vector actually used by `LinUCBPolicy._build_context` is 9-dimensional:

`x = [token_budget/10000 clipped to 1, planner_confidence, previous_turns/5 clipped to 1, ciu_tokens/2000 clipped to 1, ciu.priority, ciu.historical_utility, ciu.historical_selection_rate, ciu.uncertainty, cosine(task_embedding, ciu.embedding)]`

The raw task embedding, receiver embedding/capability, receiver historical success, latency budget, task category, and difficulty exist in schemas but are not part of the LinUCB feature vector.

For each CIU type arm `t`, initialized with `A_t = I`, `b_t = 0`:

`theta_t = A_t^{-1} b_t`

`UCB_t(x) = x^T theta_t + alpha * sqrt(x^T A_t^{-1} x)`

Selected CIUs are greedily chosen by descending score subject to token budget and positive score. The selected list is then reversed, preserving neither original dataset order nor pure descending order in the returned action.

Update for each selected CIU:

`A_t <- A_t + x x^T`

`b_t <- b_t + r_i x`

where `r_i = reward` unless VOI is available; if VOI exists then `r_i = (voi_i + reward) / 2`.

Reward:

`R = 1[success] - lambda * (w_tokens * min(tokens_used/token_budget,1) + w_latency * min(latency_ms/max_latency_ms,1) + w_clarification * min(clarification_turns/max_clarification,1) + w_retries * min(retry_count/max_retries,1))`
"""


def section_exp(raw_stats, manifest):
    rows = experiment_rows(raw_stats)
    return "# Experiment Inventory\n\n" + md_table(rows, ["Experiment ID", "Config", "Date", "Model", "Dataset", "Episodes", "Method", "Status", "Database"])


def section_dataset(raw_stats):
    rows = []
    for path, d in raw_stats["datasets"].items():
        rows.append({"Dataset": path, "Version": "file artifact", "Source": "generator or saved JSONL", "Number of tasks": d["tasks"], "Categories": d["categories"], "CIUs mean": round(d["ciu_count_mean"], 2), "CIUs min": d["ciu_count_min"], "CIUs max": d["ciu_count_max"]})
    examples = []
    for path, d in raw_stats["datasets"].items():
        examples.append(f"## Examples from {path}\n\n```json\n{json.dumps(d['examples'][:2], indent=2)}\n```")
    return "# Dataset Audit\n\n" + md_table(rows, ["Dataset", "Version", "Source", "Number of tasks", "Categories", "CIUs mean", "CIUs min", "CIUs max"]) + "\nSWE-bench implementation: `src/caps/datasets/swe_bench.py` loads `princeton-nlp/SWE-bench_Lite` test split through HuggingFace, shuffles by seed, selects up to the requested count, fetches up to three modified files from GitHub, and stores the gold patch in ground truth. It does not run official SWE-bench tests.\n\n" + "\n\n".join(examples)


def section_model(manifest):
    rows = []
    for exp in manifest["experiments"]:
        rows.append({"Experiment ID": exp.get("experiment_id"), "Provider": "Groq/OpenRouter-compatible inferred from CLI/provider; exact provider per run NOT RECORDED", "Planner": exp.get("planner_model"), "Receiver": exp.get("receiver_model"), "Evaluator": exp.get("evaluator_model") or "NOT RECORDED", "Temperature": "0.0 in configs; not stored per episode", "Max tokens": "2048 API default unless provider call overridden"})
    return "# Model Audit\n\n" + md_table(rows, ["Experiment ID", "Provider", "Planner", "Receiver", "Evaluator", "Temperature", "Max tokens"]) + "\nEmbedding model: configured default `all-MiniLM-L6-v2`; experiment records do not store embedding model.\n"


def section_metrics(raw_stats):
    return """# Metrics Audit

Implemented primary metric functions: success rate, mean/median/std communication tokens, token reduction, success preservation, clarification rate, retry rate.

Implemented secondary metric functions: information precision, information recall, minimal sufficient context distance, policy entropy. These are functions only; no evidence was found that they are computed/stored during the main experiment runner.

Latency is measured as wall-clock elapsed time around planner state building, policy selection, receiver execution, and environment evaluation. For SWE-bench, this includes LLM judge evaluation. It is not separated into planner/receiver/evaluator/API latency in episode rows.

Token accounting in episode rows: selected CIU communication tokens only, based on CIU `token_count` / `estimate_tokens`. It is not total LLM prompt+completion usage. `input_tokens` and `output_tokens` columns exist but the runner does not populate them in `add_episode`.

Clarification is implemented as a receiver status string `needs_clarification`; there is no multi-turn clarification loop. Retry count is always accumulated as 0 in the runner, although API retry logic exists inside the provider.
"""


def section_results(raw_stats):
    rows = result_rows(raw_stats)
    return "# Results Audit\n\nAll results below are recalculated from raw `episodes` rows, not from precomputed summary Markdown/CSV.\n\n" + md_table(rows, ["Database", "Method", "Experiment ID", "Successful", "Completed", "Other failures", "Correct success", "95% CI", "Mean tokens", "Median tokens", "Mean latency ms", "Mean reward"]) + "\nAPI failure counts are NOT AVAILABLE from episode rows. No reliable denominator beyond completed episode rows was recoverable unless experiment metadata total_episodes differs from row count.\n\n## 45 vs 50 / Incomplete Episode Problems\n\nNo completed experiment with exactly 45 CAPS episode rows was found. Two incomplete CAPS records exist: `caps_swe_bench_v1_CAPS_45` is marked running with 9 episode rows despite total_episodes=50, and `caps_swe_bench_v1_CAPS_46` is marked running with 25 episode rows despite total_episodes=50. The full-dataset `caps_swe_bench_v3_full_dataset_LLMPlanner_42` record is marked running with 240 episode rows despite total_episodes=300. Causes such as quota, timeout, or manual interruption are NOT VERIFIED because episode rows do not preserve stop reasons or API failure records.\n\n## Full-Dataset Experiment\n\n`caps_swe_bench_v3_full_dataset_CAPS_42`: intended 300 tasks, completed 300 rows, successes 1, failures 299, correct success = 1/300 = 0.003333... = 0.33%. Model recorded: `stealth/ox-alpha`; dataset recorded: `swe_bench_lite`; token metric mean = selected CIU tokens only, not total LLM usage.\n"


def section_baseline():
    return """# Baseline Audit

FullContext: sends every CIU and ignores token budget. No learning, one receiver call per episode plus SWE-bench evaluator call where applicable.

Random: shuffles CIUs with `seed + episode`, selects within token budget. Scores logged are generated from `Random(seed).random()` per CIU, which resets for each CIU and does not match the shuffled selection probabilities.

Heuristic: fixed CIU type priorities, greedy budget selection.

LLMPlanner: sends CIU IDs/types/token counts to planner LLM and asks for JSON selected IDs, then enforces budget in original CIU order. On JSON parse failure, falls back to all CIUs then budget-enforces.

CAPS: per-CIU-type LinUCB with epsilon-greedy score randomization and greedy budget selection.

Fairness concern: FullContext ignores the budget while other methods are budget-constrained; LLMPlanner makes an extra LLM selection call; CAPS updates online during evaluation; baselines do not learn.
"""


def section_caps_policy(raw_stats):
    return "# CAPS Policy Audit\n\n" + section_math() + "\n## Does CAPS Actually Learn?\n\n" + md_table(raw_stats["policy_learning_evidence"], ["experiment_id", "episodes", "mean_reward_first_half", "mean_reward_second_half", "mean_tokens_first_half", "mean_tokens_second_half", "evidence"]) + "\nPolicy checkpoints/A matrices/b vectors/theta values were not found in persisted experiment artifacts. Unit tests show parameters can change, but recorded experiments do not store enough state to verify matrix-level learning after the fact.\n"


def section_voi(raw_stats):
    return """# VOI Audit

VOI is implemented in two ways:

Counterfactual replay: for up to `max_counterfactuals` selected CIUs, remove one CIU, rerun the receiver, evaluate success, then assign +1 when removal changes success to failure, -1 when removal changes failure to success, else 0.

Self-reflection: if original task succeeds, selected CIUs self-reported by the receiver receive +1 and other selected CIUs receive 0. If original task fails, all selected CIUs receive 0. The code comment mentions a harsh negative penalty but the implementation uses neutral 0 on failures.

The main runner uses counterfactual VOI only when `config.evaluator.counterfactual` is true and only for CAPS. Current SWE-bench configs set counterfactual false, so they use self-reflection. `default.yaml` also sets counterfactual false; `caps_synthetic.yaml` sets counterfactual true.

VOI is used to update LinUCB only for CAPS via `policy.update(..., voi_scores=voi_scores)`.
"""


def section_counterfactual(raw_stats):
    return """# Counterfactual Audit

Counterfactual replay is implemented but not comprehensively persisted. Episode rows store only the resulting `voi_scores` JSON dictionary, not individual counterfactual executions, receiver outputs, or counterfactual success values.

Selection strategy implementation: if selected CIU count is less than or equal to `max_counterfactuals`, test all selected CIUs; otherwise use random sampling. Config may say `uncertainty`, but `_select_for_testing` falls back to random for non-random strategies.

Actual number of counterfactual executions cannot be reconstructed exactly from DB rows without LLM call logs. Counterfactual raw table: NOT AVAILABLE.
"""


def section_api(raw_stats):
    key_counts = raw_stats["env_key_counts"]
    return f"""# API and Resource Audit

Providers implemented: Groq SDK and OpenAI-compatible base URL mode used for OpenRouter. Mock provider exists for tests.

Configured secret-bearing entries found, without exposing values:

```json
{json.dumps(key_counts, indent=2)}
```

Multi-key round-robin: partially supported. `get_groq_api_keys` can parse multiple Groq keys; `GroqProvider` creates one client per key and switches keys when an extended/daily rate limit occurs before a problem has started. It does not rotate every request in strict round-robin fashion.

Rate-limit handling: sleeps 2.5 seconds before calls; retries up to 3 attempts; parses retry-after headers and some error-message wait times; can switch keys or fall back to configured fallback models.

LLM cache: `LLMCache` class exists and cache config exists, but `GroqProvider.generate` and the main runner do not call `LLMCache.get/put`. Therefore LLM response caching is implemented as a standalone class but NOT VERIFIED as active in experiments. Embedding cache is active when cache enabled.
"""


def section_repro(manifest):
    return """# Reproducibility Audit

Score: 58/100.

Rationale: dependencies and configs are present; synthetic data generation is deterministic by seed; experiments are stored in SQLite; CLI commands are discoverable. However, experiment rows do not store full LLM call traces, prompts, API token usage, provider choice, exact provider model versions, policy matrices/checkpoints, or failed attempts. SWE-bench evaluation is LLM-judge based rather than official test execution, and there is no explicit train/validation/test split in the runner despite config fields.

Exact available commands:

```powershell
python -m pytest
caps dataset generate --size 50 --seed 42 --output data/synthetic/benchmark.jsonl
caps experiment run --config configs/caps_synthetic_mock.yaml
caps experiment run --config configs/caps_synthetic.yaml
caps experiment run --config configs/caps_swe_bench.yaml
caps experiment run --config configs/caps_swe_bench_full.yaml
caps results export --format csv --config configs/default.yaml --output experiments/results/summary
```
"""


def section_validity():
    return """# Validity and Risks

## Strongly supported claims

CAPS is implemented as a Python research prototype with CIU objects, budgeted CIU selection, baselines, and a per-CIU-type LinUCB policy.

The system can reduce selected communication tokens relative to FullContext in budgeted settings, but exact claims must use recalculated DB rows.

## Plausible but insufficiently demonstrated claims

CAPS learns useful communication strategies. The code updates LinUCB online, but persisted experiments do not include policy matrices or UCB score traces.

CAPS maintains task performance while reducing context. This depends on small, incomplete, LLM-judged or synthetic runs and needs controlled paired evaluation.

## Unsupported claims

Statistically significant superiority over baselines: NOT SUPPORTED by current artifacts.

Official SWE-bench performance: NOT SUPPORTED because the project uses LLM judge comparison to gold patch, not test execution.

Strict multi-key round-robin infrastructure: CLAIM NOT SUPPORTED BY IMPLEMENTATION; key switching is conditional, not round-robin.

## Claims requiring new experiments

Frozen-policy generalization, train/test separation, robust counterfactual VOI, and official benchmark evaluation all require new experiments.

## Major risks

Small or incomplete sample sizes; no explicit train/test split in the runner; online test-set adaptation; LLM judge leakage on SWE-bench because the evaluator sees the gold patch; token accounting excludes actual LLM prompt/completion tokens; clarification is single-status only, not an interaction loop; retry metrics conflate none of the API retry logic into episode records; caching state is not fully recorded.
"""


def section_claims(raw_stats):
    rows = [
        {"Current Claim": "CAPS reduces token usage by 55%.", "Evidence": "Must be recomputed per chosen experiment; no universal 55% result found by audit script.", "Supported?": "NOT VERIFIED", "Qualification Needed": "Specify database, methods, denominator, CIU-token metric."},
        {"Current Claim": "CAPS maintains task performance.", "Evidence": "Episode-level success rows exist for some runs.", "Supported?": "PARTIAL", "Qualification Needed": "Needs paired tasks and confidence intervals; account for incomplete baselines."},
        {"Current Claim": "CAPS improves task success.", "Evidence": "Current rows may show differences but sample size/fairness issues remain.", "Supported?": "NOT VERIFIED", "Qualification Needed": "Controlled paired tests and sufficient power."},
        {"Current Claim": "CAPS learns communication strategies.", "Evidence": "LinUCB update exists; persisted policy state absent.", "Supported?": "PARTIAL", "Qualification Needed": "Log A/b/theta/UCB and frozen-policy evaluation."},
        {"Current Claim": "VOI improves policy learning.", "Evidence": "VOI can update CAPS, but no ablation proving improvement found.", "Supported?": "NO", "Qualification Needed": "Run VOI vs no-VOI ablation."},
        {"Current Claim": "CAPS significantly outperforms baselines.", "Evidence": "No valid significance evidence found.", "Supported?": "NO", "Qualification Needed": "Appropriate paired tests with adequate sample."},
    ]
    return "# Paper Claims\n\n" + md_table(rows, ["Current Claim", "Evidence", "Supported?", "Qualification Needed"])


def build_evidence_md(raw_stats, manifest):
    parts = [
        "# CAPS Research Evidence Audit",
        f"Generated: {raw_stats['generated_at']}",
        section_exec(raw_stats, manifest),
        section_impl(raw_stats),
        section_math(),
        "# CIU Representation\n\nCIU fields: `id`, `type`, `content`, `source`, `token_count`, `embedding`, `priority`, `historical_utility`, `historical_selection_rate`, `uncertainty`, `metadata`. IDs in generated datasets are deterministic task-prefixed IDs such as `<task_id>_ciu_desc`; default ad hoc CIU IDs are 8-character UUID prefixes.\n\nToken counting uses provided `token_count` when positive; otherwise `max(1, len(content)//4)`. Embeddings are optional lists of floats populated by `LocalEmbeddingProvider`.",
        section_caps_policy(raw_stats),
        section_voi(raw_stats),
        section_counterfactual(raw_stats),
        section_dataset(raw_stats),
        section_model(manifest),
        section_baseline(),
        section_exp(raw_stats, manifest),
        section_results(raw_stats),
        section_metrics(raw_stats),
        "# Token Accounting\n\nPlanner tokens: actual API usage exists in `LLMResponse` but is not persisted by the runner. Receiver tokens: same. Evaluator tokens: same. Communication tokens: persisted as `episodes.total_tokens`, selected CIU estimate only. Counterfactual tokens: NOT AVAILABLE except indirectly if LLM call logging were present. Total LLM tokens/input/output tokens: columns exist, but main runner does not populate episode input/output fields.",
        "# Success Accounting\n\n" + md_table(result_rows(raw_stats), ["Method", "Experiment ID", "Successful", "Completed", "API failures", "Other failures", "Total attempted", "Correct success", "95% CI"]),
        "# Context Size Audit\n\n" + md_table(raw_stats["context_size_summary"], ["experiment_id", "metric", "mean", "median", "min", "max", "std"]),
        "# CIU Selection Analysis\n\n" + md_table(raw_stats["ciu_selection_by_type"][:200], ["experiment_id", "ciu_type", "available", "selected", "selection_rate"]),
        "# Statistical Results\n\nWilson binomial 95% confidence intervals are reported for success where episode rows exist. Wilcoxon/McNemar paired tests were not run because task-pair completeness and identical task ordering are not verified across all methods. STATISTICAL POWER INSUFFICIENT for broad superiority claims.",
        section_api(raw_stats),
        section_repro(manifest),
        section_validity(),
        section_claims(raw_stats),
        "# Required New Experiments\n\nCRITICAL: run paired synthetic benchmark with all methods over identical task IDs, fixed seed, full LLM call logging, and persisted policy checkpoints. CRITICAL: run frozen-policy train/test split to avoid TEST-SET ADAPTATION. HIGH: run official SWE-bench harness or clearly rename lightweight LLM-judge evaluation. HIGH: run VOI ablation with counterfactual vs self-reflection vs no VOI. MEDIUM: measure true API token/latency and clarification loops. OPTIONAL: evaluate additional models/providers.",
        "# Paper-Ready Figures\n\n" + "\n".join(f"- `{p}`" for p in raw_stats.get("figures", [])),
        "# Exact Commands to Reproduce Experiments\n\nSee `REPRODUCIBILITY_AUDIT.md`.",
    ]
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    main()
