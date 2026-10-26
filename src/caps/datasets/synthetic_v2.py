"""Synthetic v2 benchmark for communication-selection experiments.

The generated task dictionaries contain hidden labels for evaluation. Call
``visible_cius`` before sending CIUs to agents or policies.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HIDDEN_FIELDS = {
    "information_role",
    "ground_truth_necessity",
    "ground_truth_sufficiency",
    "relevance",
    "dependency_group",
    "source_agent",
}

FAMILIES = [
    "debugging",
    "configuration",
    "data_transformation",
    "code_repair",
    "documentation_lookup",
    "multi_hop_reasoning",
    "contradiction_resolution",
]

DIFFICULTIES = ["easy", "medium", "hard"]

ROLE_BY_FAMILY = {
    "debugging": "debugger",
    "configuration": "sysadmin",
    "data_transformation": "data_engineer",
    "code_repair": "developer",
    "documentation_lookup": "analyst",
    "multi_hop_reasoning": "research_analyst",
    "contradiction_resolution": "investigator",
}

TYPE_BY_ROLE = {
    "necessary": "task_constraint",
    "useful": "implementation_hint",
    "redundant": "redundant_context",
    "irrelevant": "irrelevant_documentation",
    "misleading": "outdated_or_conflicting_context",
    "conditional": "conditional_context",
}


def _pad(text: str, target_tokens: int) -> str:
    """Pad text to a rough token target without adding label leakage."""
    words = text.split()
    if not words:
        words = ["context"]
    target_words = max(1, int(target_tokens * 0.75))
    extra = []
    while len(words) + len(extra) < target_words:
        extra.extend([
            "operational",
            "trace",
            "constraint",
            "version",
            "module",
            "behavior",
            "observed",
            "expected",
        ])
    return text + "\n\nDetails: " + " ".join(extra[: max(0, target_words - len(words))])


def _token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _ciu(
    task_id: str,
    idx: int,
    role: str,
    content: str,
    target_tokens: int,
    dependency_group: str,
    relevance: float,
) -> Dict[str, Any]:
    content = _pad(content, target_tokens)
    return {
        "id": f"{task_id}_ciu_{idx:02d}",
        "type": TYPE_BY_ROLE[role],
        "content": content,
        "token_count": _token_count(content),
        "source": "synthetic_v2",
        "source_agent": "benchmark_generator",
        "information_role": role,
        "dependency_group": dependency_group,
        "relevance": relevance,
        "ground_truth_necessity": role == "necessary",
        "ground_truth_sufficiency": role in {"necessary", "useful", "conditional"},
        "metadata": {"benchmark_version": "synthetic_v2"},
    }


def _family_prompt(family: str, index: int) -> Dict[str, str]:
    cases = {
        "debugging": (
            "Resolve the production failure in the request parser.",
            "parser_guard",
            "The correct fix requires adding an empty-input guard before parsing.",
        ),
        "configuration": (
            "Determine the correct service configuration for deployment.",
            "port_5432",
            "The service must use PostgreSQL on port 5432 with TLS enabled.",
        ),
        "data_transformation": (
            "Write the transformation rule for the incoming partner feed.",
            "normalize_iso_dates",
            "Dates must be normalized to ISO-8601 and missing fields become empty strings.",
        ),
        "code_repair": (
            "Repair the failing function described by the tests.",
            "inclusive_upper_bound",
            "The loop must include the upper bound and preserve zero as a valid result.",
        ),
        "documentation_lookup": (
            "Answer the operational policy question from the provided documentation.",
            "three_retries",
            "The API client allows exactly three retries with exponential backoff.",
        ),
        "multi_hop_reasoning": (
            "Infer the correct remediation by combining incident notes and version facts.",
            "upgrade_connector",
            "The connector must be upgraded because the current runtime removed the old endpoint.",
        ),
        "contradiction_resolution": (
            "Choose the valid runbook instruction when documents disagree.",
            "prefer_current_runbook",
            "The current runbook supersedes the archived note and requires the safer rollback.",
        ),
    }
    task, answer_key, answer_text = cases[family]
    return {
        "task": f"{task} Case {index}.",
        "answer_key": answer_key,
        "answer_text": answer_text,
    }


def generate_task(family: str, difficulty: str, index: int, seed: int) -> Dict[str, Any]:
    r = random.Random(seed * 100_003 + index)
    task_id = f"synthetic_v2_{family}_{difficulty}_{index:04d}_s{seed}"
    spec = _family_prompt(family, index)
    token_scale = {"easy": 65, "medium": 95, "hard": 135}[difficulty]
    counts = {
        "easy": {"necessary": 2, "useful": 1, "redundant": 1, "irrelevant": 2, "misleading": 1, "conditional": 0},
        "medium": {"necessary": 3, "useful": 2, "redundant": 2, "irrelevant": 4, "misleading": 2, "conditional": 1},
        "hard": {"necessary": 4, "useful": 3, "redundant": 3, "irrelevant": 6, "misleading": 3, "conditional": 2},
    }[difficulty]
    cius: List[Dict[str, Any]] = []
    idx = 0
    for role, n in counts.items():
        for j in range(n):
            if role == "necessary":
                content = f"{spec['answer_text']} Required evidence fragment {j + 1} for {family}."
                group = f"req_{j + 1}"
                relevance = 1.0
            elif role == "useful":
                content = f"Helpful implementation note for {family}: validate the final answer key {spec['answer_key']} against examples."
                group = "supporting"
                relevance = 0.75
            elif role == "redundant":
                content = f"Secondary note repeats known context about {family} without adding a new required constraint."
                group = "redundant"
                relevance = 0.45
            elif role == "misleading":
                content = f"Archived note suggests an obsolete remediation for {family}; it conflicts with current operational behavior."
                group = "conflict"
                relevance = -0.5
            elif role == "conditional":
                content = f"Conditional note for {family}: use only when the task mentions legacy compatibility or emergency rollback."
                group = "conditional"
                relevance = 0.35
            else:
                content = f"Unrelated background document about billing dashboards and office scheduling for case {index}."
                group = "noise"
                relevance = 0.0
            cius.append(_ciu(task_id, idx, role, content, token_scale + r.randint(-15, 25), group, relevance))
            idx += 1
    r.shuffle(cius)
    necessary = [c["id"] for c in cius if c["ground_truth_necessity"]]
    useful = [c["id"] for c in cius if c["information_role"] in {"necessary", "useful", "conditional"}]
    misleading = [c["id"] for c in cius if c["information_role"] == "misleading"]
    return {
        "task_id": task_id,
        "benchmark_version": "synthetic_v2",
        "generator_version": "synthetic_v2.0",
        "category": family,
        "difficulty": difficulty,
        "task": spec["task"],
        "receiver_role": ROLE_BY_FAMILY[family],
        "answer_key": spec["answer_key"],
        "answer_text": spec["answer_text"],
        "cius": cius,
        "ground_truth": {
            "necessary_cius": necessary,
            "useful_cius": useful,
            "misleading_cius": misleading,
            "success_rule": "all necessary CIUs selected and selected misleading CIUs below threshold",
        },
    }


def generate_synthetic_v2_dataset(
    size: int = 210,
    seed: int = 1,
    families: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    families = families or FAMILIES
    difficulties = difficulties or DIFFICULTIES
    tasks = []
    idx = 0
    while len(tasks) < size:
        for family in families:
            for difficulty in difficulties:
                if len(tasks) >= size:
                    break
                tasks.append(generate_task(family, difficulty, idx, seed))
                idx += 1
    random.Random(seed).shuffle(tasks)
    return tasks


def visible_ciu(ciu: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in ciu.items() if k not in HIDDEN_FIELDS}


def visible_task(task: Dict[str, Any]) -> Dict[str, Any]:
    clean = {k: v for k, v in task.items() if k != "ground_truth"}
    clean["cius"] = [visible_ciu(c) for c in task.get("cius", [])]
    return clean


def save_jsonl(tasks: Iterable[Dict[str, Any]], path: str | Path, visible_only: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            row = visible_task(task) if visible_only else task
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_hash(tasks: Iterable[Dict[str, Any]]) -> str:
    data = "\n".join(json.dumps(t, sort_keys=True) for t in tasks)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def create_splits(
    tasks: List[Dict[str, Any]],
    seed: int = 1,
    train: float = 0.6,
    validation: float = 0.2,
    test: float = 0.2,
) -> Dict[str, Any]:
    if round(train + validation + test, 6) != 1.0:
        raise ValueError("split fractions must sum to 1.0")
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs in synthetic_v2 dataset")
    r = random.Random(seed)
    shuffled = list(ids)
    r.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * validation)
    return {
        "benchmark_version": "synthetic_v2",
        "generator_version": "synthetic_v2.0",
        "seed": seed,
        "dataset_hash": dataset_hash(tasks),
        "splits": {
            "train": shuffled[:n_train],
            "validation": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        },
    }


def save_splits(splits: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
