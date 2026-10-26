"""
SWE-Bench Dataset Generator for CAPS (Lightweight).

Downloads a subset of SWE-Bench Lite from HuggingFace and formats
it into the CAPS task dictionary structure.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List
import requests

logger = logging.getLogger(__name__)


def fetch_file_from_github(repo: str, commit: str, filepath: str) -> str:
    """Fetch raw file content from GitHub."""
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{filepath}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.text
        return f"[Error fetching {filepath}: {resp.status_code}]"
    except Exception as e:
        return f"[Exception fetching {filepath}: {e}]"


def generate_swebench_dataset(num_tasks: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate a lightweight SWE-Bench dataset for CAPS.
    
    1. Loads swebench/swe-bench_lite from HuggingFace.
    2. Uses problem_statement as the task.
    3. Fetches modified files from GitHub to act as source_code CIUs.
    4. Includes the gold patch as ground truth for LLM evaluation.
    """
    import sys
    original_path = list(sys.path)
    sys.path = [p for p in sys.path if not p.endswith('src') and not p.endswith('caps')]
    try:
        from datasets import load_dataset
    finally:
        sys.path = original_path
    import re

    logger.info(f"Loading SWE-Bench Lite dataset from HuggingFace...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    # Shuffle and select subset
    dataset = dataset.shuffle(seed=seed).select(range(min(num_tasks, len(dataset))))
    
    tasks = []
    
    for row in dataset:
        task_id = row["instance_id"]
        repo = row["repo"]
        commit = row["base_commit"]
        desc = row["problem_statement"]
        patch = row["patch"]
        
        # 1. CIU for task description
        cius = [
            {
                "id": f"{task_id}_ciu_desc",
                "type": "task_description",
                "content": desc,
                "token_count": max(10, len(desc) // 4),
                "necessary": True
            }
        ]
        
        # 2. Extract files modified in the patch
        # diff --git a/file b/file
        files_changed = set(re.findall(r"^diff --git a/(.*?) b/", patch, flags=re.MULTILINE))
        
        # Limit to 3 files to keep requests/context manageable
        files_changed = list(files_changed)[:3]
        
        for filepath in files_changed:
            content = fetch_file_from_github(repo, commit, filepath)
            # Truncate content to avoid blowing up the context completely (e.g. max 5000 chars)
            if len(content) > 5000:
                content = content[:5000] + "\n...[truncated]..."
                
            cius.append({
                "id": f"{task_id}_ciu_{filepath.split('/')[-1]}",
                "type": "source_code",
                "content": f"# File: {filepath}\n{content}",
                "token_count": max(10, len(content) // 4),
                "necessary": True
            })
            
        # 3. Add a mock distractor CIU
        cius.append({
            "id": f"{task_id}_ciu_distractor",
            "type": "irrelevant_documentation",
            "content": "This is unrelated documentation. It does not help solve the issue.",
            "token_count": 20,
            "necessary": False
        })
        
        tasks.append({
            "task_id": task_id,
            "category": "software_engineering",
            "difficulty": "hard",
            "task": f"Fix this issue:\n{desc}\nOutput ONLY a git patch that resolves the issue.",
            "cius": cius,
            "ground_truth": {
                "patch": patch
            }
        })
        logger.info(f"Generated task for {task_id} with {len(cius)} CIUs.")
        
    return tasks
