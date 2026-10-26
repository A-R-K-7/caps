"""
Synthetic Benchmark Generator.

Generates controlled tasks with known CIU dependency structure.
This is the PRIMARY development benchmark per §34-36.

Each task has:
    - A task description
    - A set of CIUs (some necessary, some unnecessary)
    - Ground truth: which CIUs are necessary for success
    - A deterministic validator: checks if the receiver's answer is correct

Task Families:
    1. Debugging — fix code errors given partial context
    2. Configuration — resolve config issues
    3. Data Transformation — transform data given specs
    4. Code Repair — repair code given failing tests
    5. Documentation — answer questions from docs

Ground truth labels are NEVER included in receiver prompts.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from caps.communication.ciu import CIU


# ---------------------------------------------------------------------------
# Task templates for each family
# ---------------------------------------------------------------------------

DEBUGGING_TASKS = [
    {
        "task": "Fix the ZeroDivisionError in the calculate_average function.",
        "answer_keyword": "len(numbers) == 0",
        "answer_pattern": "check if the list is empty",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "The calculate_average function crashes with ZeroDivisionError when called with an empty list.", "necessary": True},
            {"id": "ciu_stack", "type": "stack_trace", "content": "Traceback (most recent call last):\n  File 'main.py', line 12, in calculate_average\n    return sum(numbers) / len(numbers)\nZeroDivisionError: division by zero", "necessary": True},
            {"id": "ciu_code", "type": "source_code", "content": "def calculate_average(numbers):\n    return sum(numbers) / len(numbers)", "necessary": True},
            {"id": "ciu_env", "type": "environment", "content": "Python 3.11.4, Ubuntu 22.04", "necessary": False},
            {"id": "ciu_readme", "type": "irrelevant_documentation", "content": "# Project README\nThis project provides utility functions for data processing.\nLicense: MIT\nAuthor: John Doe", "necessary": False},
            {"id": "ciu_log", "type": "irrelevant_logs", "content": "[2024-01-15 10:30:22] INFO: Server started on port 8080\n[2024-01-15 10:30:23] INFO: Database connection established", "necessary": False},
        ],
    },
    {
        "task": "Fix the ImportError in the data processing module.",
        "answer_keyword": "import pandas",
        "answer_pattern": "install pandas",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "The data processing module fails to start with an ImportError for pandas.", "necessary": True},
            {"id": "ciu_stack", "type": "stack_trace", "content": "Traceback (most recent call last):\n  File 'processor.py', line 1, in <module>\n    import pandas as pd\nModuleNotFoundError: No module named 'pandas'", "necessary": True},
            {"id": "ciu_code", "type": "source_code", "content": "import pandas as pd\nimport numpy as np\n\ndef process_data(filepath):\n    df = pd.read_csv(filepath)\n    return df.describe()", "necessary": False},
            {"id": "ciu_deps", "type": "dependency_versions", "content": "requirements.txt:\nnumpy==1.24.0\nscipy==1.10.0", "necessary": True},
            {"id": "ciu_readme", "type": "irrelevant_documentation", "content": "# Data Processor\nA tool for processing CSV files.\nVersion 2.1.0", "necessary": False},
            {"id": "ciu_unrelated", "type": "unrelated_files", "content": "# test_utils.py\ndef test_string_utils():\n    assert capitalize('hello') == 'Hello'", "necessary": False},
        ],
    },
    {
        "task": "Fix the TypeError when concatenating string and integer.",
        "answer_keyword": "str(",
        "answer_pattern": "convert.*to.*string",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "The format_output function throws TypeError: can only concatenate str to str.", "necessary": True},
            {"id": "ciu_stack", "type": "stack_trace", "content": "Traceback (most recent call last):\n  File 'output.py', line 5, in format_output\n    return 'Result: ' + value\nTypeError: can only concatenate str (not \"int\") to str", "necessary": True},
            {"id": "ciu_code", "type": "source_code", "content": "def format_output(value):\n    return 'Result: ' + value\n\nresult = format_output(42)", "necessary": True},
            {"id": "ciu_env", "type": "environment", "content": "Python 3.10.12, macOS Ventura 13.4", "necessary": False},
            {"id": "ciu_test", "type": "failing_test", "content": "def test_format():\n    assert format_output(42) == 'Result: 42'", "necessary": False},
            {"id": "ciu_log", "type": "irrelevant_logs", "content": "[DEBUG] Module loaded successfully\n[DEBUG] Cache initialized", "necessary": False},
        ],
    },
    {
        "task": "Fix the IndexError in the get_last_element function.",
        "answer_keyword": "len(items)",
        "answer_pattern": "check.*empty|length",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "get_last_element crashes with IndexError when given an empty list.", "necessary": True},
            {"id": "ciu_code", "type": "source_code", "content": "def get_last_element(items):\n    return items[-1]", "necessary": True},
            {"id": "ciu_stack", "type": "stack_trace", "content": "IndexError: list index out of range\n  in get_last_element, line 2", "necessary": False},
            {"id": "ciu_env", "type": "environment", "content": "Python 3.11, Windows 11", "necessary": False},
            {"id": "ciu_readme", "type": "irrelevant_documentation", "content": "# Utils Module\nProvides utility functions for list operations.", "necessary": False},
            {"id": "ciu_license", "type": "unrelated_files", "content": "MIT License\nCopyright (c) 2024", "necessary": False},
        ],
    },
    {
        "task": "Fix the KeyError in the config parser.",
        "answer_keyword": ".get(",
        "answer_pattern": "get.*default|check.*key",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "The config parser crashes with KeyError when 'database_url' is missing from config.", "necessary": True},
            {"id": "ciu_code", "type": "source_code", "content": "def parse_config(config_dict):\n    db_url = config_dict['database_url']\n    port = config_dict['port']\n    return {'db': db_url, 'port': port}", "necessary": True},
            {"id": "ciu_config", "type": "config_file", "content": "# config.yaml\nport: 5432\nlog_level: INFO", "necessary": True},
            {"id": "ciu_stack", "type": "stack_trace", "content": "KeyError: 'database_url'\n  in parse_config, line 2", "necessary": False},
            {"id": "ciu_env", "type": "environment", "content": "Python 3.12, Docker container", "necessary": False},
            {"id": "ciu_unrelated", "type": "irrelevant_logs", "content": "[INFO] Starting application...\n[INFO] Loading plugins...", "necessary": False},
        ],
    },
]

CONFIGURATION_TASKS = [
    {
        "task": "Fix the database connection configuration error.",
        "answer_keyword": "port",
        "answer_pattern": "port.*5432|change.*port",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "The application cannot connect to PostgreSQL. Connection refused error.", "necessary": True},
            {"id": "ciu_config", "type": "config_file", "content": "database:\n  host: localhost\n  port: 3306\n  name: myapp\n  user: admin", "necessary": True},
            {"id": "ciu_error", "type": "error_message", "content": "psycopg2.OperationalError: could not connect to server: Connection refused\n  Is the server running on host \"localhost\" and accepting TCP/IP connections on port 3306?", "necessary": True},
            {"id": "ciu_env", "type": "environment", "content": "PostgreSQL 15.2 running on port 5432", "necessary": True},
            {"id": "ciu_readme", "type": "irrelevant_documentation", "content": "# MyApp\nA web application built with Flask.\nVersion 3.0.1", "necessary": False},
            {"id": "ciu_unrelated", "type": "unrelated_files", "content": "# Makefile\nbuild:\n\tdocker-compose up -d", "necessary": False},
        ],
    },
    {
        "task": "Fix the CORS configuration issue.",
        "answer_keyword": "origins",
        "answer_pattern": "allow.*origin|cors.*config",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "Frontend requests are blocked by CORS policy. The API returns 403 for cross-origin requests.", "necessary": True},
            {"id": "ciu_config", "type": "config_file", "content": "cors:\n  enabled: true\n  allowed_origins:\n    - http://localhost:3000\n  allowed_methods:\n    - GET", "necessary": True},
            {"id": "ciu_error", "type": "error_message", "content": "Access to XMLHttpRequest at 'https://api.example.com/data' from origin 'https://app.example.com' has been blocked by CORS policy", "necessary": True},
            {"id": "ciu_env", "type": "environment", "content": "Frontend: https://app.example.com\nAPI: https://api.example.com", "necessary": True},
            {"id": "ciu_readme", "type": "irrelevant_documentation", "content": "# API Documentation\nEndpoints:\n- GET /data\n- POST /submit", "necessary": False},
            {"id": "ciu_log", "type": "irrelevant_logs", "content": "[2024-03-01] Server started\n[2024-03-01] Connected to Redis", "necessary": False},
        ],
    },
    {
        "task": "Fix the environment variable configuration for the secret key.",
        "answer_keyword": "SECRET_KEY",
        "answer_pattern": "set.*secret.*key|environment.*variable",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "Application fails to start: 'SECRET_KEY must be set'.", "necessary": True},
            {"id": "ciu_code", "type": "source_code", "content": "import os\n\nSECRET_KEY = os.environ['SECRET_KEY']\nDEBUG = os.environ.get('DEBUG', 'false')", "necessary": True},
            {"id": "ciu_env", "type": "environment", "content": "Environment variables set:\nDEBUG=true\nDATABASE_URL=postgres://...\nPORT=8000", "necessary": True},
            {"id": "ciu_error", "type": "error_message", "content": "KeyError: 'SECRET_KEY'", "necessary": False},
            {"id": "ciu_readme", "type": "irrelevant_documentation", "content": "# Deployment Guide\nDeploy using Docker or Heroku.", "necessary": False},
            {"id": "ciu_unrelated", "type": "unrelated_files", "content": "# Dockerfile\nFROM python:3.11\nCOPY . /app", "necessary": False},
        ],
    },
]

DATA_TRANSFORMATION_TASKS = [
    {
        "task": "Write a function to convert a list of dicts to CSV format.",
        "answer_keyword": "csv",
        "answer_pattern": "csv|join|header",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "Convert a list of dictionaries to CSV string format with headers.", "necessary": True},
            {"id": "ciu_schema", "type": "schema", "content": "Input: List[Dict[str, Any]]\nExample: [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\nOutput: CSV string with header row", "necessary": True},
            {"id": "ciu_sample", "type": "sample_data", "content": "[{'name': 'Alice', 'age': 30, 'city': 'NYC'}, {'name': 'Bob', 'age': 25, 'city': 'LA'}]", "necessary": True},
            {"id": "ciu_constraints", "type": "constraints", "content": "Must handle missing keys gracefully. Use empty string for missing values.", "necessary": True},
            {"id": "ciu_metadata", "type": "metadata", "content": "Created by: data team\nLast updated: 2024-01-01\nPriority: medium", "necessary": False},
            {"id": "ciu_unrelated", "type": "irrelevant_documentation", "content": "# Data Pipeline Architecture\nWe use Apache Airflow for orchestration.", "necessary": False},
        ],
    },
    {
        "task": "Write a function to flatten a nested JSON object.",
        "answer_keyword": "flatten",
        "answer_pattern": "recursive|flatten|separator",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "Flatten a nested JSON object into a single-level dict with dot-separated keys.", "necessary": True},
            {"id": "ciu_schema", "type": "schema", "content": "Input: {'a': {'b': 1, 'c': {'d': 2}}}\nOutput: {'a.b': 1, 'a.c.d': 2}", "necessary": True},
            {"id": "ciu_sample", "type": "sample_data", "content": "{'user': {'name': 'Alice', 'address': {'city': 'NYC', 'zip': '10001'}}}", "necessary": False},
            {"id": "ciu_constraints", "type": "constraints", "content": "Handle lists by indexing: key.0, key.1, etc.", "necessary": True},
            {"id": "ciu_env", "type": "environment", "content": "Python 3.11, no external libraries allowed", "necessary": False},
            {"id": "ciu_unrelated", "type": "irrelevant_documentation", "content": "# JSON Schema Specification\nVersion: draft-07", "necessary": False},
        ],
    },
]

CODE_REPAIR_TASKS = [
    {
        "task": "Fix the failing test for the fibonacci function.",
        "answer_keyword": "fibonacci",
        "answer_pattern": "fib|base.*case|recursive",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "The fibonacci function returns incorrect values. Fix the implementation.", "necessary": True},
            {"id": "ciu_test", "type": "failing_test", "content": "def test_fibonacci():\n    assert fibonacci(0) == 0\n    assert fibonacci(1) == 1\n    assert fibonacci(5) == 5\n    assert fibonacci(10) == 55", "necessary": True},
            {"id": "ciu_code", "type": "source_code", "content": "def fibonacci(n):\n    if n <= 1:\n        return 1  # Bug: should return n\n    return fibonacci(n-1) + fibonacci(n-2)", "necessary": True},
            {"id": "ciu_stack", "type": "stack_trace", "content": "AssertionError: assert fibonacci(0) == 0\n  assert 1 == 0", "necessary": False},
            {"id": "ciu_readme", "type": "irrelevant_documentation", "content": "# Math Utils\nA collection of mathematical utility functions.", "necessary": False},
            {"id": "ciu_unrelated", "type": "unrelated_files", "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)", "necessary": False},
        ],
    },
    {
        "task": "Fix the failing test for the binary search function.",
        "answer_keyword": "mid",
        "answer_pattern": "mid|binary.*search|comparison",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "Binary search function fails to find elements. Fix the off-by-one error.", "necessary": True},
            {"id": "ciu_test", "type": "failing_test", "content": "def test_binary_search():\n    arr = [1, 3, 5, 7, 9, 11]\n    assert binary_search(arr, 5) == 2\n    assert binary_search(arr, 11) == 5\n    assert binary_search(arr, 4) == -1", "necessary": True},
            {"id": "ciu_code", "type": "source_code", "content": "def binary_search(arr, target):\n    low, high = 0, len(arr)  # Bug: should be len(arr) - 1\n    while low < high:  # Bug: should be low <= high\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1", "necessary": True},
            {"id": "ciu_stack", "type": "stack_trace", "content": "AssertionError: binary_search([1,3,5,7,9,11], 11) returned -1, expected 5", "necessary": False},
            {"id": "ciu_env", "type": "environment", "content": "Python 3.11.4", "necessary": False},
            {"id": "ciu_unrelated", "type": "unrelated_files", "content": "# Sorting algorithms module\ndef quicksort(arr): ...", "necessary": False},
        ],
    },
]

DOCUMENTATION_TASKS = [
    {
        "task": "What is the maximum number of retries for the API client?",
        "answer_keyword": "3",
        "answer_pattern": "3|three",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "Find the maximum retry count configured for the API client.", "necessary": True},
            {"id": "ciu_doc_relevant", "type": "relevant_documentation", "content": "# API Client Configuration\n\nThe client supports automatic retries for transient failures.\n- max_retries: 3 (default)\n- backoff_factor: 2.0\n- retry_on: [500, 502, 503, 504]", "necessary": True},
            {"id": "ciu_doc_irrelevant", "type": "irrelevant_documentation", "content": "# Authentication Guide\nThe API uses OAuth2 with JWT tokens.\nToken expiry: 1 hour.\nRefresh tokens are valid for 30 days.", "necessary": False},
            {"id": "ciu_version", "type": "version", "content": "API Client v2.4.1\nReleased: 2024-02-15", "necessary": False},
            {"id": "ciu_constraints", "type": "constraints", "content": "Rate limit: 100 requests per minute per API key.", "necessary": False},
            {"id": "ciu_unrelated", "type": "unrelated_files", "content": "# Contributing Guide\nPlease submit PRs to the develop branch.", "necessary": False},
        ],
    },
    {
        "task": "What authentication method does the API use?",
        "answer_keyword": "OAuth2",
        "answer_pattern": "oauth|jwt|bearer",
        "cius": [
            {"id": "ciu_desc", "type": "task_description", "content": "Determine the authentication method used by the API.", "necessary": True},
            {"id": "ciu_doc_relevant", "type": "relevant_documentation", "content": "# Authentication\nThe API uses OAuth2 with Bearer tokens.\nTo authenticate:\n1. POST /auth/token with client_id and client_secret\n2. Include the token in Authorization header: Bearer <token>", "necessary": True},
            {"id": "ciu_doc_irrelevant", "type": "irrelevant_documentation", "content": "# API Endpoints\nGET /users - List all users\nPOST /users - Create a new user\nGET /users/{id} - Get user by ID", "necessary": False},
            {"id": "ciu_version", "type": "version", "content": "API v3.0.0", "necessary": False},
            {"id": "ciu_env", "type": "environment", "content": "Production server: api.example.com\nStaging: api-staging.example.com", "necessary": False},
            {"id": "ciu_unrelated", "type": "unrelated_files", "content": "# Changelog\nv3.0.0: Breaking changes to auth endpoints\nv2.5.0: Added rate limiting", "necessary": False},
        ],
    },
]


TASK_FAMILIES = {
    "debugging": DEBUGGING_TASKS,
    "configuration": CONFIGURATION_TASKS,
    "data_transformation": DATA_TRANSFORMATION_TASKS,
    "code_repair": CODE_REPAIR_TASKS,
    "documentation": DOCUMENTATION_TASKS,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _make_task_id(family: str, index: int, seed: int) -> str:
    return f"{family}_{index:03d}_s{seed}"


def generate_task(
    template: Dict[str, Any],
    family: str,
    index: int,
    seed: int,
) -> Dict[str, Any]:
    """Generate a single task instance from a template."""
    r = random.Random(seed + index)

    cius = []
    necessary_ids = []
    for ciu_data in template["cius"]:
        ciu_id = f"{_make_task_id(family, index, seed)}_{ciu_data['id']}"
        cius.append({
            "id": ciu_id,
            "type": ciu_data["type"],
            "content": ciu_data["content"],
            "token_count": max(1, len(ciu_data["content"]) // 4),
        })
        if ciu_data["necessary"]:
            necessary_ids.append(ciu_id)

    # Shuffle CIUs so order doesn't leak information
    r.shuffle(cius)

    return {
        "task_id": _make_task_id(family, index, seed),
        "category": family,
        "difficulty": "easy" if len(necessary_ids) <= 2 else "medium",
        "task": template["task"],
        "receiver_role": _family_to_role(family),
        "answer_keyword": template.get("answer_keyword", ""),
        "answer_pattern": template.get("answer_pattern", ""),
        "cius": cius,
        "ground_truth": {
            "necessary_cius": necessary_ids,
        },
    }


def _family_to_role(family: str) -> str:
    roles = {
        "debugging": "debugger",
        "configuration": "sysadmin",
        "data_transformation": "data_engineer",
        "code_repair": "developer",
        "documentation": "analyst",
    }
    return roles.get(family, "general")


def generate_dataset(
    families: Optional[List[str]] = None,
    size: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate a complete synthetic dataset.

    Each family's templates are cycled through to reach the desired size.
    Tasks are shuffled with the seed for reproducibility.
    """
    families = families or list(TASK_FAMILIES.keys())
    tasks_per_family = max(1, size // len(families))
    remainder = size - tasks_per_family * len(families)

    all_tasks = []
    for family in families:
        templates = TASK_FAMILIES.get(family, [])
        if not templates:
            continue
        for i in range(tasks_per_family + (1 if remainder > 0 else 0)):
            template = templates[i % len(templates)]
            task = generate_task(template, family, i, seed)
            all_tasks.append(task)
        if remainder > 0:
            remainder -= 1

    # Shuffle with seed
    random.Random(seed).shuffle(all_tasks)
    return all_tasks[:size]


def save_dataset(tasks: List[Dict[str, Any]], output_path: str) -> None:
    """Save dataset as JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL."""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks
