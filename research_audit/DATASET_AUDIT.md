# Dataset Audit

| Dataset | Version | Source | Number of tasks | Categories | CIUs mean | CIUs min | CIUs max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| data\synthetic\my_benchmark.jsonl | file artifact | generator or saved JSONL | 100 | {'data_transformation': 20, 'documentation': 20, 'debugging': 20, 'code_repair': 20, 'configuration': 20} | 6.0 | 6 | 6 |
| data\synthetic\swe_bench_lite.jsonl | file artifact | generator or saved JSONL | 300 | {'software_engineering': 300} | 3.0 | 3 | 3 |
| data\synthetic\synthetic_v1.jsonl | file artifact | generator or saved JSONL | 200 | {'configuration': 40, 'documentation': 40, 'data_transformation': 40, 'code_repair': 40, 'debugging': 40} | 6.0 | 6 | 6 |

SWE-bench implementation: `src/caps/datasets/swe_bench.py` loads `princeton-nlp/SWE-bench_Lite` test split through HuggingFace, shuffles by seed, selects up to the requested count, fetches up to three modified files from GitHub, and stores the gold patch in ground truth. It does not run official SWE-bench tests.

## Examples from data\synthetic\my_benchmark.jsonl

```json
[
  {
    "task_id": "data_transformation_002_s42",
    "category": "data_transformation",
    "task": "Write a function to convert a list of dicts to CSV format.",
    "ciu_examples": [
      {
        "id": "data_transformation_002_s42_ciu_sample",
        "type": "sample_data",
        "token_count": 21,
        "content_excerpt": "[{'name': 'Alice', 'age': 30, 'city': 'NYC'}, {'name': 'Bob', 'age': 25, 'city': 'LA'}]"
      },
      {
        "id": "data_transformation_002_s42_ciu_schema",
        "type": "schema",
        "token_count": 32,
        "content_excerpt": "Input: List[Dict[str, Any]]\nExample: [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\nOutput: CSV string with header row"
      },
      {
        "id": "data_transformation_002_s42_ciu_unrelated",
        "type": "irrelevant_documentation",
        "token_count": 17,
        "content_excerpt": "# Data Pipeline Architecture\nWe use Apache Airflow for orchestration."
      }
    ]
  },
  {
    "task_id": "data_transformation_001_s42",
    "category": "data_transformation",
    "task": "Write a function to flatten a nested JSON object.",
    "ciu_examples": [
      {
        "id": "data_transformation_001_s42_ciu_unrelated",
        "type": "irrelevant_documentation",
        "token_count": 11,
        "content_excerpt": "# JSON Schema Specification\nVersion: draft-07"
      },
      {
        "id": "data_transformation_001_s42_ciu_env",
        "type": "environment",
        "token_count": 10,
        "content_excerpt": "Python 3.11, no external libraries allowed"
      },
      {
        "id": "data_transformation_001_s42_ciu_constraints",
        "type": "constraints",
        "token_count": 11,
        "content_excerpt": "Handle lists by indexing: key.0, key.1, etc."
      }
    ]
  }
]
```

## Examples from data\synthetic\swe_bench_lite.jsonl

```json
[
  {
    "task_id": "sympy__sympy-17139",
    "category": "software_engineering",
    "task": "Fix this issue:\nsimplify(cos(x)**I): Invalid comparison of complex I (fu.py)\n```\r\n>>> from sympy import *\r\n>>> x = Symbol('x')\r\n>>> print(simplify(cos(x)**I))\r\nTraceback (most recent call last):\r\n  File \"<stdin>\", line 1, in <module>\r\n  File \"/home/e/se/sympy/simplify/simplify.py\", line 587, in simp",
    "ciu_examples": [
      {
        "id": "sympy__sympy-17139_ciu_desc",
        "type": "task_description",
        "token_count": 479,
        "content_excerpt": "simplify(cos(x)**I): Invalid comparison of complex I (fu.py)\n```\r\n>>> from sympy import *\r\n>>> x = Symbol('x')\r\n>>> print(simplify(cos(x)**I))\r\nTraceback (most recent call last):\r\n"
      },
      {
        "id": "sympy__sympy-17139_ciu_fu.py",
        "type": "source_code",
        "token_count": 1254,
        "content_excerpt": "# File: sympy/simplify/fu.py\n\"\"\"\nImplementation of the trigsimp algorithm by Fu et al.\n\nThe idea behind the ``fu`` algorithm is to use a sequence of rules, applied\nin what is heuri"
      },
      {
        "id": "sympy__sympy-17139_ciu_distractor",
        "type": "irrelevant_documentation",
        "token_count": 20,
        "content_excerpt": "This is unrelated documentation. It does not help solve the issue."
      }
    ]
  },
  {
    "task_id": "sympy__sympy-23191",
    "category": "software_engineering",
    "task": "Fix this issue:\ndisplay bug while using pretty_print with sympy.vector object in the terminal\nThe following code jumbles some of the outputs in the terminal, essentially by inserting the unit vector in the middle -\r\n```python\r\nfrom sympy import *\r\nfrom sympy.vector import CoordSys3D, Del\r\n\r\ninit_pri",
    "ciu_examples": [
      {
        "id": "sympy__sympy-23191_ciu_desc",
        "type": "task_description",
        "token_count": 399,
        "content_excerpt": "display bug while using pretty_print with sympy.vector object in the terminal\nThe following code jumbles some of the outputs in the terminal, essentially by inserting the unit vect"
      },
      {
        "id": "sympy__sympy-23191_ciu_pretty.py",
        "type": "source_code",
        "token_count": 1254,
        "content_excerpt": "# File: sympy/printing/pretty/pretty.py\nimport itertools\n\nfrom sympy.core import S\nfrom sympy.core.add import Add\nfrom sympy.core.containers import Tuple\nfrom sympy.core.function i"
      },
      {
        "id": "sympy__sympy-23191_ciu_distractor",
        "type": "irrelevant_documentation",
        "token_count": 20,
        "content_excerpt": "This is unrelated documentation. It does not help solve the issue."
      }
    ]
  }
]
```

## Examples from data\synthetic\synthetic_v1.jsonl

```json
[
  {
    "task_id": "configuration_026_s42",
    "category": "configuration",
    "task": "Fix the environment variable configuration for the secret key.",
    "ciu_examples": [
      {
        "id": "configuration_026_s42_ciu_code",
        "type": "source_code",
        "token_count": 22,
        "content_excerpt": "import os\n\nSECRET_KEY = os.environ['SECRET_KEY']\nDEBUG = os.environ.get('DEBUG', 'false')"
      },
      {
        "id": "configuration_026_s42_ciu_readme",
        "type": "irrelevant_documentation",
        "token_count": 12,
        "content_excerpt": "# Deployment Guide\nDeploy using Docker or Heroku."
      },
      {
        "id": "configuration_026_s42_ciu_env",
        "type": "environment",
        "token_count": 18,
        "content_excerpt": "Environment variables set:\nDEBUG=true\nDATABASE_URL=postgres://...\nPORT=8000"
      }
    ]
  },
  {
    "task_id": "documentation_027_s42",
    "category": "documentation",
    "task": "What authentication method does the API use?",
    "ciu_examples": [
      {
        "id": "documentation_027_s42_ciu_doc_relevant",
        "type": "relevant_documentation",
        "token_count": 46,
        "content_excerpt": "# Authentication\nThe API uses OAuth2 with Bearer tokens.\nTo authenticate:\n1. POST /auth/token with client_id and client_secret\n2. Include the token in Authorization header: Bearer "
      },
      {
        "id": "documentation_027_s42_ciu_doc_irrelevant",
        "type": "irrelevant_documentation",
        "token_count": 27,
        "content_excerpt": "# API Endpoints\nGET /users - List all users\nPOST /users - Create a new user\nGET /users/{id} - Get user by ID"
      },
      {
        "id": "documentation_027_s42_ciu_version",
        "type": "version",
        "token_count": 2,
        "content_excerpt": "API v3.0.0"
      }
    ]
  }
]
```