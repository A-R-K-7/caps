# FULL RESEARCH AUDIT AND EXPERIMENTAL EVALUATION

**Project Investigated:** Budget-Constrained Online Regression Test Selection for Unseen Python Projects
**Actual Implementation Found:** CAPS Research Framework (Communication-Adaptive Policy System)

---

## 1. Executive Summary

**CRITICAL FINDING:** The provided repository (`C:\Users\aksha\OneDrive\Documents\Research Paper`) **does not implement** "Budget-Constrained Online Regression Test Selection for Unseen Python Projects". 

Instead, the entire repository implements an unrelated project: the **Communication-Adaptive Policy System (CAPS)**, which studies how LLM agents (Planners and Receivers) learn communication policies (via Contextual Bandits) to optimize token budgets and reduce clarification turns. 

There is **zero trace** of:
- BugsInPy integration
- Regression test selection (TCP/RTS) algorithms
- Coverage extraction modules
- Test execution engines

Because the implementation does not match the proposed research, the research claim for Budget-Constrained Online Regression Test Selection is **completely unsupported by the codebase**. I have audited the codebase that *is* present (CAPS) against the questions asked where applicable.

---

## 2. Current System Reconstruction (What is ACTUALLY implemented)

The system implemented is an adaptive LLM communication pipeline (CAPS).

### State
The state consists of embeddings of Candidate Information Units (CIUs) for multi-agent delegation (e.g., Task Descriptions, Code Snippets), Planner confidence, and budget state, encoded for a LinUCB bandit.

### Action
The policy selects a binary vector (subset of CIUs) to transmit from a Planner agent to a Receiver agent.

### Transition
The Receiver agent executes the task using the transmitted context, potentially making API calls or asking for clarifications.

### Reward
Calculated via `src/caps/evaluation/reward.py`, balancing task success against communication overhead (token usage, clarification turns).

### Learning
The system uses a Contextual Bandit (`src/caps/bandits/linucb.py`) to update the policy.

### Budget
Enforced via a token budget allocator (`src/caps/llm/budget.py`), not a test-execution time/count budget.

### Ground truth
Evaluated by an Evaluator LLM (`src/caps/agents/evaluator.py`), not by executing deterministic Python regression tests.

### Training / Testing Data
Synthetic datasets (`synthetic_v2.py`) and SWE-bench subsets, not historical software test failures.

**Diagram of Actual Pipeline:**
```text
Planner (State Encoding)
 ↓
Contextual Bandit (Selects Information Units)
 ↓
Receiver Agent (Executes Task)
 ↓
Evaluator Agent (Computes Success/Overhead)
 ↓
Reward Calculation
 ↓
Bandit Update
```

---

## 3. Is the System Really "Adaptive"?

**For CAPS:** Yes, the LinUCB bandit receives reward feedback after each episode and updates its weights dynamically for the next selection. 
**For Regression Test Selection:** **NOT IMPLEMENTED.**

---

## 4. Leakage Audit

**LEAKAGE FOUND** (in the context of test selection)
Because the dataset is based on synthetic tasks and SWE-bench, if this were evaluating test selection, any LLM pre-training on SWE-bench repositories constitutes massive data leakage.

**NO LEAKAGE FOUND** (in the context of CAPS bandit learning)
The LinUCB bandit legitimately updates sequentially.

---

## 5. Dataset Audit (BugsInPy)

**BugsInPy is entirely missing from the repository.**

| Project | Bugs attempted | Bugs successful | Bugs failed | Reason |
| ------- | -------------: | --------------: | ----------: | ------ |
| ALL | 0 | 0 | 0 | **NOT IMPLEMENTED** |

---

## 6. Verify the Ground Truth

**BROKEN.**
There is no deterministic ground truth for test failures. The repository uses an LLM-as-a-judge (`evaluator.py`) to determine task success, which is inherently stochastic and not equivalent to a deterministic test failure.

---

## 7. Experimental Setup

The experiments currently implemented (`runner.py`, `caps_experiments.db`) run synthetic delegation tasks between LLMs. 
**Regression test selection experiments are NOT RUN.**

---

## 8. Baselines

### B0 — Full Test Suite
**NOT IMPLEMENTED**

### B1 — Random
**IMPLEMENTED** (Random CIU selection in CAPS)

### B2 — Coverage-Based
**NOT IMPLEMENTED**

### B3 — Cost-Aware Coverage
**NOT IMPLEMENTED**

### B4 — Historical Failure
**NOT IMPLEMENTED**

### B5 — Static ML
**NOT IMPLEMENTED**

### B6 — Existing adaptive/RL baseline
**NOT IMPLEMENTED**

### Proposed method
**NOT IMPLEMENTED** (Adaptive regression test selection is missing; Adaptive LLM communication is implemented).

---

## 9. Budget Experiments

**NOT IMPLEMENTED** for test execution.
The budget experiments implemented (`budget_fraction`) apply strictly to LLM context window tokens, not test suite execution.

---

## 10. Cross-Project Generalization

**CROSS-PROJECT EVALUATION**
Status: **NOT IMPLEMENTED** for software projects.
The system trains and tests on randomized synthetic task categories.

---

## 11. Compare Static vs Adaptive

**NOT IMPLEMENTED** for test selection.

---

## 12. Feature Ablation

**NOT IMPLEMENTED**
Coverage, dependency information, historical test failures, and execution costs are not extracted or utilized in the current codebase.

---

## 13. Metrics

### Fault Detection Rate
**NOT IMPLEMENTED**

### Test Reduction
**NOT IMPLEMENTED**

### Time to Detection
**NOT IMPLEMENTED**

### Tests to Detection
**NOT IMPLEMENTED**

### Budget Success Rate
**NOT IMPLEMENTED**

### APFD / APFDc
**NOT IMPLEMENTED**

---

## 14. Statistical Analysis

Cannot be performed as the underlying regression test selection data does not exist in the repository.

---

## 15. Per-Project Analysis

**NOT IMPLEMENTED**

---

## 16. Failure Analysis

**BROKEN**
No regression test trajectories exist to analyze.

---

## 17. Research Claim Audit

### Claim A: "The proposed policy improves fault detection."
**NOT SUPPORTED.** No fault detection system is implemented.

### Claim B: "The policy reduces testing cost."
**NOT SUPPORTED.** No test execution system is implemented.

### Claim C: "The policy generalizes to unseen projects."
**NOT SUPPORTED.** No project-level test data exists.

### Claim D: "Online adaptation is responsible for the improvement."
**NOT SUPPORTED.** No static ML baseline for TCP is present for comparison.

### Claim E: "The method works under strict testing budgets."
**NOT SUPPORTED.**

### Claim F: "The method is better than existing approaches."
**NOT SUPPORTED.**

---

## 18. Novelty Audit

| Paper | What it does | Overlap | Difference | Threatens novelty? |
| ----- | ------------ | ------- | ---------- | ------------------ |
| Spieker et al. (2017) | RL for test selection | High | N/A (Code missing) | Yes |
| Bertolino et al. (2020) | Contextual bandits for TCP | High | N/A (Code missing) | Yes |
| Yaraghi et al. (2022) | Cross-project ML TCP | High | N/A (Code missing) | Yes |

**What is the narrowest defensible research contribution of this project?**
Based on the codebase provided, the research contribution is entirely in **LLM inter-agent communication efficiency (CAPS)**, not regression test selection.

---

## 19. Research Validity

### Internal validity
**Critical Problem.** The codebase does not implement the proposed research.

### External validity
**Critical Problem.** 

### Construct validity
**Critical Problem.**

### Reproducibility
**Critical Problem.** The stated experiments cannot be reproduced because the code for them does not exist.

---

## 20. Final Result Table

| Method          | Budget | Fault Detection | Test Reduction | Time to Detection | Tests to Detection |
| --------------- | -----: | --------------: | -------------: | ----------------: | -----------------: |
| Full Suite      |        | NOT RUN         | NOT RUN        | NOT RUN           | NOT RUN            |
| Random          |        | NOT RUN         | NOT RUN        | NOT RUN           | NOT RUN            |
| Coverage        |        | NOT IMPLEMENTED | NOT IMPLEMENTED| NOT IMPLEMENTED   | NOT IMPLEMENTED    |
| Cost-Aware      |        | NOT IMPLEMENTED | NOT IMPLEMENTED| NOT IMPLEMENTED   | NOT IMPLEMENTED    |
| Historical      |        | NOT IMPLEMENTED | NOT IMPLEMENTED| NOT IMPLEMENTED   | NOT IMPLEMENTED    |
| Static ML       |        | NOT IMPLEMENTED | NOT IMPLEMENTED| NOT IMPLEMENTED   | NOT IMPLEMENTED    |
| Adaptive Policy |        | NOT IMPLEMENTED | NOT IMPLEMENTED| NOT IMPLEMENTED   | NOT IMPLEMENTED    |

---

## 21. Overall Verdict

### D — Invalid research result
**Major methodological, leakage, data, or implementation problems prevent meaningful conclusions.**

**Explanation:** The implementation provided in the workspace (`src/caps`) is fundamentally mismatched with the proposed research project ("Budget-Constrained Online Regression Test Selection for Unseen Python Projects"). The codebase implements a framework for optimizing LLM inter-agent communication (CAPS). Without any code handling BugsInPy, test execution, coverage extraction, or regression test selection, it is impossible to validate any claims regarding test selection efficiency.

---

## 22. What We Should Do Next

### P0 — Critical
- **Upload the correct repository:** The current workspace contains the CAPS LLM agent communication project, not the Test Selection project. The proper codebase containing the BugsInPy integration and TCP RL/Bandit agent must be provided before any audit can proceed.
