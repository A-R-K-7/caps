# CAPS Research Viva / Defense Preparation

> **Experiment**: research_f5f93acf (Pilot Mock Run)  
> **Document**: Viva questions with honest, defensible answers  
> **Warning**: Some answers disclose limitations that would disqualify the current results as final evidence

---

## Basic Questions

**Q1. What is CAPS?**

CAPS (Context-Adaptive Communication Policy Selection) is a framework for learning which pieces of information to transmit from a planning agent to a receiving agent, under token budget constraints. It uses a contextual multi-armed bandit (LinUCB) to learn, from experience, which Candidate Information Units (CIUs) are most valuable for which task/receiver contexts.

**Q2. What problem are you solving?**

In multi-agent LLM systems, the planning agent often has access to large amounts of context (code, documents, logs, constraints) but the receiving agent has limited context window and attention. Sending everything is expensive and can distract the receiver with irrelevant or misleading information. CAPS learns to select the minimum sufficient context — transmitting high-utility information while avoiding noise.

**Q3. Why is personalization needed?**

The utility of any given piece of information depends on the task, the receiver's role, and the context. A "stack trace" is essential for a debugging task but irrelevant for a documentation lookup. A static heuristic (e.g., "always send source code first") cannot adapt to this variability. CAPS learns context-conditioned utility estimates.

**Q4. What is a CIU?**

A Candidate Information Unit (CIU) is an independently selectable piece of context. Examples: a stack trace, a configuration file, a piece of source code, a documentation snippet. Each CIU has an ID, type, content, and token count. The policy scores each CIU and selects a budget-constrained subset.

**Q5. What is the synthetic v2 benchmark?**

A synthetic benchmark with 35 tasks across 7 categories (debugging, configuration, data_transformation, code_repair, documentation_lookup, multi_hop_reasoning, contradiction_resolution) at 3 difficulty levels. Each task has a set of CIUs labeled as necessary, useful, redundant, misleading, or conditional. The ground-truth labels are hidden from agents and used only by the evaluator.

---

## Technical Questions

**Q6. How does the CAPS policy work?**

CAPS uses LinUCB (Linear Upper Confidence Bound), a contextual bandit algorithm. For each CIU type, it maintains a precision matrix A_i and reward vector b_i. The UCB score for CIU i in context x is:

```
UCB_i = x^T θ_i + α sqrt(x^T A_i^{-1} x)
where θ_i = A_i^{-1} b_i
```

CIUs are ranked by UCB score and selected greedily within the token budget. After each episode, A_i and b_i are updated using the observed reward.

**Q7. What is VOI?**

Value of Information (VOI) is a per-CIU reward signal that attributes credit to individual CIUs for their contribution to task success. In CAPS-VOI, the receiver reports which CIUs it actually used. CIUs that were selected and used receive +1.0 reward; selected but unused receive −0.5. This per-CIU credit assignment allows LinUCB to learn which specific CIU types are valuable, rather than attributing all credit/blame to the entire selected set.

**Q8. Why LinUCB?**

LinUCB is appropriate because: (1) it handles the exploration-exploitation tradeoff naturally via UCB scores, (2) it is provably efficient in terms of regret bounds, (3) it is computationally cheap (no neural network, runs locally), and (4) it supports per-arm feature vectors, enabling context-conditioned learning. Alternatives like deep RL would require far more training data.

**Q9. Why embeddings?**

Embeddings provide semantic similarity features that allow the policy to generalize across tasks. Without embeddings, the only features would be structural (budget, CIU token count, type). With embeddings, the policy can score CIUs based on their semantic relevance to the current task description.

**Q10. What are the 9 LinUCB features?**

Three budget features: (1) normalized token budget, (2) planner confidence, (3) number of previous turns.
Five CIU features: (4) normalized CIU token count, (5) CIU priority score, (6) historical utility estimate, (7) historical selection rate, (8) uncertainty estimate.
One cross-feature: (9) cosine similarity between task embedding and CIU embedding.

**Q11. Why these specific baselines?**

- FullContext-Unbounded: upper bound (maximum information)
- FullContext-Budgeted: budget-constrained upper bound without intelligence
- Random: lower bound (no domain knowledge)
- Heuristic: type-priority rule baseline (existing practice)
- LLMPlanner: represents current LLM-agent paradigm
- CAPS-NoVOI: ablates VOI contribution from CAPS
- CAPS-VOI: full proposed system

Together they enable ablations of: budget constraint effect, learning effect, and VOI effect.

**Q12. Why greedy selection?**

Greedy selection by UCB score is provably near-optimal for subset selection problems (submodularity does not directly apply here, but greedy performs well empirically). Exact optimization of the budget-constrained selection problem is NP-hard in general. Greedy gives O(N log N) selection for any N CIUs.

**Q13. How is task success evaluated?**

Deterministic keyword matching: the evaluator checks if the receiver's answer contains the `answer_key` (exact word boundary match) or the first phrase of `answer_text`. No LLM is used for evaluation. This makes the evaluation reproducible and removes LLM-as-judge bias.

**Q14. What is the reward function?**

R = task_success − λ · (w₁·norm_tokens + w₂·norm_latency + w₃·norm_clarification + w₄·norm_retries)

where λ=0.5, w₁=0.4, w₂=0.2, w₃=0.2, w₄=0.2. All cost terms are normalized to [0,1] for unit consistency. The function penalizes communication cost even when the task succeeds — CAPS should find the minimum sufficient context, not just any context that works.

**Q15. Can the reward function incentivize bad behavior?**

Yes — and this is a key finding. When task success is always 0 (mock mode), the reward reduces to a pure cost penalty: R = −0.5·cost. In this case, sending nothing (0 tokens) achieves near-zero reward (only latency penalty, ≈ −0.00007), while sending all CIUs achieves −0.2. LinUCB correctly learns that sending nothing is optimal under this reward. This is a reward design weakness: the agent is penalized for communicating even when it cannot know whether communication will lead to success.

---

## Experimental Questions

**Q16. Why these budget fractions (30%, 50%, 100%)?**

The pilot covers three representative points: 100% (full budget — how does selection compare without hard constraints?), 50% (moderate constraint — requires meaningful tradeoffs), 30% (tight constraint — forces aggressive selection). A fuller campaign would add 75% and 15% for a complete curve.

**Q17. Why 3 seeds?**

Three seeds provide minimal variance estimation. The pilot design uses seeds to initialize RandomPolicy differently. Five seeds are recommended for publication. In the current mock run, most methods have zero variance across seeds because all inputs are deterministic.

**Q18. Why 7 test tasks?**

The dataset size (35 tasks) was chosen for fast iteration (no API cost). With 60/20/20 split, this gives 7 test tasks. This is acknowledged as critically underpowered for statistical testing — it is a pilot, not a final experiment.

**Q19. Why only 3 seeds, not 5?**

Cost and iteration speed in the pilot. The runner is designed to support 5 seeds with a config change. However, since seeds produce zero variance in mock mode (deterministic embeddings, no LLM randomness), additional seeds would add no information in this run.

**Q20. Why this specific train/test split ratio?**

60/20/20 is a standard split. With N=35, this gives 21 train / 7 val / 7 test. The split is fixed (seed=1) and reproducible. Validation is used to monitor training stability; test is used for final evaluation.

**Q21. Why not cross-validation?**

Cross-validation would require running the policy multiple times with different training sets, which conflicts with the sequential online learning nature of LinUCB. The bandit algorithm learns cumulatively; cross-validation would require resetting and retraining. A held-out test set is more appropriate.

**Q22. Why synthetic data?**

Synthetic data provides: (1) controlled ground-truth CIU labels (necessary/useful/misleading) that would be expensive to annotate on real data, (2) reproducibility (same tasks, same structure), (3) no privacy/copyright issues, (4) ability to vary difficulty and CIU composition systematically. Real-data experiments are needed for generalization claims.

**Q23. Why all-MiniLM-L6-v2 embeddings?**

It is a standard, well-understood, locally runnable embedding model. For the pilot, local execution is important (no API cost, no latency, reproducibility). For the real experiment, a more powerful model (e.g., text-embedding-3-large) might provide better semantic similarity features.

---

## Critical Questions

**Q24. Why is CAPS-VOI selecting zero tokens?**

This is a confirmed implementation bug, not an algorithmic finding. The greedy selector had a guard (`if score <= 0: continue`) designed for heuristic priority scores. For LinUCB, UCB scores can be negative after training with negative rewards — the guard then filters all CIUs. The root cause chain: all mock rewards are negative → LinUCB theta becomes negative → exploitation term negative → UCB < 0 → selector skips all CIUs. Fix: removed the `<= 0` guard. CAPS-VOI results must be regenerated.

**Q25. Is the mock environment realistic?**

No. Key unrealistic properties:
1. Receiver always outputs default JSON with no task answer — task success is always 0%
2. All rewards are negative — LinUCB cannot learn positive selection behavior
3. No randomness in LLM responses — seeds produce zero variance
4. Latency is 10ms (fixed) — latency penalty is negligible

Mock mode is only suitable for infrastructure validation, not for performance claims.

**Q26. Can you claim CAPS improves performance?**

No, not from the current results. You can say: "Preliminary evidence from selection quality metrics suggests CAPS-NoVOI achieves higher information precision than heuristic baselines at restricted token budgets, under a controlled synthetic evaluation. Task success comparison requires a real LLM experiment."

**Q27. What are the limitations of synthetic v2?**

1. All tasks of the same family share the same answer key (e.g., all debugging tasks have answer_key='parser_guard') — the evaluation cannot distinguish between tasks that require different reasoning
2. CIU content is template-generated — may not reflect real-world information diversity
3. Ground-truth labels (necessary/useful/misleading) are set deterministically by the generator — no human judgment
4. Task descriptions are minimal — no complex reasoning required

**Q28. Why is the sample size so small?**

It is a pilot. The dataset size (N=35) and split (7 test tasks) are designed for fast, cost-free iteration. The infrastructure and bugs needed to be validated before running expensive real LLM experiments. The correct conclusion is: "This pilot shows the infrastructure works and reveals two bugs. The full experiment requires N≥100 test tasks."

**Q29. With all success rates at 0%, what can you actually claim?**

Only infrastructure-level claims:
- The pipeline runs end-to-end correctly
- Train/test splits have no leakage
- All 63 experiment combinations completed
- Selection quality metrics (precision, recall) are computed correctly
- Heuristic selection achieves higher precision than random (descriptive, not statistically tested)

**Q30. Is CAPS novel?**

The specific combination of: per-CIU-type contextual LinUCB + VOI signal from receiver usage feedback + budget-constrained greedy selection for multi-agent communication is not directly replicated in prior work to our knowledge. The closest prior work would be: (1) RAG relevance scoring (static, not adaptive), (2) information selection in communication-efficient multi-agent systems (typically game-theoretic, not bandit-based), (3) LinUCB for recommendation (different domain). However, without a thorough related work search, novelty cannot be definitively claimed.

---

## Research Questions

**Q31. What is the hypothesis?**

H1: Adaptive communication (CAPS) achieves higher task success rate than non-adaptive baselines at equivalent token budgets.
H2: VOI-guided policy updates (CAPS-VOI) produce higher information precision than global-reward updates (CAPS-NoVOI).
H3: There exists a token budget threshold below which adaptive selection preserves more task success than sequential/random selection.

**Q32. What are the threats to validity?**

Internal validity:
- Mock LLM confound (success always 0) — addressed by next experiment
- Selector bug affecting CAPS-VOI — fixed
- Small N (7 test tasks) — addressed by next experiment

External validity:
- Synthetic data may not represent real-world tasks
- Single receiver model — results may not generalize to other LLMs
- Specific embedding model choice may affect feature quality

Construct validity:
- Precision/recall computed against synthetic ground-truth labels — may not reflect real utility
- Answer key matching is a crude proxy for task success

**Q33. What would you change in the next experiment?**

1. Real LLM for receiver (e.g., GPT-4o-mini or Llama-3)
2. N≥100 test tasks (minimum for 80% power)
3. All 7 task families balanced in test set
4. 5 seeds for variance estimation
5. Add CAPS-OracleVOI as upper bound ablation
6. Proper tuning of LinUCB alpha via validation set
7. Learning curves from train episodes
8. Budget sweep: [15%, 30%, 50%, 75%, 100%]
9. Multiple receiver models (Llama, GPT-4o-mini, Claude Haiku)
10. Human evaluation on 100-task subset

**Q34. What would it mean if CAPS-VOI underperforms CAPS-NoVOI even with a real LLM?**

It would suggest that the receiver's self-reported CIU usage is an unreliable VOI signal — the receiver may claim to have used CIUs it didn't actually benefit from, or fail to report CIUs it implicitly used. This would motivate: (1) counterfactual evaluation (remove CIU, measure performance drop), (2) oracle VOI using ground-truth labels (already implemented as CAPS-OracleVOI), (3) alternative VOI signals (e.g., LLM attention weights).

**Q35. What is the contribution relative to RAG?**

RAG retrieves documents for a query using semantic similarity. CAPS differs in: (1) selection is budget-constrained (RAG retrieves top-k regardless of cost), (2) CAPS learns adaptive selection policies from reward feedback (RAG uses fixed retrieval), (3) CAPS operates in a multi-agent communication setting (planner → receiver), not a retrieval setting, (4) CAPS uses per-CIU-type arms in LinUCB, enabling type-aware adaptation. CAPS is a reinforcement learning approach to context selection, not a retrieval approach.

**Q36. How does CAPS relate to chain-of-thought prompting?**

Chain-of-thought prompting affects the receiver's reasoning process. CAPS affects what information the receiver receives before it reasons. They are orthogonal: CAPS selects the input context; chain-of-thought structures the output reasoning. CAPS could potentially select which chain-of-thought examples to include as CIUs, making them complementary.

**Q37. What is the minimum experiment needed for a conference paper?**

- Real LLM (GPT-4o-mini or Llama-3-8B)
- N≥150 test tasks (all 7 families, 3 difficulties, ~7 tasks/cell)
- 5 seeds
- 5 budget fractions
- Formal paired bootstrap confidence intervals
- All 7 baselines including CAPS-OracleVOI
- Learning curve showing policy improvement over training episodes
- Ablation: CAPS-VOI vs NoVOI vs OracleVOI
- Estimated: 2-3 days of GPU/API compute

**Q38. What if the LinUCB never converges?**

With 21 training episodes and a 9-dimensional feature space, LinUCB is very unlikely to converge. Minimum sample for meaningful LinUCB learning is roughly 2–3 times the feature dimension per arm, so ~20–30 episodes per CIU type. With 7 CIU types, we need ~140–210 training episodes minimum. This is 7–10x more than the current 21 episodes.

**Q39. Could the reward function cause the optimal policy to never communicate?**

Yes, under certain conditions. If task success is always 0 (as in mock mode), then the optimal policy under R = 0 − λ·cost is to minimize cost, which means sending nothing. This is a pathological optimum. In real mode, task success creates positive reward that offsets cost, making communication worthwhile. The reward function is valid for real LLM experiments but problematic for mock mode.

**Q40. What is the most important single change you could make to the research design?**

Run with a real LLM receiver. This single change would: (1) make task success meaningful (non-zero), (2) provide positive reward signals to LinUCB, (3) enable valid comparisons across all methods, (4) allow VOI signals from actual receiver usage, and (5) support statistical significance testing. Everything else (larger N, more seeds, longer training) is secondary to having a meaningful reward signal.
