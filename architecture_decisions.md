# CAPS Architecture Decisions & Deviations

This document tracks the architectural decisions made that deviate from the original `plan.md` specification. These deviations were introduced to ensure the framework remains mathematically sound, scalable, and suitable for rigorous machine learning research. These notes should be referenced when writing the methodology section of the research paper.

## 1. Evaluator Mechanism: Self-Reflection vs. Counterfactual Replay

*   **Original Plan**: The evaluator was intended to use Leave-One-Out (LOO) Counterfactual Replay to estimate the Value of Information (VOI) of each Candidate Information Unit (CIU).
*   **The Problem**: 
    *   **Scalability**: Replaying the receiver execution by removing one CIU at a time requires $O(N)$ LLM inferences per interaction (where $N$ is the number of selected CIUs). Given the cost and latency of LLM APIs (like Gemini Pro), this bottleneck makes large-scale benchmarking infeasible.
    *   **Validity**: LOO evaluation is statistically flawed when dealing with redundant information. If the policy sends two CIUs that contain the exact same necessary context, removing either one individually will not negatively impact task success. Both CIUs would incorrectly receive a VOI of 0.
*   **New Architecture**: We use **LLM Self-Reflection Attribution** as the primary evaluator. The Receiver LLM is prompted post-execution to explicitly cite which CIUs contributed to the solution. This requires only 1 additional inference call ($O(1)$) and serves as a proxy for VOI. Counterfactual Replay is retained only as an optional, computationally expensive toggle.

## 2. Action Space and Policy Representation

*   **Original Plan**: The policy scores every CIU independently and ranks them, filling up a token budget.
*   **The Problem**: This assumes the value of each CIU is independent and additive. In reality, software engineering contexts are highly synergistic. For instance, sending a stack trace is heavily dependent on also sending the corresponding source code file. An independent pointwise ranking policy fails to capture these cross-CIU dependencies.
*   **New Architecture**: While retaining the independent pointwise scoring as the baseline (Versions 1-6), the state representation and framework architecture are designed to allow for **cross-CIU features** (e.g., embedding similarity between CIUs) and enable future autoregressive or subset-selection policies.

## 3. Reward Shaping and the "Send Nothing" Degeneracy

*   **Original Plan**: The reward function was defined purely as `Task Success - (λ × Communication Cost)`.
*   **The Problem**: In complex problem-solving domains (e.g., SWE-bench), initial success rates for a randomly initialized policy or naive baseline are very low. If task success is sparse (mostly 0) and communication incurs a strict penalty, an RL agent or contextual bandit will rapidly learn the trivial optimal policy: **send nothing** (Cost = 0) to avoid negative rewards.
*   **New Architecture**: We introduced **Reward Shaping**. Intermediate rewards are issued for partial progress (e.g., successful API parsing, valid clarification requests, or identifying correct files). Alternatively, the communication cost is normalized against a baseline to prevent early policy collapse before it learns to solve the task.

## 4. Framework Choice: Pure Python vs. LangGraph

*   **Original Plan**: The architecture was suggested to be built using LangGraph to manage the agent states.
*   **The Problem**: LangGraph's `StateGraph` abstraction is powerful for cyclic, open-ended multi-agent conversations, but CAPS is a strictly sequential reinforcement learning environment (`Planner -> Policy -> Receiver -> Evaluator`). Using LangGraph obscures the standard MDP (Markov Decision Process) loop, complicating the implementation of ML algorithms like LinUCB or Policy Gradients, and reducing research transparency.
*   **New Architecture**: We replaced LangGraph with a **clean, pure Python sequential RL loop**. This aligns with standard ML research practices (similar to OpenAI Gym or DeepMind Acme), maximizing transparency, reproducibility, and flexibility for policy integration.
