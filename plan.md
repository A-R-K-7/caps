# AI Engineering Prompt: Build the CAPS Research Framework (Google AI Pro Edition)

You are a senior AI systems engineer and ML researcher. Your task is to build a **research-grade implementation** of the **Communication-Adaptive Policy System (CAPS)** for adaptive multi-agent LLM communication.

This project is intended for an undergraduate engineering research project with publication potential. The implementation must prioritize:

* Modular architecture
* Reproducibility
* Clean experimentation
* Extensibility
* Strong software engineering practices

---

# Core Research Question

Can an LLM agent **learn what information should be communicated** to another agent instead of always sending every available piece of context?

The focus of this project is **learning communication policies**, **not improving the reasoning ability of the LLM itself**.

The LLM should be treated as a black-box reasoning engine.

---

# Important Constraints

## Available Resources

* Google AI Pro (Gemini)
* Python
* Single GPU (optional)
* One LLM instance at a time

Do **NOT** design the system assuming multiple concurrent LLM instances.

Agents are **logical roles**, not separate models.

Example:

Planner Agent
↓

Gemini

↓

Receiver Agent
↓

Gemini

Both planner and receiver use the **same Gemini model** with different system prompts.

Execution is entirely **sequential**.

---

# High-Level Architecture

```text
                User Task
                     │
                     ▼
              Planner Agent
                     │
     Generate Candidate Information Units
                     │
                     ▼
         Communication Policy (CAPS)
                     │
      Select Important Information
                     │
                     ▼
             Receiver Agent
                     │
        Success / Failure / Retry
                     │
                     ▼
       Communication Evaluator
                     │
 Counterfactual Replay (Sequential)
                     │
                     ▼
          Policy Learning Module
```

The Communication Policy is the research contribution.

The LLM is only responsible for reasoning.

---

# Technology Stack

Language

* Python 3.12+

Package Manager

* uv

LLM

* Google Gen AI Python SDK
* Gemini models through Google AI Pro

Agent Framework

* LangGraph (preferred)
* Avoid unnecessary LangChain abstractions

Embeddings

* sentence-transformers
* or Gemini embeddings if available

Database

* SQLite
* SQLAlchemy

Configuration

* Pydantic Settings

Logging

* structlog

Experiment Tracking

* MLflow

Testing

* pytest

Formatting

* Ruff
* Black

Typing

* Full type hints

---

# Project Structure

```text
caps/

    planner/
    receiver/
    policy/
    evaluator/
    replay/
    ciu/
    metrics/
    experiments/
    prompts/
    storage/
    benchmark/
    utils/

configs/

tests/

scripts/

docs/
```

Each component must communicate through interfaces.

Avoid tight coupling.

---

# Candidate Information Units (CIUs)

Never manipulate raw prompt strings.

Represent every piece of information as a CIU.

Each CIU contains

* id
* type
* content
* token_count
* embedding
* metadata

Example CIUs

* task description
* repository file
* stack trace
* execution logs
* README
* retrieved documentation
* API specification
* environment variables
* previous messages
* unit tests

---

# Planner Agent

Responsibilities

* Understand the user task
* Build the list of available CIUs
* Build planner state
* Call the Communication Policy
* Send only selected CIUs

Planner must never implement communication heuristics.

Those belong to the Communication Policy.

---

# Receiver Agent

The receiver receives

* task
* selected CIUs

The receiver performs the assigned work.

If required information is missing, it must explicitly call

ask_clarification(question)

instead of hallucinating.

Every clarification request must be logged.

---

# Communication Policy

This is the primary research contribution.

Design a generic interface

```python
select(state, cius) -> selected_cius

update(feedback)
```

Implement policies incrementally.

Version 1

Always Send Everything

Version 2

Random Selection

Version 3

Token Budget Greedy

Version 4

Heuristic Importance

Version 5

Logistic Regression

Version 6

Contextual Bandit (LinUCB)

Future policies should be pluggable.

---

# State Representation

State should include

* task embedding
* receiver embedding
* planner confidence
* token budget
* latency budget
* CIU embeddings

Keep dimensions configurable.

---

# Action Representation

Do **NOT** model the action as every possible subset of CIUs.

Instead

1. Score every CIU independently.
2. Rank them.
3. Keep adding CIUs until

   * token budget exhausted
   * score below threshold

This scales much better and matches production retrieval systems.

---

# Reward

Reward =

Task Success

− λ × Communication Cost

Communication Cost includes

* tokens
* latency
* clarification requests
* retries

Weights must be configurable.

---

# Communication Evaluator

Implement a Communication Evaluator that estimates the Value of Information (VOI) of every CIU.

Methods

## Counterfactual Replay

Replay receiver execution while removing one CIU at a time.

Measure

* task success
* clarification count
* latency
* retries

Estimate the causal contribution of each CIU.

Replay is **sequential**, not parallel.

Cache results whenever possible to reduce API usage.

---

# Metrics

Automatically compute

* Task Success
* Token Usage
* Latency
* Retry Count
* Clarification Count
* Information Precision
* Information Recall
* Average CIUs Sent
* VOI Distribution
* Policy Regret
* Communication Cost

Export every experiment to CSV.

---

# Logging

Log every delegation

Task

↓

Available CIUs

↓

Planner State

↓

Policy Scores

↓

Selected CIUs

↓

Receiver Output

↓

Clarifications

↓

Reward

↓

Policy Update

Store structured JSON logs.

---

# Benchmarks

Phase 1

Create a synthetic debugging benchmark.

Tasks

* missing imports
* stack traces
* incorrect configuration
* dependency errors
* failing unit tests

Phase 2

Support

* SWE-bench Lite
* WebArena

Benchmark implementation must remain independent of CAPS.

---

# Configuration

Everything must be configurable

* Gemini model
* embedding model
* token budget
* replay frequency
* reward weights
* policy type
* logging level

---

# Experiment Pipeline

Implement reproducible experiments comparing

1. Always Send Everything
2. Random
3. Greedy
4. Heuristic
5. Contextual Bandit

Automatically generate

* CSV
* Markdown tables
* publication-ready plots

---

# Development Roadmap

## Phase 1

* Project skeleton
* Configuration
* Logging
* CIU abstraction

## Phase 2

* Planner
* Receiver
* Baseline policy

## Phase 3

* Metrics
* Reward function
* Experiment runner

## Phase 4

* Counterfactual Replay

## Phase 5

* Contextual Bandit policy

## Phase 6

* Full benchmark evaluation

Do not proceed to the next phase until the current phase is complete and fully tested.

---

# Code Quality Requirements

* SOLID principles
* Full type hints
* Comprehensive documentation
* Unit tests
* Minimal dependencies
* Clean interfaces
* No circular imports
* Reproducible experiments

Prioritize readability and maintainability over clever optimizations.

---

# Deliverables

For every completed phase:

1. Implement the feature.
2. Write unit tests.
3. Explain the architecture.
4. Explain design decisions.
5. Document assumptions.
6. Update project documentation.
7. Commit only after all tests pass.

Work incrementally. Do not skip phases. Do not implement placeholders when a functional implementation is possible. Build a research framework that can support future experimentation without major architectural changes.
