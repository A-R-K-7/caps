"""
Planner Agent.

The planner receives the complete task context.
It does NOT select CIUs — that is the policy's job.
It:
    1. Understands the task
    2. Identifies the receiver role
    3. Assigns confidence
    4. Returns structured output
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from caps.communication.ciu import CIU
from caps.communication.state import CommunicationState
from caps.embeddings import EmbeddingProvider
from caps.llm import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

PLANNER_PROMPT_VERSION = "planner_v1"


class PlannerOutput(BaseModel):
    """Structured output from the planner LLM."""
    task_summary: str = ""
    receiver_role: str = "general"
    subtask: str = ""
    confidence: float = 0.8
    task_category: str = "unknown"
    difficulty: str = "medium"


class PlannerAgent:
    """
    Planner agent that constructs the communication state.

    For synthetic tasks: uses pre-defined CIUs from the benchmark.
    For real tasks: could use LLM to decompose context into CIUs.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.llm = llm_provider
        self.embedder = embedding_provider
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def build_state(
        self,
        task: str,
        cius: List[CIU],
        token_budget: int = 4000,
        receiver_role: str = "general",
        task_category: str = "unknown",
        difficulty: str = "medium",
        planner_confidence: float = 0.8,
        receiver_model: str = "",
    ) -> CommunicationState:
        """
        Build the communication state from task info and CIUs.

        For synthetic benchmark: we already have CIUs from the dataset.
        For real tasks: CIUs would be extracted by the planner LLM.
        """
        # Embed task
        task_embedding = self.embedder.embed_single(task)

        # Embed CIUs that don't have embeddings yet
        cius_to_embed = [c for c in cius if c.embedding is None]
        if cius_to_embed:
            texts = [c.content for c in cius_to_embed]
            embeddings = self.embedder.embed(texts)
            for ciu, emb in zip(cius_to_embed, embeddings):
                ciu.embedding = emb

        return CommunicationState(
            task_embedding=task_embedding,
            task_category=task_category,
            task_difficulty=difficulty,
            task_length=len(task),
            planner_confidence=planner_confidence,
            receiver_role=receiver_role,
            receiver_model=receiver_model,
            token_budget=token_budget,
            available_cius=cius,
        )

    def plan_with_llm(
        self,
        task: str,
        raw_context: str,
    ) -> PlannerOutput:
        """
        Use LLM to analyze the task and produce structured planning output.
        Used for real (non-synthetic) tasks where CIUs need extraction.
        """
        prompt = (
            f"Analyze the following task and context.\n\n"
            f"TASK:\n{task}\n\n"
            f"CONTEXT:\n{raw_context}\n\n"
            f"Provide a JSON response with:\n"
            f"- task_summary: brief summary\n"
            f"- receiver_role: who should handle this (debugger/sysadmin/developer/analyst/general)\n"
            f"- subtask: specific subtask\n"
            f"- confidence: your confidence 0.0-1.0\n"
            f"- task_category: category (debugging/configuration/data_transformation/code_repair/documentation)\n"
            f"- difficulty: easy/medium/hard"
        )

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a task analysis agent. Analyze the task and provide structured output.",
            model=self.model,
            temperature=self.temperature,
            response_schema=PlannerOutput,
            seed=self.seed,
        )

        try:
            parsed = json.loads(response.text)
            return PlannerOutput(**parsed)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Planner output parse error: {e}")
            return PlannerOutput(task_summary=task[:100])
