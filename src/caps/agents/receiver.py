"""
Receiver Agent.

The receiver gets ONLY the task description and selected CIUs.
It must NOT see unselected CIUs — this is critical for experimental validity.

Returns structured output indicating:
    - status: success/failure/needs_clarification
    - answer: the actual response
    - used_ciu_ids: which CIUs it reports using (self-report, not ground truth)
    - missing_information: what it thinks is missing
    - confidence: self-assessed confidence
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from caps.communication.ciu import CIU
from caps.llm import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

RECEIVER_PROMPT_VERSION = "receiver_v1"


class ReceiverOutput(BaseModel):
    """Structured receiver response."""
    status: str = "success"  # success | failure | needs_clarification
    answer: str = ""
    used_ciu_ids: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    llm_usage: Optional[Any] = None


class ReceiverAgent:
    """
    Receiver agent that attempts to solve the task using only selected CIUs.

    The receiver explicitly identifies which CIUs it used.
    This self-report is one evaluation signal, NOT ground truth.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.llm = llm_provider
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def execute(
        self,
        task: str,
        selected_cius: List[CIU],
        receiver_role: str = "general",
    ) -> ReceiverOutput:
        """
        Attempt to solve the task using only the provided CIUs.

        The receiver MUST NOT see unselected CIUs.
        """
        # Build context from selected CIUs
        context_parts = []
        ciu_ids_in_context = []
        for ciu in selected_cius:
            context_parts.append(
                f"--- [ID: {ciu.id}] [TYPE: {ciu.type}] ---\n{ciu.content}"
            )
            ciu_ids_in_context.append(ciu.id)

        context_str = "\n\n".join(context_parts) if context_parts else "(No context provided)"

        # Truncate task description if it's absurdly long (approx. 4000 tokens = 16000 chars)
        safe_task = task[:16000] + "\n...[TRUNCATED]" if len(task) > 16000 else task

        prompt = (
            f"TASK:\n{safe_task}\n\n"
            f"PROVIDED CONTEXT:\n{context_str}\n\n"
            f"Instructions:\n"
            f"1. Solve the task using ONLY the provided context.\n"
            f"2. If critical information is missing, indicate what is needed.\n"
            f"3. Respond with a JSON object:\n"
            f'{{\n'
            f'  "status": "success" or "failure" or "needs_clarification",\n'
            f'  "answer": "your solution/answer",\n'
            f'  "used_ciu_ids": ["list of CIU IDs you actually used"],\n'
            f'  "missing_information": ["list of what you need but do not have"],\n'
            f'  "confidence": 0.0 to 1.0\n'
            f'}}'
        )

        system_prompt = (
            f"You are a {receiver_role} agent. "
            f"Solve the task using ONLY the provided context. "
            f"Do not hallucinate information that was not provided. "
            f"If you cannot solve the task with the given context, say so clearly. "
            f"Always respond with valid JSON."
        )

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=800,
            response_schema=ReceiverOutput,
            seed=self.seed,
        )

        return self._parse_response(response, ciu_ids_in_context)

    def _parse_response(
        self,
        response: LLMResponse,
        available_ciu_ids: List[str],
    ) -> ReceiverOutput:
        """Parse the LLM response into a structured ReceiverOutput."""
        try:
            parsed = json.loads(response.text)
            parsed["llm_usage"] = response.token_usage
            output = ReceiverOutput(
                status=parsed.get("status", "success"),
                answer=parsed.get("answer", response.text),
                used_ciu_ids=[
                    cid for cid in parsed.get("used_ciu_ids", [])
                    if cid in available_ciu_ids
                ],
                missing_information=parsed.get("missing_information", []),
                confidence=float(parsed.get("confidence", 0.5)),
            )
            return output
        except (json.JSONDecodeError, Exception):
            # If parsing fails, treat the raw text as the answer
            return ReceiverOutput(
                status="success",
                answer=response.text,
                used_ciu_ids=[],
                missing_information=[],
                confidence=0.3,
                llm_usage=response.token_usage,
            )
