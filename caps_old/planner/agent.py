import json
from typing import List

from pydantic import BaseModel, Field

from caps.ciu.embedder import CIUEmbedder
from caps.ciu.models import CIU
from caps.policy.models import PlannerState
from caps.utils.llm import BaseLLMClient
from configs.settings import settings


class ExtractedCIU(BaseModel):
    type: str = Field(description="The type of information (e.g., file, log, instruction)")
    content: str = Field(description="The actual content of the information unit")

class ExtractionSchema(BaseModel):
    cius: List[ExtractedCIU]

class PlannerAgent:
    def __init__(self, llm_client: BaseLLMClient, temperature: float = None, seed: int = None):
        self.llm = llm_client
        self.embedder = CIUEmbedder()
        self.temperature = temperature if temperature is not None else settings.temperature
        self.seed = seed if seed is not None else settings.seed
        
    def extract_state(self, task: str, raw_context: str) -> tuple[PlannerState, List[CIU]]:
        """
        Parses raw context into discrete CIUs and formulates the MDP state.
        """
        system_instruction = (
            "You are a context parsing agent. Your job is to extract discrete units "
            "of information from the raw workspace context. Break the context down "
            "into logical pieces like individual files, stack traces, or instructions."
        )
        
        prompt = f"TASK:\n{task}\n\nRAW CONTEXT:\n{raw_context}"
        
        # We enforce JSON extraction of CIUs
        text_resp, _ = self.llm.generate_content(
            system_instruction=system_instruction,
            prompt=prompt,
            response_schema=ExtractionSchema,
            temperature=self.temperature,
            seed=self.seed
        )
        
        try:
            parsed = json.loads(text_resp)
            extracted = parsed.get("cius", [])
        except json.JSONDecodeError:
            extracted = []
            
        available_cius = []
        for ex in extracted:
            content = ex.get("content", "")
            # Naive token count approximation: 1 token ~ 4 chars
            token_count = len(content) // 4 
            ciu = CIU(type=ex.get("type", "unknown"), content=content, token_count=token_count)
            available_cius.append(ciu)
            
        # Embed everything
        available_cius = self.embedder.embed(available_cius)
        
        # Simple task embedding (in practice, might want a separate task embedder)
        # Using a dummy CIU to embed the task text
        task_dummy = CIU(type="task", content=task, token_count=len(task)//4)
        task_dummy = self.embedder.embed([task_dummy])[0]
        
        state = PlannerState(
            task_embedding=task_dummy.embedding,
            planner_confidence=1.0,
            token_budget=settings.token_budget,
            latency_budget=settings.latency_budget_ms,
            available_cius=available_cius
        )
        
        return state, available_cius
