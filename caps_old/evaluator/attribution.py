from pydantic import BaseModel, Field
from typing import List, Dict
import json
from caps.utils.llm import BaseLLMClient
from caps.ciu.models import CIU

class EvaluatedCIU(BaseModel):
    id: str = Field(description="The ID of the CIU")
    utilized: bool = Field(description="Whether this CIU was explicitly utilized by the receiver to form the answer")

class AttributionSchema(BaseModel):
    attributions: List[EvaluatedCIU]

class SelfReflectionEvaluator:
    def __init__(self, llm_client: BaseLLMClient, temperature: float = None, seed: int = None):
        from configs.settings import settings
        self.llm = llm_client
        self.temperature = temperature if temperature is not None else settings.temperature
        self.seed = seed if seed is not None else settings.seed
        
    def evaluate(self, task: str, selected_cius: List[CIU], receiver_answer: str) -> Dict[str, float]:
        """
        Calculates the Value of Information (VOI) for each selected CIU using LLM Self-Reflection.
        Returns a dictionary mapping ciu_id -> VOI score.
        """
        system_instruction = (
            "You are an objective judge evaluating how an LLM agent solved a task. "
            "You will be given the original task, the context that was provided to the agent, "
            "and the final answer the agent generated. "
            "Your job is to explicitly identify which pieces of context (CIUs) were ACTUALLY utilized "
            "to produce the answer."
        )
        
        context_str = "\n\n".join([f"--- [ID: {c.id}] [TYPE: {c.type}] ---\n{c.content}" for c in selected_cius])
        
        prompt = (
            f"TASK:\n{task}\n\n"
            f"PROVIDED CONTEXT:\n{context_str}\n\n"
            f"AGENT'S GENERATED ANSWER:\n{receiver_answer}\n\n"
            "Analyze the answer. Which CIU IDs were explicitly utilized? Output JSON matching the schema."
        )
        
        text_resp, _ = self.llm.generate_content(
            system_instruction=system_instruction,
            prompt=prompt,
            response_schema=AttributionSchema,
            temperature=self.temperature,
            seed=self.seed
        )
        
        # Initialize all selected CIUs with VOI = 0.0
        voi_scores = {ciu.id: 0.0 for ciu in selected_cius}
        
        try:
            parsed = json.loads(text_resp)
            attributions = parsed.get("attributions", [])
            for attr in attributions:
                ciu_id = attr.get("id")
                utilized = attr.get("utilized", False)
                if ciu_id in voi_scores and utilized:
                    voi_scores[ciu_id] = 1.0
        except Exception:
            pass # On failure, assume no attribution was made
            
        return voi_scores
