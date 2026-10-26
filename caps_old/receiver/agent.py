from caps.ciu.models import CIU
from caps.utils.llm import BaseLLMClient
from google.genai import types

def ask_clarification(question: str) -> str:
    """
    Call this if you need more information to solve the task.
    """
    pass

class ReceiverResult:
    def __init__(self, answer: str, asked_clarification: bool, clarification_question: str = None):
        self.answer = answer
        self.asked_clarification = asked_clarification
        self.clarification_question = clarification_question

class ReceiverAgent:
    def __init__(self, llm_client: BaseLLMClient, temperature: float = None, seed: int = None):
        from configs.settings import settings
        self.llm = llm_client
        self.temperature = temperature if temperature is not None else settings.temperature
        self.seed = seed if seed is not None else settings.seed
        
    def execute(self, task: str, selected_cius: list[CIU]) -> ReceiverResult:
        """
        Attempts to solve the task using only the provided CIUs.
        """
        system_instruction = (
            "You are a problem solving agent. You have been provided with a task "
            "and a limited set of information (context). You must solve the task "
            "using ONLY this context. If the context is missing critical files or "
            "data required to solve the task, you MUST call the ask_clarification tool."
        )
        
        context_str = "\n\n".join([f"--- [TYPE: {c.type}] ---\n{c.content}" for c in selected_cius])
        prompt = f"TASK:\n{task}\n\nPROVIDED CONTEXT:\n{context_str}"
        
        # Define the tool using genai types
        clarification_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="ask_clarification",
                    description="Call this if you need more information to solve the task.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "question": types.Schema(type=types.Type.STRING)
                        },
                        required=["question"]
                    )
                )
            ]
        )
        
        text_resp, function_calls = self.llm.generate_content(
            system_instruction=system_instruction,
            prompt=prompt,
            tools=[clarification_tool],
            temperature=self.temperature,
            seed=self.seed
        )
        
        asked = False
        question = None
        
        # Check if the model invoked the clarification tool
        for call in function_calls:
            if call.name == "ask_clarification":
                asked = True
                # Safely extract args
                args = getattr(call, "args", {})
                if isinstance(args, dict):
                    question = args.get("question", "")
                else:
                    # Depending on SDK version, it might be a structured object
                    question = getattr(args, "question", "")
                break
                
        return ReceiverResult(
            answer=text_resp if text_resp else "Clarification requested.",
            asked_clarification=asked,
            clarification_question=question
        )
