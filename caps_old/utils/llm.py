import abc
import json
import sqlite3
import hashlib
from google import genai
from google.genai import types
from configs.settings import settings
import structlog
import os

logger = structlog.get_logger()

class BaseLLMClient(abc.ABC):
    @abc.abstractmethod
    def generate_content(self, system_instruction: str, prompt: str, tools: list = None, response_schema=None, temperature: float = 0.0, seed: int = 42) -> tuple[str, list]:
        """
        Generates content from the LLM.
        Returns a tuple: (text_response, function_calls)
        """
        pass

class LLMCache:
    def __init__(self, db_path="llm_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response_text TEXT,
                    function_calls TEXT
                )
            ''')
            conn.commit()

    def _generate_key(self, system_instruction: str, prompt: str, tools: list, response_schema, temperature: float, seed: int) -> str:
        # Create a deterministic representation of the inputs
        state = {
            "system_instruction": system_instruction,
            "prompt": prompt,
            "tools": json.dumps([t.__name__ if hasattr(t, '__name__') else str(t) for t in (tools or [])]),
            "schema": str(response_schema) if response_schema else None,
            "temperature": temperature,
            "seed": seed
        }
        state_str = json.dumps(state, sort_keys=True)
        return hashlib.sha256(state_str.encode("utf-8")).hexdigest()

    def get(self, system_instruction: str, prompt: str, tools: list, response_schema, temperature: float, seed: int):
        key = self._generate_key(system_instruction, prompt, tools, response_schema, temperature, seed)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT response_text, function_calls FROM cache WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                func_calls = json.loads(row[1]) if row[1] else []
                # Reconstruct function calls if necessary, here we just return the raw dicts or objects. 
                # Since we don't fully serialize complex protobufs, we assume mostly text responses for CAPS.
                return row[0], func_calls
        return None

    def set(self, system_instruction: str, prompt: str, tools: list, response_schema, temperature: float, seed: int, text: str, function_calls: list):
        key = self._generate_key(system_instruction, prompt, tools, response_schema, temperature, seed)
        func_calls_json = json.dumps([str(f) for f in function_calls]) if function_calls else json.dumps([])
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO cache (key, response_text, function_calls)
                VALUES (?, ?, ?)
            ''', (key, text, func_calls_json))
            conn.commit()


class GeminiLLMClient(BaseLLMClient):
    def __init__(self, use_cache: bool = True):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_id = settings.gemini_model
        self.use_cache = use_cache
        self.cache = LLMCache() if use_cache else None

    def generate_content(self, system_instruction: str, prompt: str, tools: list = None, response_schema=None, temperature: float = 0.0, seed: int = 42) -> tuple[str, list]:
        if self.use_cache:
            cached = self.cache.get(system_instruction, prompt, tools, response_schema, temperature, seed)
            if cached:
                logger.debug("llm_cache_hit", model=self.model_id)
                return cached[0], cached[1]

        config_kwargs = {
            "system_instruction": system_instruction,
            "temperature": temperature,
            # seed parameter might not be universally supported depending on SDK version, 
            # but we can pass it if we assume a modern SDK version.
        }
        
        # In the Google GenAI SDK (types), random seed might not be in the kwargs natively unless using the experimental SDK.
        # We will wrap it in a try-except or just pass if the prompt implies it's supported.
        # The prompt says: "Freeze every component... Temperature, Random seed". Let's add it.
        # We will manually inject if possible, or ignore if SDK rejects it, but we'll try.
        try:
            config_kwargs["seed"] = seed
        except Exception:
            pass
            
        if tools:
            config_kwargs["tools"] = tools
            
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
            
        config = types.GenerateContentConfig(**config_kwargs)
        
        logger.debug("llm_request", model=self.model_id, prompt_length=len(prompt))
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=config,
        )
        
        function_calls = []
        if response.function_calls:
            function_calls = response.function_calls
            
        if self.use_cache:
            self.cache.set(system_instruction, prompt, tools, response_schema, temperature, seed, response.text, function_calls)
            
        return response.text, function_calls

class MockLLMClient(BaseLLMClient):
    def __init__(self, deterministic_responses: dict = None):
        self.responses = deterministic_responses or {}
        
    def generate_content(self, system_instruction: str, prompt: str, tools: list = None, response_schema=None, temperature: float = 0.0, seed: int = 42) -> tuple[str, list]:
        logger.info("mock_llm_request", prompt_length=len(prompt))
        
        default_response = '{"mock": "response"}' if response_schema else "Mocked LLM Response."
        text = self.responses.get(prompt, default_response)
        
        return text, []

