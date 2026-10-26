from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    token_budget: int = 8192
    latency_budget_ms: int = 15000
    reward_lambda: float = 0.1
    policy_type: str = "always_send"
    logging_level: str = "INFO"
    sqlite_db_path: str = "sqlite:///caps_experiments.db"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    temperature: float = 0.0
    seed: int = 42

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
