from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/uistate"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str = "ui-state-capture"
    llm_provider: str = "huggingface"
    hf_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    headless: bool = True

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
