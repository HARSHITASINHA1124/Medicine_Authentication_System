# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    device_id: str = "PI_001"
    database_url: str = "sqlite:///./local_scans.db"
    postgres_url: Optional[str] = None
    sync_interval_seconds: int = 300
    sync_retry_backoff_seconds: int = 30
    sync_retry_max_seconds: int = 300

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
