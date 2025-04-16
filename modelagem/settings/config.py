import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from datetime import datetime


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8"
    )

    MODEL_NAME: str
    _timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def timestamp(self) -> str:
        """
        Estrategia para evitar multiplas pastas a cada instancia de Settings
        """
        return self._timestamp

    @property
    def MODEL_DIR(self) -> Path:
        path = Path(f"database/models/{self.timestamp}_{self.MODEL_NAME}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def LOG_DIR(self) -> Path:
        path = self.MODEL_DIR / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def FT_DIR(self) -> Path:
        path = self.MODEL_DIR / "features"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def MAPPING_DIR(self) -> Path:
        path = self.MODEL_DIR / "mapping"
        path.mkdir(parents=True, exist_ok=True)
        return path
