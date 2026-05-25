import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Determine the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    GROQ_API_KEY: str = Field(default="", env="GROQ_API_KEY")
    TAVILY_API_KEY: str = Field(default="", env="TAVILY_API_KEY")

    # Local Directory Configurations
    CHROMA_DB_PATH: str = Field(default=os.path.join(BASE_DIR, "data", "vector_db"), env="CHROMA_DB_PATH")
    PROCESSED_DIR: str = Field(default=os.path.join(BASE_DIR, "data", "processed"), env="PROCESSED_DIR")
    CHARTS_DIR: str = Field(default=os.path.join(BASE_DIR, "data", "processed", "charts"), env="CHARTS_DIR")
    RAW_DATA_DIR: str = Field(default=os.path.join(BASE_DIR, "data", "raw"), env="RAW_DATA_DIR")

    # Model Settings
    GROQ_TEXT_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Server Settings
    HOST: str = Field(default="127.0.0.1", env="HOST")
    PORT: int = Field(default=8000, env="PORT")

    def ensure_directories(self):
        """Ensures all necessary directories exist in the workspace."""
        for path in [self.CHROMA_DB_PATH, self.PROCESSED_DIR, self.CHARTS_DIR, self.RAW_DATA_DIR]:
            os.makedirs(path, exist_ok=True)

# Instantiate the global settings object and ensure folders are created immediately
settings = Settings()
settings.ensure_directories()
