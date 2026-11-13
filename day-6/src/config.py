from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

   
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "API_KEY"),
    )
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    model: str = "gemini-3.8-flash"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    max_tokens_field: str = "max_tokens"

    @property
    def llm_configured(self) -> bool:
       
        return bool(self.api_key.strip())


settings = Settings()
