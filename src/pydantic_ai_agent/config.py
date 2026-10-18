from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    test_database_url: str
    secret_key: SecretStr = Field(min_length=32)
    access_token_expire_minutes: int = 30


settings = Settings()  # type: ignore[call-arg]