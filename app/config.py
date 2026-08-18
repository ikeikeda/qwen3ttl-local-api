from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server (0.0.0.0 so other LAN devices can reach the API)
    host: str = "0.0.0.0"
    port: int = 8000

    # Model — 1.7B CustomVoice fits ~8GB VRAM; good default for 12GB GPUs
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_path: str | None = None  # local dir overrides model_id when set

    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    # Empty string disables flash-attn (safer default across GPU generations)
    attn_implementation: str = ""

    # Optional API key; empty = no auth (LAN-only recommended)
    api_key: str = ""

    # Concurrency: one generation at a time avoids OOM on 12GB
    max_concurrent: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
